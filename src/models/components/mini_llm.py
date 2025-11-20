from torch import nn as nn
import torch
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel


class MiniLLMConfig(PretrainedConfig):
    model_type = "mini-llm"

    def __init__(
        self,
        vocab_size=32000,
        num_layers=12,
        dim=1024,
        rope_base=10000,
        num_attention_q_heads=64,
        num_attention_kv_heads=32,
        qkv_bias=True,
        drop_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dim = dim
        self.rope_base = rope_base
        self.num_attention_q_heads = num_attention_q_heads
        self.num_attention_kv_heads = num_attention_kv_heads
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.auto_map = {
            "AutoConfig": "mini_llm.MiniLLMConfig",
            "AutoModelForCausalLM": "mini_llm.MiniLLM",
        }


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        norm_x = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        output = self.weight * norm_x
        return output


class RopePositionEmbedding(nn.Module):
    def __init__(self, dim: int, base=10000):
        super().__init__()
        inv_freq = 1 / base ** (torch.arange(0, dim, 2).float() / dim)
        inv_freq = inv_freq.unsqueeze(0)
        self.register_buffer("inv_freq", inv_freq)

    def rotate_half(self, x: torch.Tensor):
        odd = x[..., 1::2]
        even = x[..., 0::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)

    def apply_rope(self, x: torch.Tensor):
        x_len = x.shape[2]
        t = torch.arange(0, x_len, device=x.device, dtype=torch.float32).unsqueeze(1)
        freq = t * self.inv_freq
        freq = torch.repeat_interleave(freq, repeats=2, dim=-1)[None, None, :, :]
        xf = x.float()
        y = xf * freq.cos() + self.rotate_half(xf) * freq.sin()
        return y.to(x.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return self.apply_rope(q), self.apply_rope(k)


class GroupQueryAttention(nn.Module):
    def __init__(
        self,
        num_attention_q_heads,
        num_attention_kv_heads,
        dim,
        qkv_bias,
        drop_rate,
        rope_base,
    ):
        super().__init__()

        self.head_dim = dim // num_attention_q_heads

        assert dim % num_attention_q_heads == 0, "dim 必须被 Q 头数整除"
        assert (
            num_attention_q_heads % num_attention_kv_heads == 0
        ), "Q头数必须是KV头数的整数倍"
        assert self.head_dim % 2 == 0, "head_dim 必须为偶数以应用 RoPE"

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(
            dim, self.head_dim * num_attention_kv_heads, bias=qkv_bias
        )
        self.v_proj = nn.Linear(
            dim, self.head_dim * num_attention_kv_heads, bias=qkv_bias
        )
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.num_repeat_kv = num_attention_q_heads // num_attention_kv_heads
        self.drop = nn.Dropout(drop_rate)

        self.position_embedding = RopePositionEmbedding(self.head_dim, rope_base)

        self.num_attention_q_heads = num_attention_q_heads
        self.num_attention_kv_heads = num_attention_kv_heads
        self.drop_rate = drop_rate

    def repeat_kv(self, k: torch.Tensor, v: torch.Tensor):
        k = k.repeat_interleave(self.num_repeat_kv, dim=1)
        v = v.repeat_interleave(self.num_repeat_kv, dim=1)
        return k, v

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, dim = x.shape
        Q = (
            self.q_proj(x)
            .reshape(batch_size, seq_len, self.num_attention_q_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(x)
            .reshape(batch_size, seq_len, self.num_attention_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(x)
            .reshape(batch_size, seq_len, self.num_attention_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        Q, K = self.position_embedding(Q, K)

        K, V = self.repeat_kv(K, V)

        out = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=self.drop_rate if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.out_proj(out)
        out = self.drop(out)
        return out


class FFN(nn.Module):
    def __init__(self, dim, drop_rate):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.ffn(x)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_attention_q_heads,
        num_attention_kv_heads,
        dim,
        qkv_bias,
        drop_rate,
        rope_base,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim=dim)
        self.attn = GroupQueryAttention(
            num_attention_q_heads=num_attention_q_heads,
            num_attention_kv_heads=num_attention_kv_heads,
            dim=dim,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            rope_base=rope_base,
        )
        self.norm2 = RMSNorm(dim=dim)
        self.ffn = FFN(dim=dim, drop_rate=drop_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MiniLLM(PreTrainedModel):
    model_type = "mini-llm"
    config_class = MiniLLMConfig

    def __init__(self, config: MiniLLMConfig, pretrain_ckpt=None):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([])
        for _ in range(config.num_layers):
            self.layers.append(
                DecoderLayer(
                    num_attention_q_heads=config.num_attention_q_heads,
                    num_attention_kv_heads=config.num_attention_kv_heads,
                    dim=config.dim,
                    qkv_bias=config.qkv_bias,
                    drop_rate=config.drop_rate,
                    rope_base=config.rope_base,
                )
            )
        self.norm = RMSNorm(dim=config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self.init_weight)
        self.head.weight = self.embedding.weight
        if pretrain_ckpt is not None:
            self.load_ckpt(pretrain_ckpt)

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[len("net._orig_mod.") :]
            new_state_dict[new_k] = v
        self.load_state_dict(new_state_dict, strict=True)
        print(f"load state dict from {ckpt_path}")

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            nn.init.constant_(m.weight, 1)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, input_ids: torch.Tensor):
        hidden_state = self.embedding(input_ids)
        for layer in self.layers:
            hidden_state = layer(hidden_state)

        hidden_state = self.norm(hidden_state)
        logits = self.head(hidden_state)
        return logits

    def top_k_sample(self, logits, top_k=5):

        weights, indices = torch.topk(logits, k=top_k, dim=-1)

        probs = torch.softmax(weights, dim=-1)
        chosssed_id = torch.multinomial(probs, num_samples=1)
        new_token = torch.gather(indices, dim=-1, index=chosssed_id)
        return new_token

    @torch.no_grad()
    def chat(self, conversations, tokenizer, max_new_token=256, top_k=5):
        ids = tokenizer.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=True
        )
        eos_ids = tokenizer.eos_token_id
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
        for _ in range(max_new_token):

            logits = self(input_ids)  # batch, seq_len, dim
            last_logits = logits[:, -1]  # batch, dim
            new_token = self.top_k_sample(last_logits, top_k=top_k)
            input_ids = torch.cat((input_ids, new_token), dim=-1)

            if new_token.detach()[0].cpu().item() == eos_ids:
                break

        output_id = input_ids.detach().cpu()[0].tolist()
        output_id = output_id[len(ids) :]
        answer = tokenizer.decode(output_id, skip_special_tokens=True)
        return answer
