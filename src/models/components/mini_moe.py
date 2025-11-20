from torch import nn as nn
import torch
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel


class MiniMoEConfig(PretrainedConfig):
    model_type = "mini-moe"

    def __init__(
        self,
        vocab_size=32000,
        num_layers=12,
        dim=1024,
        rope_base=10000,
        num_attention_q_heads=16,
        num_attention_kv_heads=8,
        num_expert=8,
        top_k=4,
        qkv_bias=False,
        drop_rate=0.0,
        use_aux_loss=True,
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
        self.num_expert = num_expert
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss
        self.auto_map = {
            "AutoConfig": "mini_moe.MiniMoEConfig",
            "AutoModelForCausalLM": "mini_moe.MiniMoE",
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


class Expert(nn.Module):
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


class NoiseRouter(nn.Module):
    def __init__(self, num_expert, top_k, dim):
        super().__init__()
        self.gate = nn.Linear(dim, num_expert)
        self.noise_gate = nn.Linear(dim, num_expert)
        self.top_k = top_k

    def forward(self, x):
        gate = self.gate(x)
        logits = gate + torch.randn_like(gate) + self.noise_gate(x)

        top_k_val, top_k_ids = torch.topk(logits, k=self.top_k, dim=-1)
        scores = torch.full_like(logits, -torch.inf)
        scores.scatter_(dim=-1, index=top_k_ids, src=top_k_val)
        scores = scores.softmax(dim=-1)
        return scores, top_k_ids


class SparseMoe(nn.Module):
    def __init__(self, num_expert, top_k, dim, drop_rate, use_aux_loss=True):
        super().__init__()
        self.route = NoiseRouter(num_expert=num_expert, top_k=top_k, dim=dim)
        self.experts = nn.ModuleList(
            [Expert(dim=dim, drop_rate=drop_rate) for _ in range(num_expert)]
        )
        self.use_aux_loss = use_aux_loss
        self.num_expert = num_expert

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, dim = x.shape

        scores, indices = self.route(x)
        flatten_x = x.reshape(-1, dim)
        flatten_scores = scores.reshape(-1, scores.shape[-1])

        final_out = torch.zeros_like(flatten_x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            expert_mask = expert_mask.reshape(-1)
            if expert_mask.any():
                expert_in = flatten_x[expert_mask]
                expert_out = expert(expert_in)
                expert_weight = flatten_scores[expert_mask, i].unsqueeze(1)
                expert_out = expert_weight * expert_out

                final_out[expert_mask] += expert_out

        final_out = final_out.reshape(batch_size, seq_len, dim)

        if self.use_aux_loss:
            importance = flatten_scores.mean(dim=0).float()
            uniform = torch.full_like(
                importance, fill_value=1.0 / self.num_expert
            ).float()

            importance_log = (importance + 1e-8).log()
            uniform_log = uniform.log()

            aux_loss = F.kl_div(
                input=importance_log,
                target=uniform_log,
                log_target=True,
                reduction="sum",
            )
            return final_out, aux_loss
        return final_out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_attention_q_heads,
        num_attention_kv_heads,
        dim,
        qkv_bias,
        drop_rate,
        rope_base,
        num_expert,
        top_k,
        use_aux_loss,
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
        self.moe = SparseMoe(
            num_expert=num_expert,
            top_k=top_k,
            dim=dim,
            drop_rate=drop_rate,
            use_aux_loss=use_aux_loss,
        )
        self.use_aux_loss = use_aux_loss

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        hidden_state = self.moe(self.norm2(x))
        if self.use_aux_loss:
            x = x + hidden_state[0]
            aux_loss = hidden_state[1]

            return x, aux_loss
        else:
            x = x + hidden_state
            return x


class MiniMoE(PreTrainedModel):
    model_type = "mini-moe"
    config_class = MiniMoEConfig

    def __init__(self, config: MiniMoEConfig, pretrain_ckpt=None):
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
                    num_expert=config.num_expert,
                    top_k=config.top_k,
                    use_aux_loss=config.use_aux_loss,
                )
            )
        self.norm = RMSNorm(dim=config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self.init_weight)
        self.head.weight = self.embedding.weight
        self.use_aux_loss = config.use_aux_loss
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
        aux_loss = None
        for layer in self.layers:
            hidden_state = layer(hidden_state)
            if self.use_aux_loss:
                if aux_loss is None:
                    aux_loss = hidden_state[1]
                else:
                    aux_loss += hidden_state[1]
                hidden_state = hidden_state[0]

        hidden_state = self.norm(hidden_state)
        logits = self.head(hidden_state)
        if self.use_aux_loss:
            return logits, aux_loss
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
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
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
