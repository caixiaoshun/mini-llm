"""
运行：
    streamlit run src/app/mini_llm_app.py
"""
import typing as T

import torch
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerFast
import hydra
import streamlit as st

# =============================
# 固定路径（不提供 UI 切换）
# =============================
CONFIG_PATH = "configs/model/mini-llm.yaml"
CKPT_PATH = "mini-llm-checkpoint.pth"
TOKENIZER_PATH = "checkpoints"

SYSTEM_PROMOT = """\
你是一个名为小顺AI机器人的智能助手。你的核心职责是提供帮助、解答问题和进行各种形式的交流。
你的主要特点和能力包括：
友善且乐于助人： 总是以积极、礼貌和有益的态度与用户互动。
知识渊博： 努力提供准确、全面和有用的信息。
富有创造力： 能够生成文本、提供创意建议，并根据需要生成图像。
适应性强： 能够理解并响应各种用户需求和对话情境。
记住自己的名字： 在对话中，如果被问及名字，请始终回答“我叫小顺AI机器人”。
在你的每一次回复中，请确保：
清晰地理解用户的意图。
提供相关且高质量的回答。
保持一致的AI机器人“小顺”的身份。
"""

# =============================
# Streamlit page setup
# =============================
st.set_page_config(page_title="Mini LLM Chat", page_icon="🤖", layout="centered")
st.title("🤖 Mini LLM Chat UI")
st.caption("使用固定的本地配置与 checkpoint，支持多轮对话与流式生成。首轮历史参与推理但不展示。")

# =============================
# Sidebar controls（不包含任何 checkpoint/tokenizer 切换控件）
# =============================
with st.sidebar:
    st.header("⚙️ 推理设置")
    device_choice = st.selectbox(
        "设备 (Device)",
        options=["auto", "cuda", "cpu"],
        index=0,
        help="auto 会优先使用 CUDA，如果不可用则回退到 CPU。",
    )
    dtype_choice = st.selectbox(
        "精度 (dtype)",
        options=["auto", "float16", "bfloat16", "float32"],
        index=0,
        help="仅在 CUDA 可用时建议使用 float16/bfloat16。",
    )
    max_new_tokens = st.slider("最大生成长度 (max_new_tokens)", 1, 1024, 256, 1)
    top_k = st.slider("Top‑K 采样", 1, 100, 5, 1)
    temperature = st.slider("温度 (temperature)", 0.1, 2.0, 1.0, 0.1)
    seed = st.number_input("随机种子 (seed)", value=42, min_value=0, step=1)
    truncate_ctx = st.number_input(
        "上下文截断 (tokens)",
        value=4096,
        min_value=512,
        step=512,
        help="为避免上下文过长导致显存溢出，保留最后 N 个 token 作为输入。",
    )
    clear_btn = st.button("🧹 清空对话")

# =============================
# Helpers
# =============================

def _select_device(device_choice: str) -> str:
    if device_choice == "cuda" and torch.cuda.is_available():
        return "cuda"
    if device_choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _select_dtype(device: str, dtype_choice: str) -> torch.dtype:
    if dtype_choice == "float16":
        return torch.float16
    if dtype_choice == "bfloat16":
        return torch.bfloat16
    if dtype_choice == "float32":
        return torch.float32
    # auto
    if device == "cuda":
        return torch.float16
    return torch.float32


@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(
    device_choice: str,
    dtype_choice: str,
):
    """加载模型与分词器（使用固定路径），并缓存。"""
    device = _select_device(device_choice)
    dtype = _select_dtype(device, dtype_choice)

    # Load model via Hydra/OmegaConf
    model_cfg = OmegaConf.load(CONFIG_PATH)["net"]
    model = hydra.utils.instantiate(model_cfg)
    # 期望模型提供 .load_ckpt(path)
    model.load_ckpt(CKPT_PATH)
    model.eval()

    # Place on device with dtype
    if device == "cuda":
        model = model.to(device=device, dtype=dtype)
    else:
        model = model.to(device=device)

    # Tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)

    # EOS handling
    eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_set = {eos_ids}
    elif isinstance(eos_ids, (list, tuple, set)):
        eos_set = set(eos_ids)
    else:
        eos_set = set()

    meta = {
        "device": device,
        "dtype": str(dtype),
        "eos_set": eos_set,
        "has_eos": len(eos_set) > 0,
    }
    return model, tokenizer, meta


def top_k_sample(logits: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Top‑K 采样。Args: logits=[batch, vocab]; 返回 new_token ids=[batch,1]"""
    vocab = logits.size(-1)
    k = max(1, min(k, vocab))
    values, indices = torch.topk(logits, k=k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    chosen = torch.multinomial(probs, num_samples=1)
    new_token = torch.gather(indices, dim=-1, index=chosen)
    return new_token


def generate_stream(
    model,
    tokenizer: PreTrainedTokenizerFast,
    meta: dict,
    messages: list[dict],
    *,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    seed: int,
    truncate_ctx: int,
) -> T.Iterator[str]:
    """增量生成字符串（流式）。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = meta["device"]
    eos_set = meta["eos_set"]

    # chat template 编码
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )

    # 截断上下文
    if truncate_ctx and len(ids) > truncate_ctx:
        ids = ids[-truncate_ctx:]

    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    prev_text = ""

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)  # [B, S, V] 或 [B, V]

        last_logits = logits[:, -1, :] if logits.dim() == 3 else logits

        if temperature and temperature != 1.0:
            last_logits = last_logits / temperature

        new_token = top_k_sample(last_logits, k=top_k)
        input_ids = torch.cat((input_ids, new_token), dim=-1)

        # EOS 停止
        if new_token[0, 0].item() in eos_set:
            break

        # 计算增量
        out_tokens = input_ids[0].detach().cpu().tolist()[len(ids):]
        full_text = tokenizer.decode(out_tokens, skip_special_tokens=True)
        delta = full_text[len(prev_text):]
        prev_text = full_text
        if delta:
            yield delta

    # 最终 flush
    out_tokens = input_ids[0].detach().cpu().tolist()[len(ids):]
    final_text = tokenizer.decode(out_tokens, skip_special_tokens=True)
    tail = final_text[len(prev_text):]
    if tail:
        yield tail


# =============================
# 初始化 / 加载资源（固定路径）
# =============================
model, tokenizer, meta = load_model_and_tokenizer(
    device_choice=device_choice,
    dtype_choice=dtype_choice,
)

with st.sidebar:
    st.write(
        f"**Device:** `{meta['device']}`  ·  **Dtype:** `{meta['dtype']}`  ·  **EOS:** {'✅' if meta['has_eos'] else '⚠️ 无'}"
    )

# =============================
# 多轮对话的 Session State（包含隐藏首轮）
# =============================
if "messages" not in st.session_state:
    # 首次进入：写入隐藏首轮，并把渲染起点设置为 len(messages)
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMOT},
    ]
    st.session_state.render_start_index = len(st.session_state.messages)  # 不显示之前的历史

# 清空对话：恢复为仅隐藏首轮
if clear_btn:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMOT},
    ]
    st.session_state.render_start_index = len(st.session_state.messages)

# 渲染（跳过隐藏的前两条）
for m in st.session_state.messages[st.session_state.render_start_index:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 输入框
user_prompt = st.chat_input("输入你的问题，例如：1+1等于几？")

if user_prompt:
    # 1) 记录用户消息（完整历史，包括隐藏首轮）
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2) 流式生成助手回复（基于完整多轮历史）
    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc_text = ""
        try:
            stream = generate_stream(
                model,
                tokenizer,
                meta,
                st.session_state.messages,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                temperature=temperature,
                seed=seed,
                truncate_ctx=int(truncate_ctx),
            )
            for chunk in stream:
                acc_text += chunk
                placeholder.markdown(acc_text)
        except Exception as e:
            st.error(f"推理时发生异常：{e}")
        finally:
            if acc_text.strip():
                st.session_state.messages.append({"role": "assistant", "content": acc_text})
