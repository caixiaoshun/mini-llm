from transformers import PreTrainedTokenizerFast

# 1) 基础对话（最小可用）
messages_basic = [
    {"role": "system", "content": "你是一个乐于助人的AI助手。"},
    {"role": "user", "content": "用一句话概括 Transformer 的核心思想。"},
    {"role": "assistant", "content": "通过自注意力在序列内建立全局依赖，从而并行建模上下文。"}
]

# 2) 多语言/Emoji/标点规范化（触发 NFKC，含表情）
messages_multilingual = [
    {"role": "system", "content": "回复时保持简洁。"},
    {"role": "user", "content": "Translate to Chinese: 'The café opens at 9:00 a.m. — see you there! 😊'"},
    {"role": "assistant", "content": "咖啡馆早上 9:00 开门——到时见！😊"}
]

# 3) 代码块 + JSON + 缩进与换行
messages_code = [
    {"role": "system", "content": "你可以嵌入代码片段。"},
    {"role": "user", "content": "把下面的 Python 函数改成尾递归并给出复杂度分析：\n```python\ndef fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)\n```\n也给我一个 JSON 示例。"},
    {"role": "assistant", "content": "实现与分析：\n```python\ndef fib_tr(n, a=0, b=1):\n    if n == 0:\n        return a\n    if n == 1:\n        return b\n    return fib_tr(n-1, b, a+b)\n```\n时间复杂度 O(n)，空间复杂度 O(1)（若改写为循环）。\n示例 JSON：\n```json\n{\"input\": 10, \"output\": 55}\n```"}
]

# 4) 空白字符原样保留（行首制表符、前后空格、多空行）
messages_whitespace = [
    {"role": "system", "content": "保持原样输出（包括空格与缩进）。"},
    {"role": "user", "content": "以下文本前后有空格与制表符，请原样返回：\n\t  leading tabs & spaces  \nline2\n\nline4 (two newlines above)\n"},
    {"role": "assistant", "content": "\t  leading tabs & spaces  \nline2\n\nline4 (two newlines above)\n"}
]

# 5) 特殊标记/URL/邮箱/区域旗帜与肤色Emoji（鲁棒性）
messages_special_tokens = [
    {"role": "system", "content": "测试包含特殊标记的文本。"},
    {"role": "user", "content": "这一行包含特殊串：<|im_start|> 和 <|im_end|>；还有 URL https://example.com?a=1&b=2 以及邮箱 a+b@example.co.uk 🇨🇳👍🏽"},
    {"role": "assistant", "content": "收到。我会把这些字面量当作普通文本来处理（如果你的流程需要如此）。"}
]

# 6) CJK 混排 + 全角/半角（触发 NFKC 规范化）
messages_cjk_mix = [
    {"role": "system", "content": "测试多脚本：中日韩混排。"},
    {"role": "user", "content": "混排示例：中文繁體「測試」、かな（ひらがな）、カタカナ、漢字、한글 Hangul；数字１２３４５６７８９０ vs 1234567890；标点：，。、；？！"},
    {"role": "assistant", "content": "👌 已读。请留意 NFKC 对全角/半角的统一，例如：ｅｘａｍｐｌｅ → example。"}
]



tok = PreTrainedTokenizerFast.from_pretrained("checkpoints")

for name, msgs in [
    ("basic", messages_basic),
    ("multilingual", messages_multilingual),
    ("code", messages_code),
    ("whitespace", messages_whitespace),
    ("special_tokens", messages_special_tokens),
    ("cjk_mix", messages_cjk_mix),
]:
    print(f"\n=== {name} (string) ===")
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(s)

    print(f"=== {name} (ids) ===")
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    print(ids[:64], "...len=", len(ids))

tok.save_pretrained("checkpoints2")
