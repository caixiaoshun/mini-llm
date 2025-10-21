import json
from tokenizers import (
    models,
    Tokenizer,
    pre_tokenizers,
    normalizers,
    decoders,
    trainers,
)
from transformers import PreTrainedTokenizerFast

FILE_NAME = "data/mobvoi_seq_monkey_general_open_corpus.jsonl"
MAX_LINES = 2000000
VOCAB_SIZE = 32000


def generate_text(filename="data/mobvoi_seq_monkey_general_open_corpus.jsonl"):
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAX_LINES:
                break
            item = json.loads(line)
            yield item["text"]


def main():
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<|im_start|>", "<|im_end|>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    tokenizer.train_from_iterator(generate_text(), trainer, length=MAX_LINES)

    chat_template = """\
{% for message in messages %}
{% if message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{% generation %}{{ message['content'] }}<|im_end|>{% endgeneration %}
{% endif %}
{% endfor %}
{% if  add_generation_prompt %}
<|im_start|>assistant
{% endif %}
"""

    tokenizer_fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        padding_side="right",
        truncation_side="right",
        chat_template =chat_template,
        unk_token="<unk>",
        pad_token="<|im_end|>",
    )

    tokenizer_fast.save_pretrained("checkpoints")



if __name__ == "__main__":
    main()
