from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import torch
import json


class SFTDataset(Dataset):
    def __init__(
        self,
        tokenizer_path="checkpoints",
        sft_data_path="data/train_3.5M_CN.json",
        max_len=512,
    ):
        super().__init__()
        self.tokenizer: PreTrainedTokenizerFast = (
            PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        )
        self.sft_data_path = sft_data_path
        with open(self.sft_data_path, mode="r") as f:
            self.lines = f.readlines()
        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def norm_conversation(self, conversations):
        new_conversations = []

        for conversation in conversations:
            new_conversation = {}
            role = "user" if conversation["from"] == "human" else conversation["from"]
            new_conversation["role"] = role
            new_conversation["content"] = conversation["value"]
            new_conversations.append(new_conversation)

        return new_conversations

    def __getitem__(self, index):
        line = self.lines[index]
        conversations = json.loads(line)['conversations']
        conversations = self.norm_conversation(conversations)

        tokenized = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            padding="max_length",
            truncation=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            max_length=self.max_len
        )

        ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        assistant_masks = tokenized["assistant_masks"]


        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        assistant_masks = torch.tensor(assistant_masks[1:], dtype=torch.bool)
        attention_mask = torch.tensor(attention_mask[1:], dtype=torch.bool)
        
        labels[~assistant_masks] = -100
        labels[~attention_mask] = -100 

        return {"input_ids": input_ids, "labels": labels}

if __name__ == "__main__":
    dataset = SFTDataset()
    item = dataset[0]
    print(item["input_ids"].shape, item["labels"].shape)