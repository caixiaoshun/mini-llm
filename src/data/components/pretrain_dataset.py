from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import torch
import json

class PretrainDataset(Dataset):
    def __init__(
        self,
        tokenizer_path="checkpoints",
        pretrain_data_path="data/mobvoi_seq_monkey_general_open_corpus.jsonl",
        max_len=512
    ):
        super().__init__()
        self.tokenizer:PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.pretrain_data_path = pretrain_data_path
        with open(self.pretrain_data_path, mode="r") as f:
            self.lines = f.readlines()
        self.max_len = max_len
    
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        line = json.loads(line)["text"]
        line = f"{self.tokenizer.bos_token}{line}"
        tokenized = self.tokenizer(line, max_length=self.max_len, add_special_tokens=False, truncation=True, padding="max_length")
        ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(attention_mask[1:], dtype=torch.bool)
        labels[~loss_mask] = -100
        
        return {
            "input_ids" : input_ids,
            "labels" : labels
        }
