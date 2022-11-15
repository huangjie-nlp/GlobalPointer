import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import json
import numpy as np

class MyDataset(Dataset):
    def __init__(self, config, fn):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, do_lower_case=False)
        with open(fn, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        with open(self.config.schema_fn, "r", encoding="utf-8") as f:
            self.label2id = json.load(f)[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        json_data = self.data[idx]
        text = json_data["text"]
        ners = json_data["entity"]

        # token = ["[CLS]"] + self.tokenizer.tokenize(text) + ["[SEP]"]
        token = ["[CLS]"] + list(text) + ["[SEP]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        token_len = len(token)
        mask = [1] * token_len
        input_ids = np.array(token_ids)
        mask = np.array(mask)
        label_matrix = np.zeros((self.config.num_type, token_len, token_len))
        for i in ners:
            start, end, e = i.split("$")
            e_type, entity = e.split("@")
            # label_matrix[self.label2id[e_type]][int(start) + 1][int(end) + 1] = 1
            label_matrix[self.label2id[e_type], int(start) + 1, int(end) + 1] = 1
        label_matrix = np.array(label_matrix)

        return text, ners, token, token_len, input_ids, mask, label_matrix

def collate_fn(batch):
    text, ners, token, token_len, input_ids, mask, label_matrix = zip(*batch)
    cur_batch = len(batch)
    max_text_len = max(token_len)

    batch_input_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_mask = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_label_matrix = torch.Tensor(cur_batch, 8, max_text_len, max_text_len).zero_()

    for i in range(cur_batch):
        batch_input_ids[i, :token_len[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_mask[i, :token_len[i]].copy_(torch.from_numpy(mask[i]))
        batch_label_matrix[i, :, :token_len[i], :token_len[i]].copy_(torch.from_numpy(label_matrix[i]))

    return {"text": text,
            "ners": ners,
            "token": token,
            "input_ids": batch_input_ids,
            "mask": batch_mask,
            "label_matrix": batch_label_matrix}

if __name__ == '__main__':
    from config.config import Config
    from torch.utils.data import DataLoader
    config = Config()

    dataset = MyDataset(config, config.train_fn)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    n = 0
    for data in dataloader:
        print("*"*50)
        print(torch.sum(data["label_matrix"]))
        print(data["label_matrix"].shape)
        if n == 10:
            exit()
        n += 1