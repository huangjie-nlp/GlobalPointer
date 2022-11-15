import json
from models.models import GlobalPointer
import torch
from transformers import BertTokenizer
import numpy as np

class Inference(object):
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(self.config.schema_fn, "r", encoding="utf-8") as f:
            self.id2label = json.load(f)[1]
        self.model = GlobalPointer(self.config)
        self.model.load_state_dict(torch.load(self.config.save_model, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sentence, bar=0.5):
        data = self.processing(sentence)
        logits = self.model(data).cpu()
        pred = []
        token = data["token"]
        for matrix in logits:
            for e_type, h, t in zip(*np.where(matrix > bar)):
                entity_type = self.id2label[str(e_type)]
                if t >= h:
                    pred.append(str(h-1)+"/"+str(t-1)+"/"+entity_type+"@"+"".join(token[int(h): int(t)+1]))
        return {"sentence": sentence, "predict": pred}

    def processing(self, sentence):
        token = ['[CLS]'] + list(sentence) + ['[SEP]']
        token2id = self.tokenizer.convert_tokens_to_ids(token)
        mask = [1] * len(token)

        input_ids = torch.LongTensor([token2id])
        attention_mask = torch.LongTensor([mask])
        return {"input_ids": input_ids,
                "mask": attention_mask,
                "token": token}
