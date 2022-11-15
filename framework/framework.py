
from models.models import GlobalPointer
# from models.GlobalPointer import GlobalPointer
from dataloader.dataloader import MyDataset, collate_fn
from torch.utils.data import DataLoader
import torch
import json
from loss.loss import multilabel_categorical_crossentropy
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from metics.metics import MetricsCalculator
from transformers import BertModel
metics = MetricsCalculator()

class Framework():
    def __init__(self, config):
        self.config = config
        with open(self.config.schema_fn, "r", encoding="utf-8") as f:
            self.id2label = json.load(f)[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):

        dataset = MyDataset(self.config, self.config.train_fn)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.config.batch_size,
                                collate_fn=collate_fn, pin_memory=True)

        dev_dataset = MyDataset(self.config, self.config.dev_fn)
        dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=self.config.batch_size,
                                collate_fn=collate_fn, pin_memory=True)

        def loss_fun(y_pred, y_true):
            """
            y_true:(batch_size, ent_type_size, seq_len, seq_len)
            y_pred:(batch_size, ent_type_size, seq_len, seq_len)
            """
            batch_size, ent_type_size = y_true.shape[:2]
            y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
            y_true = y_true.reshape(batch_size * ent_type_size, -1)
            loss = multilabel_categorical_crossentropy(y_pred, y_true)
            return loss

        model = GlobalPointer(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        decay_rate = self.config.decay_rate
        decay_steps = self.config.decay_steps
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)

        best_epoch = 0
        global_loss = 0
        global_step = 0
        best_f1_score = 0
        precision = 0
        reacll= 0
        loss_func = torch.nn.BCELoss()
        for epoch in range(1, self.config.epochs + 1):
            print('{}/{}'.format(epoch, self.config.epochs))
            for data in tqdm(dataloader):
                logits = model(data)
                optimizer.zero_grad()
                loss = loss_fun(logits, data["label_matrix"].to(self.device))

                loss.backward()
                optimizer.step()
                scheduler.step()
                global_loss += loss.item()
                if (global_step + 1) % self.config.step == 0:
                    print("epoch: {} global_step: {} global_loss: {:5.4f}".format(epoch, global_step, global_loss))
                    global_loss = 0
                global_step += 1

            # if epoch % 5 ==0:
            p, r, f1_score, predict = self.evaluate(model, dev_dataloader)
            if f1_score > best_f1_score:
                json.dump(predict, open(self.config.dev_result, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
                best_f1_score = f1_score
                precision = p
                reacll = r
                best_epoch = epoch
                print("best_epoch: {} precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f}".
                      format(best_epoch, precision, reacll, best_f1_score))
                print("save model ......")
                torch.save(model.state_dict(), self.config.save_model)
        print("best_epoch: {} precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f}".
              format(best_epoch, precision, reacll, best_f1_score))

    def evaluate(self, model, dataloader):
        model.eval()

        predict = []
        correct_num, predict_num, gold_num = 0, 0, 0

        with torch.no_grad():
            for data in tqdm(dataloader):
                logits = model(data).cpu()
                sentence = data["text"]
                token = data["token"]
                ners = data["ners"]
                for k, matrix in enumerate(logits):
                    # matrix.shape => [num_rel, seq_len, seq_len]
                    pred = []
                    for r, h, t in zip(*np.where(matrix > self.config.bar)):
                        e_type = self.id2label[str(r)]
                        if h <= t:
                            entity = ''.join(token[k][h: t+1])
                            resl = str(h-1) + '$' + str(t-1) + '$' + e_type + '@' + entity
                            pred.append(resl)
                            # break
                    lack = list(set(ners[k]) - set(pred))
                    new = list(set(pred) - set(ners[k]))
                    predict.append({"sentence": sentence, "gold": ners[k], "predict": pred,
                                    "lack": lack, "new": new})
                    gold_num += len(ners[k])
                    predict_num += len(pred)
                    correct_num += len(set(ners[k])&set(pred))
        print("predict_num: {}, gold_num: {} correct_num: {}".format(predict_num, gold_num, correct_num))
        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-1)
        f1_score = 2 * precision * recall / (precision +recall + 1e-10)
        model.train()
        print("precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f}".format(precision, recall, f1_score))
        return precision, recall, f1_score, predict

    def test(self):
        model = GlobalPointer(self.config)
        model.load_state_dict(torch.load(self.config.save_model, map_location=self.device))
        model.to(self.device)
        model.eval()

        predict = []
        correct_num, predict_num, gold_num = 0, 0, 0

        dataset = MyDataset(self.config, self.config.test_fn)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=self.config.batch_size,
                                collate_fn=collate_fn, pin_memory=True)

        with torch.no_grad():
            for data in tqdm(dataloader):
                logits = model(data).cpu()
                sentence = data["text"]
                token = data["token"]
                ners = data["ners"]
                for k, matrix in enumerate(logits):
                    # matrix.shape => [num_rel, seq_len, seq_len]
                    pred = []
                    for r, h, t in zip(*np.where(matrix > self.config.bar)):
                        e_type = self.id2label[str(r)]
                        if h <= t:
                            entity = ''.join(token[k][h: t+1])
                            resl = str(h-1) + '$' + str(t-1) + '$' + e_type + '@' + entity
                            pred.append(resl)
                            # break
                    lack = list(set(ners[k]) - set(pred))
                    new = list(set(pred) - set(ners[k]))
                    predict.append({"sentence": sentence, "gold": ners[k], "predict": pred,
                                    "lack": lack, "new": new})
                    gold_num += len(ners[k])
                    predict_num += len(pred)
                    correct_num += len(set(ners[k])&set(pred))
        print("predict_num: {}, gold_num: {} correct_num: {}".format(predict_num, gold_num, correct_num))
        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-1)
        f1_score = 2 * precision * recall / (precision +recall + 1e-10)
        model.train()
        print("precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f}".format(precision, recall, f1_score))
        json.dump(predict, open(self.config.test_result, "w", encoding="utf-8"), indent=4)
        return precision, recall, f1_score, predict
