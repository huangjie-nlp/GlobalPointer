
class Config():
    def __init__(self):
        self.dataset_name = "public1"
        self.num_type = 8
        self.hidden_size = 64
        self.RoPE = True
        self.train_fn = "./dataset/" + self.dataset_name + '/' + "train_data.json"
        self.dev_fn = "./dataset/" + self.dataset_name + "/" + "dev_data.json"
        self.test_fn = "./dataset/" + self.dataset_name + "/" + "test_data.json"
        self.schema_fn = "./dataset/" + self.dataset_name + "/" + "schema.json"
        self.bert_path = "bert-base-chinese"
        self.dev_result = "dev_result/dev_data.json"
        self.test_result = "test_result/test_data.json"
        self.batch_size = 16
        self.learning_rate = 1e-5
        self.epochs = 30
        self.step = 500
        self.bar = 0.5
        self.save_model = "checkpoint/globalpointer.pt"
        self.decay_rate = 0.999
        self.decay_steps = 100

