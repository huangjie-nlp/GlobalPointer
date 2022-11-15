from inference.inference import Inference
from config.config import Config

config = Config()
inference = Inference(config)

while True:
    inp = input("请输入:")
    predict = inference.predict(inp)
    print(predict)