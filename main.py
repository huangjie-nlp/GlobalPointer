from framework.framework import Framework
from config.config import Config
import torch
import numpy as np

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    config = Config()
    fw = Framework(config)
    # print(config.test_fn)
    fw.train()
    print("*****************************************test***********************************************")
    fw.test()