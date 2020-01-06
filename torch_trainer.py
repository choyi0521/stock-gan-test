import torch
import numpy as np
import os
import random


class TorchTrainer(object):
    def __init__(self):
        # set random seed
        self.set_random_seed()

        # cuda setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        pass

    def validate(self):
        pass

    def set_random_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

