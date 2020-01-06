from sklearn.preprocessing import StandardScaler
import numpy as np


class ETFScaler(object):
    def __init__(self, etfs: np.array, max_pred_steps: int):
        self.scaler = StandardScaler()
        self.scaler.fit(etfs)
        self.max_pred_steps = max_pred_steps

    def transform(self, etfs: np.array, pred_steps: int):
        return self.scaler.transform(etfs), pred_steps/self.max_pred_steps