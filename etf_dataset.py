import torch
from torch.utils.data import Dataset


class ETFDataset(Dataset):
    def __init__(self, etfs, seq_len, max_pred_steps, scaler):
        self.etfs = etfs
        self.seq_len = seq_len
        self.max_pred_steps = max_pred_steps
        self.scaler = scaler

        self.block_size = self.etfs.shape[0] - self.max_pred_steps - self.seq_len + 1
        self.length = self.block_size * (self.max_pred_steps+1)
        self.dtype = torch.float32
        assert self.block_size > 0

    def __getitem__(self, index):
        i = index // self.block_size
        j = index % self.block_size
        lcond, gcond = self.scaler.transform(self.etfs[j:j+self.seq_len], i/self.max_pred_steps)
        target, _ = self.scaler.transform(self.etfs[j+i:j+i+self.seq_len], i/self.max_pred_steps)
        lcond = torch.tensor(lcond, dtype=self.dtype)
        gcond = torch.tensor(gcond, dtype=self.dtype).view((1,))
        target = torch.tensor(target, dtype=self.dtype)
        return lcond, gcond, target

    def __len__(self):
        return self.length