import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class FingerprintDataset(Dataset):
    def __init__(self, path, num_stat_samples=10000):
        mm_path = os.path.join(path, 'dummy_db.mm')
        shape_path = os.path.join(path, 'dummy_db_shape.npy')

        assert os.path.exists(mm_path), f"Missing file: {mm_path}"
        assert os.path.exists(shape_path), f"Missing file: {shape_path}"

        self.shape = np.load(shape_path)
        self.memmap = np.memmap(mm_path, dtype=np.float32, mode='r', shape=tuple(self.shape))
        assert self.memmap.shape[1] == 128, "Expected fingerprint dimension of 128"

        # Estimate mean and std from a random subset
        total = self.memmap.shape[0]
        indices = np.random.choice(total, size=min(num_stat_samples, total), replace=False)
        sample = np.array([self.memmap[i] for i in indices])  # shape: [N, 128]

        self.mean = torch.from_numpy(sample.mean(axis=0)).float()
        self.std = torch.from_numpy(sample.std(axis=0)).float() + 1e-8  # prevent division by zero

    def __len__(self):
        return self.memmap.shape[0]

    def __getitem__(self, idx):
        fp = torch.from_numpy(self.memmap[idx].copy()).float()
        return (fp - self.mean) / self.std



class FingerprintDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=512, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = FingerprintDataset(self.data_path)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
