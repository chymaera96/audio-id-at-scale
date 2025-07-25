import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class FingerprintDataset(Dataset):
    def __init__(self, npy_path):
        """
        Args:
            npy_path: Path to .npy file (fingerprints shape: [N, 128])
        """
        assert os.path.exists(npy_path), f"File not found: {npy_path}"
        self.memmap = np.load(npy_path, mmap_mode='r')
        assert self.memmap.ndim == 2 and self.memmap.shape[1] == 128, "Expected shape (N, 128)"

    def __len__(self):
        return self.memmap.shape[0]

    def __getitem__(self, idx):
        fp = self.memmap[idx]  
        return torch.from_numpy(fp).float()


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
