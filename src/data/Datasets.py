import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class AverageSequenceDataModule(pl.LightningDataModule): 
    def __init__(self, 
                 data, 
                 labels, 
                 train_batch_size=32, 
                 val_batch_size=32, 
                 test_batch_size=9999, 
                 split_ratio=(0.7, 0.2, 0.1)
                ) -> None: 
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.split_ratio = split_ratio
        self.data = data
        self.labels = labels

    def setup(self, stage: str): 
        self.data = [torch.mean(seq, 0) for seq in self.data]
        self.data = torch.utils.data.TensorDataset(torch.vstack(self.data), self.labels)
        
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(self.data, self.split_ratio)

    def train_dataloader(self): 
        return DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self): 
        return DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=True)
        
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size=self.test_batch_size)