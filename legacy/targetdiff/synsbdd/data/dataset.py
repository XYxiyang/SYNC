import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class Syn1dDataset(Dataset):
    def __init__(self, path: str, max_smiles_len: int = 192) -> None:
        super().__init__()
        self.max_smiles_len = max_smiles_len
        self.data = torch.load(path)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) :
        #print(self.data[idx])
        data = self.data[idx]
        if data['smiles'].size(0) < self.max_smiles_len:
            data['smiles'] = torch.cat([data['smiles'], torch.zeros(self.max_smiles_len - data['smiles'].size(0))])
        else:
            data['smiles'] = data['smiles'][:self.max_smiles_len]
        return data
    
class SynFpDataset(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.data = torch.load(path)


    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) :
        return self.data[idx]

class Syn2dDataset(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.data = torch.load(path)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) :
        return self.data[idx]

class Syn3dDataset(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.data = torch.load(path)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) :
        return self.data[idx]

if __name__ == "__main__":

    dataset = Syn1dDataset('data/processed/smiles/train_170w.pt')
    max_len = 0
    print(len(dataset))

