import os
import yaml
import torch
import numpy as np
import random

from easydict import EasyDict



class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream):
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.load(f, Loader))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

def read_smiles_from_txt(file_path):
    smiles_list = []
    with open(file_path, 'r') as file:
        for line in file:
            smiles = line.strip()
            if smiles:  
                smiles_list.append(smiles)
    return smiles_list

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def random_split(data:list, ratio:list):
    length = len(data)
    indices = list(range(length))
    random.shuffle(indices)
    if len(ratio) == 2:
        train_indices = indices[:int(ratio[0]*length)]
        val_indices = indices[int(ratio[0]*length):]
        return [data[i] for i in train_indices], [data[i] for i in val_indices]
    elif len(ratio) == 3:
        train_indices = indices[:int(0.8*length)]
        val_indices = indices[int(0.8*length):int(0.9*length)]
        test_indices = indices[int(0.9*length):]
        return [data[i] for i in train_indices], [data[i] for i in val_indices], [data[i] for i in test_indices]
    
def random_split_5index(data:list, ratio:list):
    length = int(len(data) / 5)
    indices = list(range(length))
    random.shuffle(indices)
    if len(ratio) == 2:
        train_indices = indices[:int(ratio[0]*length)]
        val_indices = indices[int(ratio[0]*length):]
        return [data[i] for i in train_indices], [data[i] for i in val_indices]
    elif len(ratio) == 3:
        train_indices = indices[:int(0.8*length)]
        val_indices = indices[int(0.8*length):int(0.9*length)]
        test_indices = indices[int(0.9*length):]
        train_data = [x for i in train_indices for x in data[5*i:5*i+5]]
        val_data   = [x for i in val_indices   for x in data[5*i:5*i+5]]
        test_data  = [x for i in test_indices  for x in data[5*i:5*i+5]]
        return train_data, val_data, test_data

        return [data[5*i: 5*i+5] for i in train_indices], [data[5*i: 5*i+5] for i in val_indices], [data[5*i: 5*i+5] for i in test_indices]