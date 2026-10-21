import torch
import torch.nn as nn

from .encoder import SMILESEncoder
from .output_head import ClassifierHead


class SynMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #self.encoder = SMILESEncoder(**cfg.encoder)
        self.head = ClassifierHead(**cfg.head)
        print(cfg.head)
        #self.d_modle = self.encoder.dim
    
    def forward(self, batch):

        x = self.head(batch.float())
        
        return x

