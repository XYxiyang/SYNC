import torch
import torch.nn as nn

from .encoder import SMILESEncoder
from .output_head import ClassifierHead


class Syn1d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = SMILESEncoder(**cfg.encoder)
        self.head = ClassifierHead(**cfg.head)
        self.d_modle = self.encoder.dim
        #nn.init.xavier_uniform_(self.model.layer.weight)
    
    def forward(self, batch):
        x, mask = self.encoder(batch)
        x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.head(x)

