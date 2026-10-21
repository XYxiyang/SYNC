import torch
import torch.nn as nn

from .encoder import GraphEncoder
from .output_head import ClassifierHead


class Syn2d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = GraphEncoder(**cfg.encoder)
        self.head = ClassifierHead(**cfg.head)
        #nn.init.xavier_uniform_(self.model.layer.weight)
    
    def forward(self, atoms, bonds, mask):
        node, _ = self.encoder(atoms, bonds, mask)

        x = (node * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.head(x)

