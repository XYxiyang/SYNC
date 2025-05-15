import torch
import torch.nn as nn

from .graphormer.graphormer import Graphormer
from .output_head import ClassifierHead


class Syn2dG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Graphormer(**cfg.encoder)
        self.head = ClassifierHead(**cfg.head)
        #nn.init.xavier_uniform_(self.model.layer.weight)
    
    def forward(self, data):
        node = self.encoder(data)

        #x = (node * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.head(node)

