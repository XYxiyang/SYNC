'''
Modified form https://github.com/luost26/ChemProjector.git
'''
import abc
import dataclasses
from collections.abc import Callable
#from typing import TypeAlias

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def _SimpleMLP(
    dim_in: int,
    dim_out: int,
    dim_hidden: int,
    num_layers: int = 3,
) :
    num_intermediate = num_layers - 2
    layers = [
        nn.Linear(dim_in, dim_hidden),
        nn.ReLU(),
    ]
    for _ in range(num_intermediate):
        layers.append(nn.Linear(dim_hidden, dim_hidden))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dim_hidden, dim_out))
    return nn.Sequential(*layers)


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dim_hidden: int = None):
        super().__init__()
        dim_hidden = dim_hidden or d_model * 2
        self.mlp = _SimpleMLP(d_model, num_classes, dim_hidden=dim_hidden)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)
