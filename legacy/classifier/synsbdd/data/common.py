import enum
from collections.abc import Sequence
from typing import TypedDict

import torch
from synsbdd.chem.mol import Molecule

class Syn1dBatch(TypedDict, total=False):
    smiles: torch.Tensor
    
class Syn1dData(TypedDict, total=False):
    smiles: torch.Tensor

def create_data(
    product: Molecule,
):
    data: "Syn1dData" = {
        "smiles": product.tokenize_csmiles(),
    }
    return data
