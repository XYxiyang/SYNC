
from collections.abc import Iterable, Sequence
from functools import cache, cached_property, partial
from typing import Literal, overload

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit.Chem.Pharm2D import Generate as Generate2D
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm.auto import tqdm

from .base import Drawable
from utils.misc import read_smiles_from_txt


class Molecule(Drawable):
    def __init__(self, smiles: str) -> None:
        super().__init__()
        self.smiles = smiles.strip()

    
def smiles2fp(smiles:list, label:int):

    molecules = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x,2,512) for x in molecules]
    strings = [fp.ToBitString() for fp in fps]

    array = [{"smiles": smiles[i], "fp": torch.tensor([int(bit) for bit in strings[i]], dtype=int), "y": torch.tensor([label], dtype=torch.long)}  for i in range(len(smiles))]
    
    return array
 