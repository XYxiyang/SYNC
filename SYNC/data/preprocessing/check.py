import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import torch



#graphs1 = torch.load("data/processed/sdf/enamine.pt")

supplier = Chem.SDMolSupplier("data/processed/sdf/gasa_test1_pos.sdf", removeHs=True)
print(len(supplier))
supplier = Chem.SDMolSupplier("data/processed/sdf/gasa_test2_pos.sdf", removeHs=True)
print(len(supplier))
supplier = Chem.SDMolSupplier("data/processed/sdf/gasa_test3_pos.sdf", removeHs=True)
print(len(supplier))
supplier = Chem.SDMolSupplier("data/processed/sdf/gasa_test1_neg.sdf", removeHs=True)
print(len(supplier))
supplier = Chem.SDMolSupplier("data/processed/sdf/gasa_test2_neg.sdf", removeHs=True)
print(len(supplier))

supplier = Chem.SDMolSupplier("data/processed/sdf/gasa_test3_neg.sdf", removeHs=True)

print(len(supplier))

