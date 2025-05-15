import torch
import random
import pandas as pd
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from synsbdd.chem.mol import smiles2fp
from utils.misc import read_smiles_from_txt
from utils.misc import random_split
from synsbdd.data.featurize import tokenize_smiles

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
    

def trainset():
    neg_smiles = read_smiles_from_txt("data/smiles/gasa_neg.txt")
    neg_smiles = [{"smiles": tokenize_smiles(smiles), "y" :0} for smiles in neg_smiles]
    pos_smiles = read_smiles_from_txt("data/smiles/gasa_pos.txt")
    pos_smiles = [{"smiles": tokenize_smiles(smiles), "y" :1} for smiles in pos_smiles]

    nt, nv, nte = random_split(neg_smiles, [0.8, 0.1, 0.1])
    pt, pv, pte = random_split(pos_smiles, [0.8, 0.1, 0.1])

    train, val, test = nt + pt, nv + pv, nte + pte

    os.makedirs("data/processed/smiles/", exist_ok=True)
    torch.save(train, "data/processed/smiles/train_80w.pt")
    torch.save(val, "data/processed/smiles/val_80w.pt")
    torch.save(test, "data/processed/smiles/test_80w.pt")

    print(sum(d['y'] for d in train))
    print(sum(d['y'] for d in val))
    print(sum(d['y'] for d in test))

def gasatest(idx: int):
    df = pd.read_csv(f'data/smiles/gasa_test{idx}.csv')
    os.makedirs("data/processed/smiles/", exist_ok=True)

    fps = []
    for _, row in df.iterrows():
        smiles = row['smiles']
        label = row['labels']
        if label == 1:
            fps.append({"smiles": tokenize_smiles(smiles), "y" :0})
        elif label == 0:
            fps.append({"smiles": tokenize_smiles(smiles), "y" :1})

    torch.save(fps, f"data/processed/smiles/gasa_test{idx}.pt")
    print(sum(d['y'] for d in fps))

def nonphertest():
    df = pd.read_csv(f'data/smiles/nonpher_test.csv')
    os.makedirs("data/processed/smiles/", exist_ok=True)

    fps = []
    for idx, row in df.iterrows():
        smiles = row['smiles']
        label = row['synthetic_feasibility']
        if label == "infeasible":
            fps.append({"smiles": tokenize_smiles(smiles), "y" :0})
        elif label == "feasible":
            fps.append({"smiles": tokenize_smiles(smiles), "y" :1})

    torch.save(fps, f"data/processed/smiles/nonpher_test.pt")
    print(sum(d['y'] for d in fps))

def sbdd(name: str):
    sbdd_smiles = read_smiles_from_txt(f"data/smiles/{name}_smiles.txt")
    fps = [{"smiles": tokenize_smiles(smiles), "y" :1} for smiles in sbdd_smiles]
    os.makedirs("data/sbdd/smiles/", exist_ok=True)
    torch.save(fps, f"data/sbdd/smiles/{name}.pt")
    print(sum(d['y'] for d in fps))

if __name__ == "__main__":
    models = ["3dsbdd", "d3fg", "decompdiff", "diffbp", "diffsbdd", "flag", 
              "graphbp", "ligan", "molcraft", "pocket2mol", "targetdiff", "voxbind",
              "enamine", "compas"]
    
    models = ["enamine", "compas"]
    models = ["guide_0"]
    models = ["guide_1223"]
    
    for model in models:
        sbdd(model)
    #trainset()
    #nonphertest()
    #gasatest(1)
    #gasatest(2)
    #gasatest(3)