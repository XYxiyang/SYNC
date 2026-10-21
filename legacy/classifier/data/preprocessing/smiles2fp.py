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

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
    

def trainset():
    neg_smiles = read_smiles_from_txt("data/smiles/neg_smiles.txt")
    neg_fps = smiles2fp(neg_smiles, 0) 
    pos_smiles = read_smiles_from_txt("data/smiles/pos_smiles.txt")
    pos_fps = smiles2fp(pos_smiles, 1) 

    #torch.save(neg_fps + pos_fps, "data/processed/nonpher/170w.pt")
    #print(sum(d['y'] for d in neg_fps + pos_fps))
#
    #return 
    nt, nv, nte = random_split(neg_fps, [0.8, 0.1, 0.1])
    pt, pv, pte = random_split(pos_fps, [0.8, 0.1, 0.1])

    train, val, test = nt + pt, nv + pv, nte + pte

    os.makedirs("data/processed/fps/", exist_ok=True)
    torch.save(train, "data/processed/fps/train_170w.pt")
    torch.save(val, "data/processed/fps/val_170w.pt")
    torch.save(test, "data/processed/fps/test_170w.pt")

    print(sum(d['y'] for d in train))
    print(sum(d['y'] for d in val))
    print(sum(d['y'] for d in test))

def gasatest(idx: int):
    df = pd.read_csv(f'data/smiles/gasa_test{idx}.csv')
    os.makedirs("data/processed/nonpher/", exist_ok=True)

    fps = []
    for _, row in df.iterrows():
        smiles = row['smiles']
        label = row['labels']
        if label == 1:
            fps.append(smiles2fp([smiles], 0)[0])
        elif label == 0:
            fps.append(smiles2fp([smiles], 1)[0])

    torch.save(fps, f"data/processed/nonpher/gasa_test{idx}.pt")
    print(sum(d['y'] for d in fps))

def nonphertest():
    df = pd.read_csv(f'data/smiles/nonpher_test.csv')
    os.makedirs("data/processed/fps/", exist_ok=True)

    fps = []
    for idx, row in df.iterrows():
        smiles = row['smiles']
        label = row['synthetic_feasibility']
        if label == "infeasible":
            fps.append(smiles2fp([smiles], 0)[0])
        elif label == "feasible":
            fps.append(smiles2fp([smiles], 1)[0])

    torch.save(fps, f"data/processed/fps/nonpher_test.pt")
    print(sum(d['y'] for d in fps))

def sbdd(name: str):
    sbdd_smiles = read_smiles_from_txt(f"data/smiles/{name}.txt")
    fps = smiles2fp(sbdd_smiles, 1) 
    os.makedirs("data/sbdd/fps/", exist_ok=True)
    torch.save(fps, f"data/sbdd/fps/{name}.pt")
    print(sum(d['y'] for d in fps))

def IL17():
    # 筛选出class为0的行
    df = pd.read_csv('data/smiles/IL17_class.csv')
    fps = []
    for cls in [0, 1, 2]:
        filtered_df = df[df['class'] == cls]
        drug_smiles = filtered_df['drug_smiles'].to_list()
        print(drug_smiles)
        fps += smiles2fp(drug_smiles, cls) 
    
    print(sum(d['y'] for d in fps))
    print(len(fps))
    torch.save(fps, "data/processed/fps/IL17.pt")
    

def IL17_predict():
    mol = Chem.MolFromSmiles("COC[C@H](C1=C/C2=NC([C@@H](NC(=O)OCC3=NN(C)C=N3)C3CCC(F)(F)CC3)=CN2/N=C\\1)N1C[C@@H](C(F)(F)F)NC1=O")
    exit()
    df = pd.read_csv('data/smiles/predict.csv')
    fps = []
    df = df['drug_smiles'].to_list()
    fps = smiles2fp(df, 0)
    
    print(sum(d['y'] for d in fps))
    print(len(fps))
    torch.save(fps, "data/processed/fps/predict.pt")

def chemp():
    # 筛选出class为0的行
    sbdd_smiles = read_smiles_from_txt(f"data/smiles/9fl3.txt")
    fps = smiles2fp(sbdd_smiles, 1) 
    os.makedirs("data/sbdd/fps/", exist_ok=True)
    torch.save(fps, f"data/sbdd/fps/9fl3.pt")
    print(sum(d['y'] for d in fps))


if __name__ == "__main__":
    #models = ["3dsbdd", "d3fg", "decompdiff", "diffbp", "diffsbdd", "flag", 
    #          "graphbp", "ligan", "molcraft", "pocket2mol", "targetdiff", "voxbind",
    #          "enamine", "compas"]
    #
    models = ['compas_filter', 'coconut_5w']
    models = ['decomp_0510', 'decomp_dpo', 'decomp_origin']
    models = ['flowr', 'moldiff']
    models = ['neg_samples']
    models = ['8dyg_origin', '8dyg_dpo copy']
    for model in models:
        sbdd(model)
    #sbdd("targetdiff_dpo_1800")
    #IL17_predict()
    

    #gasatest(1)
    #gasatest(2)
    #gasatest(3)
    
