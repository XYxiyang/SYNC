from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from synsbdd.data.featurize import tokenize_smiles, bond_features_simple
from utils.misc import random_split
from torch_geometric.data import Data


def sdf_to_graph(sdf_file, label, isfile=True):
    if isfile:
        supplier = Chem.SDMolSupplier(sdf_file, removeHs=True)
    else:
        supplier = sdf_file
        supplier = [Chem.RemoveAllHs(supplier[0])]
    graphs = []

    for mol in tqdm(supplier, desc="SDF to graph", total=len(supplier)):
        if mol is None:
            continue

        atoms = ''
        #for atom in mol.GetAtoms():
        #    if atom.GetIsAromatic():
        #        atoms += f"{atom.GetSymbol()}pi"
        #    else:
        #        atoms += atom.GetSymbol()
        for atom in mol.GetAtoms():
            atoms += atom.GetSymbol()
        print(atoms)
        atoms = torch.tensor(tokenize_smiles(atoms))
        #print(Chem.MolToSmiles(mol))
        #atoms = F.one_hot(atoms, 158)


        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            bond_attr = bond_features_simple(bond)

            edge_index.append([start, end])
            edge_index.append([end, start])
            edge_attr.append(bond_attr)
            edge_attr.append(bond_attr)  

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 转置为 [2, num_edges]
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        #print(edge_attr)

        #edge_attr = F.one_hot(edge_attr, 6)


        conformer = mol.GetConformer()
        positions = []
        for i in range(mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])
        positions = torch.tensor(positions, dtype=torch.float)  # 转换为张量


        graph = Data(
            x = atoms,            
            edge_index = edge_index, 
            edge_attr = edge_attr,   
            positions = positions,
            y =  torch.tensor([label], dtype=torch.long) 
        )
        graphs.append(graph)

    return graphs

def trainset():
    pos_graphs = sdf_to_graph("data/processed/sdf/80_pos.sdf", 1)
    neg_graphs = sdf_to_graph("data/processed/sdf/80_neg.sdf", 0)
    os.makedirs("data/processed/sdf/", exist_ok=True)

    nt, nv, nte = random_split(neg_graphs, [0.8, 0.1, 0.1])
    pt, pv, pte = random_split(pos_graphs, [0.8, 0.1, 0.1])
    train, val, test = nt + pt, nv + pv, nte + pte

    torch.save(train, "data/processed/sdf/train_80w.pt")
    torch.save(val, "data/processed/sdf/val_80w.pt")
    torch.save(test, "data/processed/sdf/test_80w.pt")
    print(sum(d['y'] for d in train))
    print(sum(d['y'] for d in val))
    print(sum(d['y'] for d in test))

def gasatest(idx: int):
    pos_graphs = sdf_to_graph(f"data/processed/sdf/gasa_test{idx}_pos.sdf", 1)
    neg_graphs = sdf_to_graph(f"data/processed/sdf/gasa_test{idx}_neg.sdf", 0)
    os.makedirs("data/processed/sdf/", exist_ok=True)


    torch.save(pos_graphs + neg_graphs, f"data/processed/sdf/gasa_test{idx}.pt")
    print(sum(d['y'] for d in pos_graphs))

def nonpher_test():
    pos_graphs = sdf_to_graph("data/processed/sdf/nonpher_test_pos.sdf", 1)
    neg_graphs = sdf_to_graph("data/processed/sdf/nonpher_test_neg.sdf", 0)
    os.makedirs("data/processed/sdf/", exist_ok=True)


    torch.save(pos_graphs + neg_graphs, "data/processed/sdf/nonpher_test.pt")
    print(sum(d['y'] for d in pos_graphs))

def sbdd(name: str):
    graphs = sdf_to_graph(f"data/sbdd/sdf/{name}.sdf", 1)
    os.makedirs("data/sbdd/sdf/", exist_ok=True)
    torch.save(graphs, f"data/sbdd/sdf/{name}.pt")
    print(sum(d['y'] for d in graphs))

def debugset():
    pos_graphs = sdf_to_graph("data/processed/sdf/output.sdf", 1)
    neg_graphs = sdf_to_graph("data/processed/sdf/output.sdf", 0)
    os.makedirs("data/processed/sdf/", exist_ok=True)

    

    #torch.save(pos_graphs + neg_graphs, "data/processed/sdf/debug_sbdd_1.pt")
    torch.save(pos_graphs+neg_graphs, "data/processed/sdf/debug_sbdd_8.pt")
    #torch.save(val, "data/processed/sdf/val_80w.pt")
    #torch.save(test, "data/processed/sdf/test_80w.pt")

if __name__ == "__main__":
    #models = ["3dsbdd", "d3fg", "decompdiff", "diffbp",
    #models =     ["diffsbdd", "flag", 
    #      "graphbp", "ligan", "molcraft", "pocket2mol", "targetdiff", "voxbind", "enamine", "compas"]
    models = ["enamine", "compas"]
    models = ["guide_1_aro"]
    models = ['targetdiff_guide_1', "targetdiff_regen_1"]
    models = ['decompdiff_guide']
    models = ["dpo_300", "dpo_400", "dpo_500", "dpo_700"]
    models = ["origin"]
    models = ["decompdiff_24_8"]


    
    
    #for model in models:
    #    sbdd(model)
    nonpher_test()
    gasatest(1)
    gasatest(2)
    gasatest(3)
    #debugset()
    trainset()
    sbdd("enamine")
    sbdd("compas")
