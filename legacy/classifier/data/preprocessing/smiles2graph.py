import os
import torch
import pyximport
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data

from synsbdd.data.featurize import tokenize_smiles, bond_features_simple
from utils.misc import read_smiles_from_txt
from utils.misc import random_split



pyximport.install(setup_args={"include_dirs": np.get_include()})

import algos



def conditional_tqdm(iterable):
    if len(iterable) > 1:
        return tqdm(iterable, desc="Converting SMILES to Graphs", unit="graph")
    return iterable 
   
def smiles_to_graph(smiles_list: list, label: int) -> list: 

    graphs = []

    for smiles in conditional_tqdm(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)

            num_atoms = mol.GetNumAtoms()

            atoms = ''
            for atom in mol.GetAtoms():
                atoms+=(atom.GetSymbol())


            atom_features = tokenize_smiles(atoms)
            atom_tensor = torch.tensor(atom_features)

            bond_matrix = torch.zeros([len(atoms), len(atoms)], dtype=torch.long)
            for bond in mol.GetBonds():
                
                i = bond.GetBeginAtomIdx() 
                j = bond.GetEndAtomIdx()
                bond_matrix[i, j] = bond_features_simple(bond)
                bond_matrix[j, i] = bond_matrix[i, j] 

            #edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # 转置以适应torch_geometric格式
            #edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)  # 转为列向量形式
            #graphs.append(Data(atom_tensor, bond_matrix=bond_matrix, y=torch.tensor([label], dtype=torch.int)))
            graphs.append({'atoms': atom_tensor, "bonds": bond_matrix, "y": torch.tensor([label], dtype=torch.long)})
        except Exception as e:
            print(f"Error converting SMILES to Graph: {smiles}")
            exit()
    
    return graphs

def smiles_to_graph_graphormer(smiles_list: list, label: int) -> list: 

    graphs = []

    for smiles in conditional_tqdm(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)

            num_atoms = mol.GetNumAtoms()

            atoms = ''
            for atom in mol.GetAtoms():
                atoms+=(atom.GetSymbol())


            atom_features = tokenize_smiles(atoms)
            atom_tensor = torch.tensor(atom_features).unsqueeze(1)

            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                edge_index.append([start, end])
                edge_index.append([end, start])
                bond_attr = bond_features_simple(bond)
                edge_attr.append(bond_attr)
                edge_attr.append(bond_attr)

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 转置以符合PyTorch Geometric格式
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)

            graphs.append(Data(x=atom_tensor, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.long)))

        except Exception as e:
            print(f"Error converting SMILES to Graph: {smiles}")
            exit()
    
    return graphs

def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):

    num_virtual_tokens = 1
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x

    if edge_attr is None:
        edge_attr = torch.zeros((edge_index.shape[1]), dtype=torch.long)

    N = x.size(0)
    if N == 1: # single atom, not able to transfer to graph, just C-C to be a simple graph adn synthesiable
        edge_attr = torch.tensor([1], dtype=torch.long)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.tensor([[7], [7]], dtype=torch.long)
        N = 2

    x = convert_to_single_emb(x)  

    # node adj matrix [N, N] bool
    adj_orig = torch.zeros([N, N], dtype=torch.bool)
    adj_orig[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here

    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )  # [n_nodes, n_nodes, 1] for ZINC

    shortest_path_result, path = algos.floyd_warshall(
        adj_orig.numpy()
    )  # [n_nodesxn_nodes, n_nodesxn_nodes]
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    rel_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.float
    )  # with graph token

    adj = torch.zeros(
        [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.bool
    )
    adj[edge_index[0, :], edge_index[1, :]] = True

    for i in range(num_virtual_tokens):
        adj[N + i, :] = True
        adj[:, N + i] = True

    # for i in range(N + num_virtual_tokens):
    #     for j in range(N + num_virtual_tokens):

    #         val = True if random.random() < 0.3 else False
    #         adj[i, j] = adj[i, j] or val

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.rel_pos = rel_pos
    item.in_degree = adj_orig.long().sum(dim=1).view(-1)
    item.out_degree = adj_orig.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()
    item.adj = adj

    return item

def trainset():
    neg_smiles = read_smiles_from_txt("data/smiles/gasa_neg.txt")
    neg_graphs = smiles_to_graph(neg_smiles, 0) 
    pos_smiles = read_smiles_from_txt("data/smiles/gasa_pos.txt")
    pos_graphs = smiles_to_graph(pos_smiles, 1) 
    os.makedirs("data/processed/graph/", exist_ok=True)

    nt, nv, nte = random_split(neg_graphs, [0.8, 0.1, 0.1])
    pt, pv, pte = random_split(pos_graphs, [0.8, 0.1, 0.1])

    train, val, test = nt + pt, nv + pv, nte + pte
    torch.save(train, "data/processed/graph/train_80w.pt")
    torch.save(val, "data/processed/graph/val_80w.pt")
    torch.save(test, "data/processed/graph/test_80w.pt")
    print(sum(d['y'] for d in train))
    print(sum(d['y'] for d in val))
    print(sum(d['y'] for d in test))

def trainset_graphormer():
    neg_smiles = read_smiles_from_txt("data/smiles/gasa_neg.txt")
    neg_graphs = smiles_to_graph_graphormer(neg_smiles, 0) 
    pos_smiles = read_smiles_from_txt("data/smiles/gasa_pos.txt")
    pos_graphs = smiles_to_graph_graphormer(pos_smiles, 1) 

    os.makedirs("data/processed/raw/", exist_ok=True)

    nt, nv, nte = random_split(neg_graphs, [0.8, 0.1, 0.1])
    pt, pv, pte = random_split(pos_graphs, [0.8, 0.1, 0.1])
    train, val, test = nt + pt, nv + pv, nte + pte

    #train = [preprocess_item(Data(**d)) for d in tqdm(train, desc="Processing Training Graphs", unit="graph")]
    #val = [preprocess_item(Data(**d)) for d in tqdm(val, desc="Processing Validating Graphs", unit="graph")]
    #test = [preprocess_item(Data(**d)) for d in tqdm(test, desc="Processing Testing Graphs", unit="graph")]

    torch.save(train, "data/processed/raw/train_80w.pt")
    torch.save(val, "data/processed/raw/val_80w.pt")
    torch.save(test, "data/processed/raw/test_80w.pt")
    print(sum(d['y'] for d in train))
    print(sum(d['y'] for d in val))
    print(sum(d['y'] for d in test))
    print(train[0].x.shape)

def gasatest(idx: int):
    df = pd.read_csv(f'data/smiles/gasa_test{idx}.csv')
    os.makedirs("data/processed/graph/", exist_ok=True)

    graphs = []
    for _, row in df.iterrows():
        smiles = row['smiles']
        label = row['labels']
        if label == 1:
            graphs.append(smiles_to_graph([smiles], 0)[0])
        elif label == 0:
            graphs.append(smiles_to_graph([smiles], 1)[0])

    torch.save(graphs, f"data/processed/graph/gasa_test{idx}.pt")
    print(sum(d['y'] for d in graphs))

def gasatest_graphormer(idx: int):
    df = pd.read_csv(f'data/smiles/gasa_test{idx}.csv')
    os.makedirs("data/processed/raw/", exist_ok=True)

    graphs = []
    for _, row in df.iterrows():
        smiles = row['smiles']
        label = row['labels']
        if label == 1:
            graphs.append(smiles_to_graph_graphormer([smiles], 0)[0])
        elif label == 0:
            graphs.append(smiles_to_graph_graphormer([smiles], 1)[0])
    
    graphs = [preprocess_item(Data(**d)) for d in tqdm(graphs, desc=f"Processing GASA test-{idx} Graphs", unit="graph")]

    torch.save(graphs, f"data/processed/raw/gasa_test{idx}.pt")
    print(sum(d['y'] for d in graphs))

def nonphertest():
    df = pd.read_csv(f'data/smiles/nonpher_test.csv')
    os.makedirs("data/processed/graph/", exist_ok=True)

    graphs = []
    for idx, row in df.iterrows():
        smiles = row['smiles']
        label = row['synthetic_feasibility']
        if label == "infeasible":
            graphs.append(smiles_to_graph([smiles], 0)[0])
        elif label == "feasible":
            graphs.append(smiles_to_graph([smiles], 1)[0])

    torch.save(graphs, f"data/processed/graph/nonpher_test.pt")
    print(sum(d['y'] for d in graphs))

def nonphertest_graphormer():
    df = pd.read_csv(f'data/smiles/nonpher_test.csv')
    os.makedirs("data/processed/raw/", exist_ok=True)

    graphs = []
    for idx, row in df.iterrows():
        smiles = row['smiles']
        label = row['synthetic_feasibility']
        if label == "infeasible":
            graphs.append(smiles_to_graph_graphormer([smiles], 0)[0])
        elif label == "feasible":
            graphs.append(smiles_to_graph_graphormer([smiles], 1)[0])

    #graphs = [preprocess_item(Data(**d)) for d in tqdm(graphs, desc="Processing NonPher Test Graphs", unit="graph")]
    torch.save(graphs, f"data/processed/raw/nonpher_test.pt")
    print(sum(d['y'] for d in graphs))

def sbdd(name: str):
    sbdd_smiles = read_smiles_from_txt(f"data/smiles/{name}_smiles.txt")
    fps = smiles_to_graph(sbdd_smiles, 1) 
    os.makedirs("data/sbdd/graph/", exist_ok=True)
    torch.save(fps, f"data/sbdd/graph/{name}.pt")
    print(sum(d['y'] for d in fps))

def sbdd_graphormer(name: str):
    sbdd_smiles = read_smiles_from_txt(f"data/smiles/{name}_smiles.txt")
    fps = smiles_to_graph_graphormer(sbdd_smiles, 1) 
    os.makedirs("data/sbdd/raw/", exist_ok=True)
    torch.save(fps, f"data/sbdd/raw/{name}.pt")
    print(sum(d['y'] for d in fps))

if __name__ == "__main__":
    #trainset()
    #gasatest(1)
    #gasatest(2)
    #gasatest(3)
    #nonphertest()
    models = ["3dsbdd", "d3fg", "decompdiff", "diffbp", "diffsbdd", "flag", 
              "graphbp", "ligan", "molcraft", "pocket2mol", "targetdiff", "voxbind",]
    models = ["origin", "24_8"]
    #
    for model in models:
        sbdd_graphormer(model)

    #trainset_graphormer()
    #gasatest_graphormer(1)
    #gasatest_graphormer(2)
    #gasatest_graphormer(3)
    #nonphertest_graphormer()
    #trainset_graphormer()

    
    




    
