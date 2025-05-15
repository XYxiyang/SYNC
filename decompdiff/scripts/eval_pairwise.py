import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from synsbdd.models.syn3d_lin import Syn3dlin
from sklearn.neighbors import NearestNeighbors

import warnings

import argparse
import os

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

from utils.transforms import get_atomic_number_from_index
import utils.transforms as trans
from rdkit import Chem
from utils.reconstruct import reconstruct_from_generated_with_bond
import re
from utils import misc
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL) 

warnings.simplefilter(action='ignore', category=FutureWarning)


max_nodes = [1, 4, 4, 3, 3, 2, 2, 1, 3, 3, 2, 2, 1 ] #check if is correct

def knn_graph(positions, atom_types):
    #print(atom_types)
    edge_start = []

    edge_end = []
    knn = NearestNeighbors(n_neighbors=5)  # 找每个点的 3 个最近邻
    knn.fit(positions.numpy())  # 注意：scikit-learn 要使用 numpy 数据
    distances, indices = knn.kneighbors(positions.numpy())
    for i in range(len(positions)): #过滤掉超过最大连接数的邻居
        for j in range(1, len(indices[i])):
            if j >= max_nodes[atom_types[i]] + 1:
                indices[i][j] = -1

    for i in range(len(positions)):
        for j in range(1, len(indices[i])):  # 排除自己，索引从 1 开始
            if distances[i][j] < 2.0 and indices[i][j] >= 0 and i in indices[indices[i][j]]: #距离，最大连接数，双向连接
                edge_start.append(i)
                edge_end.append(indices[i][j])

    
    return {"edge_index": torch.tensor([edge_start, edge_end], dtype=torch.long)}

def keep_bidirectional_edges(edge_index):
    # 将边转换为集合，便于快速查找
    edges = set(map(tuple, edge_index.t().tolist()))
    
    # 保留双向边
    bidirectional_edges = [
        (i, j) for i, j in edges if (j, i) in edges
    ]
    
    # 转换为张量格式
    if bidirectional_edges:
        return torch.tensor(bidirectional_edges, dtype=edge_index.dtype).t()
    else:
        return torch.empty((2, 0), dtype=edge_index.dtype)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, default="./outputs_pairwise/")
    parser.add_argument('--verbose', type=eval, default=True)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/test_set')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'], default="none")
    parser.add_argument('--exhaustiveness', type=int, default=30)
    parser.add_argument('--config', type=str, default="./configs/sampling_drift.yml")
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    config = misc.load_config(args.config)
    classifier = Syn3dlin(config.classifier)
    classifier = torch.load(config.classifier.ckpt).to(args.device)

    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')
    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    total_syn_w, total_syn_l = 0, 0 
    vaild_pairs = []
    indices = []

    #valid_indices = np.load('indices.npy').astype(int)
    #max_index = valid_indices.max()
    #results_fn_list = results_fn_list[max_index:]

    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        pred_pos_w = r[0][0]['pred_pos']  # [num_samples, num_steps, num_atoms, 3]
        pred_v_w = r[0][0]['pred_v']
        pred_pos_l = r[1][0]['pred_pos']  # [num_samples, num_steps, num_atoms, 3]
        pred_v_l = r[1][0]['pred_v']
        pred_bond_type_w = r[0][0]['pred_bond_type']
        pred_bond_index_w = r[0][0]['pred_bond_index']
        pred_bond_type_l = r[1][0]['pred_bond_type']
        pred_bond_index_l = r[1][0]['pred_bond_index']

        syn_w, syn_l = 0, 0
        
        pred_pos, pred_v = torch.tensor(pred_pos_w, device=args.device), torch.tensor(pred_v_w, dtype=torch.long, device=args.device)
        
        #batch_ligand =  torch.zeros(pred_pos.shape[0], dtype=torch.long, device=pred_pos.device)
        #pred_pos, pred_v = pred_pos[steps], pred_v[steps]
        edges = keep_bidirectional_edges(knn_graph(pred_pos.detach().cpu(), pred_v.detach().cpu())['edge_index'])
        #print(edges)
        #edges = collator_knn(edges,batch_ligand)
        batch = {
            "x": pred_v+1,
            "edge_index": edges.to(args.device),
            "positions": pred_pos.float(),
            "len": 1
        }
        logits = classifier.predict(batch)
        #print(logits)
        predictions = torch.argmax(logits, dim=-1)
        synthesizable = predictions.sum().cpu()
        syn_w += synthesizable
        
        pred_pos, pred_v = torch.tensor(pred_pos_l, device=args.device), torch.tensor(pred_v_l, dtype=torch.long, device=args.device)
        
        #batch_ligand =  torch.zeros(pred_pos.shape[0], dtype=torch.long, device=pred_pos.device)
        #pred_pos, pred_v = pred_pos[steps], pred_v[steps]
        try:
            edges = keep_bidirectional_edges(knn_graph(pred_pos.detach().cpu(), pred_v.detach().cpu())['edge_index'])
        except:
            continue
        #print(edges)
        #edges = collator_knn(edges,batch_ligand)
        batch = {
            "x": pred_v+1,
            "edge_index": edges.to(args.device),
            "positions": pred_pos.float(),
            "len": 1
        }
        logits = classifier.predict(batch)
        predictions = torch.argmax(logits, dim=-1)
        synthesizable = predictions.sum().cpu()
        syn_l += synthesizable
        #print(logits)
        #print(syn_w, syn_l)
        
        try:
            pred_atom_type = trans.get_atomic_number_from_index(pred_v_w, mode='basic')
            mol = reconstruct_from_generated_with_bond(
                xyz=pred_pos_w,
                atomic_nums=pred_atom_type,
                bond_index=pred_bond_index_w,
                bond_type=pred_bond_type_w
            )
            smiles = Chem.MolToSmiles(mol)
        except:
            continue
        
        w_smiles = Chem.MolToSmiles(mol)
        
        #sd_string = generate_sdf(mol)
        print(w_smiles)

        try:
            pred_atom_type = trans.get_atomic_number_from_index(pred_v_l, mode='basic')
            mol = reconstruct_from_generated_with_bond(
                xyz=pred_pos_l,
                atomic_nums=pred_atom_type,
                bond_index=pred_bond_index_l,
                bond_type=pred_bond_type_l
            )
            smiles = Chem.MolToSmiles(mol)
        except:
            continue
        
        l_smiles = Chem.MolToSmiles(mol)
        if syn_w == 1 and syn_l == 0 and '.' not in l_smiles and '.' not in w_smiles:
            vaild_pairs.append(r_name)
            numbers = re.findall(r'\d+', r_name)
            indices.append(numbers[0])
        #sd_string = generate_sdf(mol)
        print(l_smiles)
        
        total_syn_w += syn_w
        total_syn_l += syn_l
            

    print(total_syn_w)
    print(total_syn_l)
    with open('vaild.txt', 'w') as file:
        for item in vaild_pairs:
            file.write(item + '\n')
    
    np.save('indices.npy', np.array(indices))
    
'''
def reconstruct(v, pos):
    for 

    Chem.rdDetermineBonds.DetermineConnectivity'''