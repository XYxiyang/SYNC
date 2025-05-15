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

from utils import misc, transforms, reconstruct


warnings.simplefilter(action='ignore', category=FutureWarning)

max_nodes = [1, 4, 4, 3, 3, 2, 2, 1, 3, 3, 2, 2, 1 ]
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

def knn_graph(positions, atom_types):
    edge_start = []
    edge_end = []
    knn = NearestNeighbors(n_neighbors=5)  # 找每个点的 3 个最近邻
    knn.fit(positions.numpy())  # 注意：scikit-learn 要使用 numpy 数据
    distances, indices = knn.kneighbors(positions.numpy())
    for i in range(len(positions)):
        for j in range(1, len(indices[i])):  # 排除自己，索引从 1 开始
            if distances[i][j] < 2.0 and j < max_nodes[atom_types[i]] + 1 :
                edge_start.append(i)
                edge_end.append(indices[i][j])


    return {"edge_index": torch.tensor([edge_start, edge_end], dtype=torch.long)}


def collator_knn(items, batch_ligand):

    items = [
        (
            item['edge_index'],
        )
        for item in items
    ]
    (
        edge_indices,
    ) = zip(*items)


    counts = torch.bincount(batch_ligand)
    cumulative_counts = torch.cumsum(counts, dim=0)
    padded_edge = list(e.to(batch_ligand.device) for e in edge_indices)
    for i, count in enumerate(counts):
        if i > 0:
            padded_edge[i] += cumulative_counts[i-1]
    padded_edge = torch.cat(padded_edge, dim=1)

    return padded_edge

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, default="./outputs_24_8_noedge")
    parser.add_argument('--verbose', type=eval, default=True)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/test_set')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'], default="none")
    parser.add_argument('--exhaustiveness', type=int, default=30) 
    parser.add_argument('--config', type=str, default="./configs/sampling_debug.yml")
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
    total_syn = 0

    all = 0
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        print(r_name)
        r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += len(all_pred_ligand_pos)

        for steps in range(-1, 0):
            syn = 0
            for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
                #if sample_idx != 0:
                #    continue
                pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]
                pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode="add_aromatic")
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                try:
                    mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                except:
                    continue
                smiles = Chem.MolToSmiles(mol)
                if '.' in smiles:
                    continue
                all += 1
                print(Chem.MolToSmiles(mol))

                pred_pos, pred_v = torch.tensor(pred_pos, device=args.device), torch.tensor(pred_v, device=args.device)
                
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
                predictions = torch.argmax(logits, dim=-1)
                synthesizable = predictions.sum().cpu()
                syn += synthesizable

                

            #sdf_string = generate_sdf(mol)
                

            total_syn += syn
            print(syn)

    print(total_syn)
    print(all)
'''

def reconstruct(v, pos):
    for 

    Chem.rdDetermineBonds.DetermineConnectivity'''