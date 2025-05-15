import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from synsbdd.models.syn3d_lin import Syn3dlin
from sklearn.neighbors import NearestNeighbors
from rdkit.Chem import rdMolDescriptors
import warnings

import argparse
import os

import numpy as np
from rdkit import Chem

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL) 
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

from utils import misc, transforms, reconstruct

from synsbdd.data.featurize import  bond_features_simple

warnings.simplefilter(action='ignore', category=FutureWarning)

def syn_eval(r_name, classifier, unsyn=False):

    #print(r_name)
    syn_idx = {}
    r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
    all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
    all_pred_ligand_v = r['pred_ligand_v_traj']
    for steps in range(-1, 0):
        syn = 0
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            if sample_idx != 53:
                continue
            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode="add_aromatic")
            pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
            try:
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            except:
                continue
            weight = rdMolDescriptors.CalcExactMolWt(mol)
            if weight < 240:
                continue
            #print(Chem.MolToSmiles(mol))
            if '.' in Chem.MolToSmiles(mol):
                continue
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
            edge_attr = torch.tensor(edge_attr)
            pred_pos, pred_v = torch.tensor(pred_pos, device=args.device), torch.tensor(pred_v, device=args.device)
            #print(pred_v)
            
            #batch_ligand =  torch.zeros(pred_pos.shape[0], dtype=torch.long, device=pred_pos.device)
            #pred_pos, pred_v = pred_pos[steps], pred_v[steps]
            #print(edges)
            #edges = collator_knn(edges,batch_ligand)
            batch = {
                "x": pred_v+1,
                "edge_index": edge_index.to(args.device),
                #"edge_attr": edge_attr.to(args.device),
                "positions": pred_pos.float(),
                "len": 1
            }
            logits = classifier.predict(batch)
            predictions = torch.argmax(logits, dim=-1)
            synthesizable = predictions.sum().cpu()
            
            syn += synthesizable
            if synthesizable and not unsyn:
                syn_idx[sample_idx] = Chem.MolToSmiles(mol)
            if unsyn and not synthesizable:
                syn_idx[sample_idx] = Chem.MolToSmiles(mol)
            
        #sdf_string = generate_sdf(mol)
            
        #print(syn)
        return syn_idx

def build_smiles_dict(filename):
    smiles_dict = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            # 一行格式： name \t SMILES \t vina_score \t vina_min \t vina_dock
            parts = line.split('\t')
            if len(parts) < 5:
                # 如果列数不够，跳过该行或自行处理
                continue

            # 提取所需信息
            # name      = parts[0]  # 如果不需要，可以不使用
            smiles    = parts[1]
            vina_score= float(parts[2])
            vina_min  = float(parts[3])
            vina_dock = float(parts[4])

            # 将SMILES作为key，三个vina值放到列表/元组中
            smiles_dict[smiles] = [vina_score, vina_min, vina_dock]

    return smiles_dict

def all_corresponding_bigger(arr1, arr2):
    # 若两个数组长度不同，可先检查一下：
    if len(arr1) != len(arr2):
        return False  # 或根据需求处理

    # 逐一对比对应位置元素
    for x, y in zip(arr1, arr2):
        if x <= y:
            return False
    return True

def valid2sdf(r_name, valid, sdf_name):
    r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
    all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
    all_pred_ligand_v = r['pred_ligand_v_traj']
    for steps in range(-1, 0):
        syn = 0
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            if sample_idx not in valid:
                continue
            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode="add_aromatic")
            pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
            try:
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            except:
                continue
            print(Chem.MolToSmiles(mol))
            if '.' in Chem.MolToSmiles(mol):
                continue
            w = Chem.SDWriter(f'./debug/{sdf_name}_{sample_idx}.sdf')
            w.write(mol)
            w.close()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default="3")
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


    # Load generated data
    config = misc.load_config(args.config)
    classifier = Syn3dlin(config.classifier)
    classifier = torch.load(config.classifier.ckpt).to(args.device)
    guide_valid = torch.load("./results_guide.pt")
    dpo_valid = torch.load("./results_dpo.pt")
    for id in range(21,22):

        origin = syn_eval(f'./outputs/result_{id}.pt', classifier, True)
        guide = syn_eval(f'./outputs_24_8_noedge/result_{id}.pt', classifier)
        dpo = syn_eval(f'./outputs_dpo_1800/result_{id}.pt', classifier)

        chosen = set(guide.keys()) & set(dpo.keys()) & set(origin.keys())
        #print(origin)
        #for i in chosen:
        #    print(origin[i])
        #    print(guide[i])
        #    print(dpo[i])

        origin_vina = build_smiles_dict(f"./outputs/eval_results/vina_score_{id}.txt")
        guide_vina = build_smiles_dict(f"./outputs_24_8_noedge/eval_results/vina_score_{id}.txt")
        dpo_vina = build_smiles_dict(f"./outputs_dpo_1800/eval_results/vina_score_{id}.txt")

        valid = []
        for i in chosen:
            try:
                origin_vina_score = origin_vina[origin[i]]
                guide_vina_score = guide_vina[guide[i]]
                dpo_vina_score = dpo_vina[dpo[i]]
            except:
                continue
            #if origin_vina_score[2]> guide_vina_score[2] and origin_vina_score[2]> dpo_vina_score[2]:
            if True:
                #print("here")
                try:
                    #if guide_valid[guide[i]] and dpo_valid[dpo[i]]:
                    #if guide_valid[guide[i]] and dpo_valid[dpo[i]]:
                    print(i)
                    #print(origin[i])
                    #print(guide[i])
                    #print(dpo[i])
                    valid.append(i)
                except:
                    continue
                    print("not found in valid")

        valid2sdf(f'./outputs/result_{id}.pt', valid, f"{id}_origin")
        valid2sdf(f'./outputs_24_8_noedge/result_{id}.pt', valid, f"{id}_guide")
        valid2sdf(f'./outputs_dpo_1800/result_{id}.pt', valid, f"{id}_dpo")


    
'''
def reconstruct(v, pos):
    for 

    Chem.rdDetermineBonds.DetermineConnectivity'''