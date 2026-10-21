import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from synsbdd.models.syn3d_lin import Syn3dlin
from sklearn.neighbors import NearestNeighbors
from rdkit.Chem import AllChem, DataStructs

import shutil
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

from synsbdd.data.featurize import  bond_features_simple

warnings.simplefilter(action='ignore', category=FutureWarning)

def tanimoto_sim(mol, ref):
    fp1 = Chem.RDKFingerprint(ref)
    fp2 = Chem.RDKFingerprint(mol)
    return DataStructs.TanimotoSimilarity(fp1,fp2)
    

def tanimoto_sim_N_to_1(mols, ref):
    sim = [tanimoto_sim(m, ref) for m in mols]
    return sim


def batched_number_of_rings(mols):
    n = []
    for m in mols:
        n.append(Chem.rdMolDescriptors.CalcNumRings(m))
    return np.array(n)
def calcu(molecules):
    fingerprints = [Chem.RDKFingerprint(mol) for mol in molecules]

    # 计算所有不重复分子对的 Tanimoto similarity
    similarities = []
    n = len(fingerprints)
    for i in range(n):
        for j in range(i+1, n):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarities.append(sim)

    # 计算所有 pairwise 相似度的均值
    mean_similarity = np.mean(similarities)
    print("Mean pairwise Tanimoto similarity:", mean_similarity)
    return mean_similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, default="./outputs_dpo_1800")
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

    path = "./sdfs"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
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
    sim = []
    
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        print(r_name)
        r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += len(all_pred_ligand_pos)
        mols = []
        for steps in range(-1, 0):
            for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
                pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]
                pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode="add_aromatic")
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                try:
                    mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic,basic_mode=False)
                except:
                    continue
                print(Chem.MolToSmiles(mol))
                mols.append(mol)
                #if '.' in Chem.MolToSmiles(mol):
                #    continue
        sim.append(calcu(mols))          

    print(1 - np.mean(sim).round(8), 1- np.median(sim).round(8))
    print(total_syn)
'''
def reconstruct(v, pos):
    for 

    Chem.rdDetermineBonds.DetermineConnectivity'''

