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

ATOMS = ["None", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K"]


def save_to_sdf(atom_coordinates, output_file):
    """
    Save atomic coordinates to an SDF file.

    Parameters:
        atom_coordinates (list of tuples): Each tuple contains (atomic_symbol, x, y, z).
        output_file (str): Path to the output SDF file.
    """
    # 创建一个空的分子对象
    print("here")
    mol = Chem.RWMol()
    
    # 添加原子和坐标
    conformer = Chem.Conformer(len(atom_coordinates))
    for idx, (symbol, x, y, z) in enumerate(atom_coordinates):
        atom = Chem.Atom(symbol)  # 创建原子
        mol_idx = mol.AddAtom(atom)  # 添加原子到分子中
        conformer.SetAtomPosition(mol_idx, (x, y, z))  # 设置原子的坐标
    
    # 添加构象信息到分子中
    mol.AddConformer(conformer)
    
    # 保存到 SDF 文件
    writer = Chem.SDWriter(output_file)
    writer.write(mol)
    writer.close()
    print(f"Saved SDF to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, default="./outputs_24_8/")
    parser.add_argument('--verbose', type=eval, default=True)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/test_set')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'], default="none")
    parser.add_argument('--exhaustiveness', type=int, default=30)
    parser.add_argument('--config', type=str, default="./configs/sampling.yml")
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


    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    total_syn = 0

    r = torch.load("outputs_24_8/result_0.pt")  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']

    all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
    all_pred_ligand_v = r['pred_ligand_v_traj']
    num_samples += len(all_pred_ligand_pos)
    idx = 0
    for steps in range(-1, 0):
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            if sample_idx != idx:
                continue

            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode="add_aromatic")
            #pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
            #mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)

            ##sdf_string = generate_sdf(mol)
            #print(Chem.MolToSmiles(mol))
            #with Chem.SDWriter(f"./sdf/24_8_{sample_idx}.sdf") as writer:
            #    writer.write(mol)
            print("here")
            coord = []
            for i, pos in enumerate(pred_pos):
                coord.append((pred_atom_type[i], pos[0], pos[1], pos[2]))
            print(coord)
            save_to_sdf(coord, f"./sdf/24_8_{sample_idx}_noedge.sdf")

'''
def reconstruct(v, pos):
    for 

    Chem.rdDetermineBonds.DetermineConnectivity'''