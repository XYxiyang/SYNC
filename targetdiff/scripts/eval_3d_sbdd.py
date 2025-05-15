import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from synsbdd.models.syn3d_lin import Syn3dlin
from sklearn.neighbors import NearestNeighbors

from utils import misc, reconstruct, transforms
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

from utils import misc
from preprocessing.sdf2graph_sbdd import sdf_to_graph

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, default="./outputs_24_8")
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

    evaluator = Syn3dlin(config.evaluator)
    evaluator = torch.load(config.evaluator.ckpt).to(args.device)



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

    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        #print(r_name)
        if example_idx != 1:
            continue
        r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += len(all_pred_ligand_pos)

        for steps in range(-1, 0):
            syn_knn, syn_emb = 0, 0
            for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
                #if sample_idx != 0:
                #    continue
                
                pred_pos, pred_v = pred_pos[steps], pred_v[steps]
                try:
                    pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
                    pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                    mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                except reconstruct.MolReconsError:
                    if args.verbose:
                        logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                    continue
                #batch_ligand =  torch.zeros(pred_pos.shape[0], dtype=torch.long, device=pred_pos.device)

                #pred_pos, pred_v = pred_pos[steps], pred_v[steps]

                #pred_pos, pred_v = torch.tensor(pred_pos[args.eval_step], device=args.device), torch.tensor(pred_v[args.eval_step], device=args.device)
#
                #edges = knn_graph(pred_pos.detach().cpu())
                ##print(edges)
                ##edges = collator_knn(edges,batch_ligand)
#
                #batch = {
                #    "x": pred_v,
                #    "edge_index": edges['edge_index'].to(args.device),
                #    "positions": pred_pos.float(),
                #    "len": 1
                #}
                #logits = classifier.predict(batch)
                #predictions = torch.argmax(logits, dim=-1)
                #synthesizable = predictions.sum().cpu()
                #syn_knn += synthesizable

                
                data = sdf_to_graph(mol, 1, ismol=True)
                if data is None:
                    continue
                data = data[0].to(args.device)
                logits = evaluator.evaluate(data)
                predictions = torch.argmax(logits, dim=-1)
                synthesizable = predictions.sum().cpu()
                syn_emb += synthesizable





            total_syn += syn_emb
            print(syn_knn, syn_emb)

    print(total_syn)
'''
def reconstruct(v, pos):
    for 

    Chem.rdDetermineBonds.DetermineConnectivity'''