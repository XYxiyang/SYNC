# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import argparse
import os

import numpy as np
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from copy import deepcopy

from utils import misc
from utils.evaluation import scoring_func
from utils.evaluation.docking import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask
from multiprocessing import Pool
from functools import partial
from glob import glob
from utils.evaluation import eval_bond_length
from utils.transforms import get_atomic_number_from_index
import utils.transforms as trans
from rdkit import Chem
from utils.reconstruct import reconstruct_from_generated_with_bond
from synsbdd.models.syn3d_lin import Syn3dlin

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')

from sklearn.neighbors import NearestNeighbors
max_nodes = [1, 4, 4, 3, 3, 2, 2, 1, 3, 3, 2, 2, 1 ] #check if is correct
def knn_graph(positions, atom_types):
    edge_start = []

    edge_end = []
    knn = NearestNeighbors(n_neighbors=5)  # 找每个点的 3 个最近邻
    knn.fit(positions)  # 注意：scikit-learn 要使用 numpy 数据
    distances, indices = knn.kneighbors(positions)
    for i in range(len(positions)): #过滤掉超过最大连接数的邻居
        for j in range(1, len(indices[i])):
            if j >= max_nodes[atom_types[i]] + 1:
                indices[i][j] = -1

    for i in range(len(positions)):
        for j in range(1, len(indices[i])):  # 排除自己，索引从 1 开始
            if distances[i][j] < 2.0 and indices[i][j] >= 0 and i in indices[indices[i][j]]: #距离，最大连接数，双向连接
                edge_start.append(i)
                edge_end.append(indices[i][j])


    return torch.tensor([edge_start, edge_end], dtype=torch.long)
 

def eval_single_datapoint(index, id, args):
    config = misc.load_config('configs/sampling_drift.yml')
    if isinstance(index, dict):
        # reference set
        index = [index]

    ligand_filename = index[0]['ligand_filename']
    num_samples = len(index[:100])
    results = []
    n_eval_success = 0
    all_pair_dist, all_bond_dist = [], []
    classifier = Syn3dlin(config.classifier)
    classifier = torch.load(config.classifier.ckpt)
    for sample_idx, sample_dict in enumerate(tqdm(index[:num_samples], desc='Eval', total=num_samples)):
        print(sample_dict['pred_v'], sample_dict['pred_pos'])
        edges = knn_graph(sample_dict['pred_pos'], sample_dict['pred_v'])
        batch = {
                "x": torch.tensor(sample_dict['pred_v']+1, device="cuda:0"),
                "edge_index": edges.to("cuda:0"),
                "positions": torch.tensor(sample_dict['pred_pos'], device="cuda:0", dtype=torch.float),
                "len": 1
            }
        logits = classifier.predict(batch)
        print(logits)

    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_file', type=str, default='outputs_debug')  # 'baselines/results/pocket2mol_pre_dock.pt'
    parser.add_argument('-n', '--eval_num_examples', type=int, default=100)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--protein_root', type=str, default='./data/test_set')
    parser.add_argument('--docking_mode', type=str, default='vina_full',
                        choices=['none', 'qvina', 'vina', 'vina_full', 'vina_score'])
    parser.add_argument('--exhaustiveness', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--result_path', type=str, default="eval_result" )
    parser.add_argument('--aggregate_meta', type=eval, default=True)
    args = parser.parse_args()

    # result_path = os.path.join(os.path.dirname(args.meta_file), f'eval_results_docking_{args.docking_mode}')
    if args.result_path:
        os.makedirs(args.result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', args.result_path)
    logger.info(args)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # tmp get ligand filename
    # with open('data/crossdocked_v1.1_rmsd1.0_processed/test_index.pkl', 'rb') as f:
    #     test_index = pickle.load(f)

    if args.aggregate_meta:
        meta_file_list = sorted(glob(os.path.join(args.meta_file, '*/result.pt')))
        print(f'There are {len(meta_file_list)} files to aggregate')
        test_index = []
        for f in tqdm(meta_file_list, desc='Load meta files'):
            test_index.append(torch.load(f))
    else:
        test_index = torch.load(args.meta_file)
        if isinstance(test_index[0], dict):  # single datapoint sampling result
            test_index = [test_index]

    testset_results = []
    testset_pair_dist, testset_bond_dist = [], []
    with Pool(args.num_workers) as p:
        # testset_results = p.starmap(partial(eval_single_datapoint, args=args),
        #                             zip(test_index[:args.eval_num_examples], list(range(args.eval_num_examples))))
        for (r, pd, bd) in tqdm(p.starmap(partial(eval_single_datapoint, args=args),
                                zip(test_index[:args.eval_num_examples], list(range(args.eval_num_examples)))),
                      total=args.eval_num_examples, desc='Overall Eval'):
            testset_results.append(r)
            testset_pair_dist += pd
            testset_bond_dist += bd

    # model_name = os.path.basename(args.meta_file).split('_')[0]
    if args.result_path:
        torch.save(testset_results, os.path.join(args.result_path, f'eval_all.pt'))

    qed = [x['chem_results']['qed'] for r in testset_results for x in r]
    sa = [x['chem_results']['sa'] for r in testset_results for x in r]
    num_atoms = [len(x['pred_pos']) for r in testset_results for x in r]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    logger.info('Num atoms:   Mean: %.3f Median: %.3f' % (np.mean(num_atoms), np.median(num_atoms)))
    if args.docking_mode in ['vina', 'qvina']:
        vina = [x['vina'][0]['affinity'] for r in testset_results for x in r]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_full', 'vina_score']:
        vina_score_only = [x['vina']['score_only'][0]['affinity'] for r in testset_results for x in r]
        vina_min = [x['vina']['minimize'][0]['affinity'] for r in testset_results for x in r]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_full':
            vina_dock = [x['vina']['dock'][0]['affinity'] for r in testset_results for x in r]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    pair_length_profile = eval_bond_length.get_pair_length_profile(testset_pair_dist)
    js_metrics = eval_bond_length.eval_pair_length_profile(pair_length_profile)
    logger.info('JS pair distances: ')
    print_dict(js_metrics, logger)

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(testset_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    logger.info('JS bond distances: ')
    print_dict(c_bond_length_dict, logger)

    print_ring_ratio([x['chem_results']['ring_size'] for r in testset_results for x in r], logger)
