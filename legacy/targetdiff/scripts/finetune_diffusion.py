import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm
import utils.train as utils_train

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num

from synsbdd.models.syn3d_lin import Syn3dlin
from synsbdd.models.simple_mlp import clsf
from rdkit import RDLogger   
RDLogger.DisableLog('rdApp.*') 
import warnings
import nni

warnings.filterwarnings("ignore", message="Explicit valence for atom #")
warnings.filterwarnings("ignore", message="Molecule does not have explicit Hs")


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def finetune_model(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior', guide=False, scale=1.0, params=None):
    
    pairs = "CSCCC(O)CSC1C=NC2C=CN=CC2=N1", "CCC=NC(=C=CC(C)O)N=C1CC(N)(O)NN1"

    r = torch.load("outputs_fintune/result_1.pt")  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
    all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
    all_pred_ligand_v = r['pred_ligand_v_traj']


    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    for i in tqdm(range(num_batch)):

        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)


        t1 = time.time()
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            results = model.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch
            )
            loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
            loss = loss / config.train.n_acc_batch
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        
        

            
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/sampling.yml")
    parser.add_argument('-i', '--data_id', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./fintuned_model')
    parser.add_argument('--beta', type=float, default=0.1)

    args = parser.parse_args()
    #if args.guide:
    #    args.result_path = f"./outputs_guide{args.scale}/"
    #    #args.config = "./configs/sampling_debug.yml"

    logger = misc.get_logger('sampling')

    # Load config
    config = misc.load_config(args.config)
    config.sample.num_samples = args.num_sample
    logger.info(config)
    misc.seed_all(config.sample.seed)
    

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])

    logger.info(f'Successfully load the model! {config.model.checkpoint}')


    data = test_set[args.data_id]
    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = finetune_model(
        model , data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        guide=args.guide, scale=args.scale, params=nni.get_next_parameter()
    )
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'time': time_list
    }
    logger.info('Sample done!')

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    #torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))

    torch.save({'config': config,'model': model.state_dict(),}, os.path.join(result_path, f'finetuned.pt'))

    #torch.load(os.path.join(result_path, f'result_{args.data_id}_minus.pt'))
    

