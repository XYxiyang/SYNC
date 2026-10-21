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


def sample_diffusion_ligand(model, classifier, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior', guide='origin', scale=1.0, params=None):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            elif sample_num_atoms == 'grid':
                #pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [18, 18]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            else:
                raise ValueError

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # init ligand v
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)

            r = model.sample_diffusion(
                classifier=classifier,
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode,
                guide=guide,
                scale=scale,
                params=params

            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/sampling.yml")
    parser.add_argument('-i', '--data_id', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./debug')
    parser.add_argument('--guide_mode', type=str, default='guide', choices=['guide', 'origin', 'reverse'])
    parser.add_argument('--scale', type=float, default=2.0)
    parser.add_argument('--num_sample', type=int, default=None)
    parser.add_argument('--sample_num_atoms', type=str, default=None, choices=["prior", "ref", "range"])


    args = parser.parse_args()
    #if args.guide:
    #    args.result_path = f"./outputs_guide{args.scale}/"
    #    #args.config = "./configs/sampling_debug.yml"
    

    logger = misc.get_logger('sampling')

    # Load config
    config = misc.load_config(args.config)
    if args.sample_num_atoms is not None:
        config.sample.sample_num_atoms = args.sample_num_atoms
    if args.num_sample is not None:
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
        config=config.data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)}, {len(train_set)})!')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])

    classifier = torch.load(config.classifier.ckpt, map_location=args.device).to(args.device).eval()
    #classifier = None
    #classifier = clsf().to(args.device)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')


    data = test_set[args.data_id]
    #data = train_set[args.data_id]

    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, classifier, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        guide=args.guide_mode, scale=args.scale, params={}
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
    torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))

    exit()


    #torch.load(os.path.join(result_path, f'result_{args.data_id}_minus.pt'))
    all_pred_ligand_pos = result['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
    all_pred_ligand_v = result['pred_ligand_v_traj']
    
    from utils import misc, reconstruct, transforms
    from rdkit import Chem
    smiles_list = []
    out = open("smiles/pocket_1_smiles.txt", "w")
    for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
        pred_pos, pred_v = pred_pos[-1], pred_v[-1]
        # stability check
        pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode="add_aromatic")
        # reconstruction
        try:
            pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode="add_aromatic")
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic, False)
            smiles = Chem.MolToSmiles(mol)
        except reconstruct.MolReconsError:
            #if args.verbose:
            #    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
            continue
        if '.' in smiles:
            continue
        out.write(smiles + "\n")
        smiles_list.append(smiles)
    print(smiles_list)
    out.flush() 
    out.close()



    from preprocessing.smiles2conf import process_smiles_list
    from preprocessing.sdf2graph_sbdd import sdf_to_graph

    sdfs = process_smiles_list(smiles_list, "sbdd/sdf/pocket_1.sdf")
    graphs = sdf_to_graph(sdfs, 1, islist=True)


    from synsbdd.data.dataset import Syn3dDataset
    from trainers.syn2d_trainer import Syn2dTrainer 
    from synsbdd.data.collate import collator_3d_emb
    from torch.utils.data import DataLoader
    from functools import partial

    #evalset = Syn3dDataset("sbdd/sdf/pocket_1_sbdd_lin.pt")
    evaloader = DataLoader(graphs, batch_size=256, shuffle=True, pin_memory=True, collate_fn=partial(
                collator_3d_emb,
                max_node=155
            ))
    #evaloader = DataLoader(evalset, batch_size=256, shuffle=True, collate_fn=collate_graph)
    ###val###

    evaluator = Syn3dlin(config.evaluator)
    evaluator = torch.load(config.evaluator.ckpt).to(args.device)
    device = torch.device('cuda:0')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    t_iter = tqdm(evaloader)
    loss_list = []
    acc_list = []
    for batch in t_iter:
        batch = Syn2dTrainer.to_device(batch, device)
        labels = batch.y.squeeze().flatten().long()
        logits = evaluator.evaluate(batch)
    
        loss = criterion(logits, labels)
        #grad_x = torch.autograd.grad(loss, batch.x, create_graph=True)[0]
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().sum()
        loss_list.append(loss.cpu().item())
        acc_list.append(accuracy.cpu().item())
        #print_log(f'Acc/{type}', accuracy/len(batch[0]))
        print(predictions.sum())
        #nni.report_final_result(predictions.sum().detach().cpu().item())

    valid_metric = np.sum(acc_list) / len(evaloader.dataset)


