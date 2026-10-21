import argparse
import os
import shutil

import numpy as np
import torch
import torch.utils.tensorboard
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
from utils.evaluation import scoring_func

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_dataset
from datasets.pair_dataset import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D
from torch.utils.data import Sampler, Subset
import random
from sklearn.neighbors import NearestNeighbors

from synsbdd.models.syn3d_lin import Syn3dlin
from scripts.sample_diffusion import sample_diffusion_ligand


from rdkit import Chem
from utils import transforms, reconstruct
def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dpo.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    parser.add_argument('--beta', type=float, default=0.5)   
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        #trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform
    )
    train_set, _ = subsets['train'], subsets['test']

    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        #trans.FeaturizeLigandBond(),
    ]
    dataset, subsets = get_dataset(
        config=config.data_val,
        transform=transform
    )
    _, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    valid_indices = np.load(config.train.get('pair_indices', 'indices.npy')).astype(int)
    #valid_indices = [19565, 44601]

    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']
    pairset = Subset(train_set, valid_indices)
    
    train_iterator = utils_train.inf_iterator(DataLoader(
        pairset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))

    

    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')
    model_ref = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    model_ref.load_state_dict(ckpt['model'])
    model_ref.eval().requires_grad_(False)
    model_theta = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model_theta.load_state_dict(ckpt['model'])
    classifier = torch.load(config.classifier.ckpt, map_location=args.device).to(args.device).eval()
    # print(model)
    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model_theta) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model_theta)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)


    def train(it):
        #DPO loss: -log simga(\beta DL(w, theta(w, t)) - DL(w, ref(w, t)) - DL(l, theta(l, t)) + DL(l, ref(l, t)))
        model_theta.train()
        optimizer.zero_grad()
        total_diff = 0
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            
            w_theta = model_theta.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_w_pos,
                ligand_v=batch.ligand_w_atom_feature_full,
                batch_ligand=batch.ligand_w_atom_feature_full_batch
            )

            l_theta = model_theta.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_l_pos,
                ligand_v=batch.ligand_l_atom_feature_full,
                batch_ligand=batch.ligand_l_atom_feature_full_batch
            )

            w_ref = model_ref.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_w_pos,
                ligand_v=batch.ligand_w_atom_feature_full,
                batch_ligand=batch.ligand_w_atom_feature_full_batch
            )

            l_ref = model_ref.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_l_pos,
                ligand_v=batch.ligand_l_atom_feature_full,
                batch_ligand=batch.ligand_l_atom_feature_full_batch
            )


            loss = -1 * F.logsigmoid(-0.5 * args.beta * (w_theta['loss'] - w_ref['loss'] - l_theta['loss'] + l_ref['loss']))

            #loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
            #loss = loss / config.train.n_acc_batch
            print(w_theta['loss'].item() - l_theta['loss'].item(), w_theta['loss_pos'].item(), loss.item())
            #print(loss.item(), w_theta['loss'].item(), w_ref['loss'].item(), l_theta['loss'].item(), l_ref['loss'].item())
            total_diff += w_theta['loss'].item() - l_theta['loss'].item()
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model_theta.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f Lr: %.6f' % (
                    it, loss,  optimizer.param_groups[0]['lr'] 
                )
            )
            #for k, v in results.items():
            #    if torch.is_tensor(v) and v.squeeze().ndim == 0:
            #        writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()
        


    def validate(it):
        misc.seed_all(config.train.seed)
        data = val_set[1]

        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model_theta, classifier, data, config.sample.num_samples,
            batch_size=100, device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            guide='origin', scale=1.0, params={}
        )

        all_pred_ligand_pos = pred_pos_traj  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = pred_v_traj
        #num_samples += len(all_pred_ligand_pos)
        syn = 0
        sa = []
        for steps in range(-1, 0):
            
            for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
                #if sample_idx != 0:
                #    continue
                pred_pos, pred_v = pred_pos[-1], pred_v[-1]
                pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode="add_aromatic")
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode='add_aromatic')
                try:
                    mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                except:
                    continue
                logger.info(Chem.MolToSmiles(mol))

                if sample_idx == 0: #only show molecule 0 for visualization
                    with Chem.SDWriter(os.path.join(log_dir, f"dpo{it}.sdf")) as writer:
                        writer.write(mol)

                if '.' in Chem.MolToSmiles(mol):
                    continue

                chem_results = scoring_func.get_chem(mol)
                sa.append(chem_results['sa'])
                    
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

            #total_syn += syn
            logger.info(syn)
            logger.info(np.mean(sa))

        
        return syn
    
    


    try:
        best_syn, best_iter = None, None
        #val_syn = validate(0)
        for it in range(1, config.train.max_iters + 1):
            # with torch.autograd.detect_anomaly():
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                # Historical validation samples the held-out test set. Disable by default.
                val_syn = validate(it) if config.train.get('validate_on_test', False) else float('nan')
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model_theta.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
                if best_syn is None or val_syn > best_syn: #
                    logger.info(f'[Validate] Best val loss achieved: {val_syn:.6f}')
                    best_syn, best_iter = val_syn, it
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_syn:.6f} at iter {best_iter}')
    except KeyboardInterrupt:
        logger.info('Terminating...')

