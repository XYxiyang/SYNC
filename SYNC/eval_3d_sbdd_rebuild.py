import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from functools import partial

from utils.misc import load_config, setup_seed, count_parameters
from utils.logger import print_log
from synsbdd.models.syn3d_rebuild import Syn3dRebuild
from synsbdd.data.dataset import Syn3dDataset
from trainers.syn3d_rebuild_trainer import Syn3dRebuildTrainer
from synsbdd.data.collate import collator_3d_sbdd
from data.preprocessing.smiles2conf import process_molecule
from data.preprocessing.sdf2graph_sbdd import sdf_to_graph
from rdkit import Chem
from torch_geometric.data import Data
from synsbdd.data.collate import collator_3d_emb, collator_3d_lin, collator_3d_rebuild

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args):
    setup_seed(args.seed)
    model = Syn3dRebuild(args.model)
    print(model)
    model = torch.load('exps/syn3d_sbdd/80w_1e-5_emb_2024_12_11__12_37_26/checkpoint/epoch49_step50.ckpt')
    print(model)
    #model.eval()

    ###dataset###
        #evaloader = DataLoader(evalset, batch_size=256, shuffle=True, collate_fn=collate_graph)

    ###eval###
    #smiles = 'COC.CCOON(CON)C(C)CC(O)CCC(C)OOC.N'
    #graph = torch.load("./mol.pt")
    
    
    device = torch.device('cuda:0')
    model.to(device)
    print(model)
    testset = Syn3dDataset(args.data.root + args.data.test)
    

    testloader = DataLoader(testset, batch_size=args.train.batch_size, shuffle=False, num_workers=args.train.num_workers, pin_memory=True, collate_fn=partial(
                    collator_3d_rebuild,
                    max_node=68
                ))
    criterion = nn.MSELoss()
    

    for batch in testloader:
        print(batch)
        batch = batch.to(device)
        a = model(batch)
        loss = criterion(a[0], batch.adj_matrice[0])
        
        input()

    #print_log(f'Acc/{type}', accuracy/len(batch[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/3d_classifier_sbdd.yml')
    parser.add_argument('--seed', type=int, default=59)
    #parser.add_argument('')
    args = parser.parse_args()
    config, config_name = load_config(args.config)
    #config.train.batch_size = 256
    main(config)


