import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from functools import partial

from utils.misc import load_config, setup_seed, count_parameters
from utils.logger import print_log
from synsbdd.models.syn3d_lin import Syn3dlin
from synsbdd.data.dataset import Syn3dDataset
from trainers.syn2d_trainer import Syn2dTrainer 
from synsbdd.data.collate import collator_3d_sbdd
from data.preprocessing.smiles2conf import process_molecule
from data.preprocessing.sdf2graph_sbdd import sdf_to_graph
from rdkit import Chem
from torch_geometric.data import Data
from torchviz import make_dot
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args):
    setup_seed(args.seed)
    model = Syn3dlin(args.model)
    model = torch.load('exps/syn3d_sbdd/80w_1e-5_emb_2024_12_09__06_41_08/checkpoint/epoch21_step52492.ckpt')
    #model.eval()

    ###dataset###
        #evaloader = DataLoader(evalset, batch_size=256, shuffle=True, collate_fn=collate_graph)

    ###eval###
    #smiles = 'COC.CCOON(CON)C(C)CC(O)CCC(C)OOC.N'
    #graph = torch.load("./mol.pt")
    
    
    device = torch.device('cuda:0')
    model = model.to(device)
    

    graph = Data(x=torch.rand((24,13)).requires_grad_(), 
                 edge_index=torch.tensor([[0,1,3,4,5,6,7],[1,0,4,3,6,5,7]],dtype=torch.long),
                 mask=torch.ones((24), dtype=torch.bool),
                 edge_attr=torch.randint(0, 6, (7,)),
                positions=torch.rand((24,3)).requires_grad_(), 
                y=torch.tensor(1, dtype=torch.long))
    graph = graph.to(device)
    print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
    criterion = nn.CrossEntropyLoss()
    

    labels = graph.y.squeeze().flatten().long()
    logits = model(graph).reshape(1, -1)
    print(logits)
    
    

    loss = criterion(logits, labels)
    make_dot(logits, params=dict(model.named_parameters())).render("model_graph", format="png")
    grad_x = torch.autograd.grad(loss, graph.x, create_graph=True)[0]
    print(grad_x)
    predictions = torch.argmax(logits, dim=1)
    print(predictions)

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


