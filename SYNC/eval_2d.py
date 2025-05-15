import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from utils.misc import load_config, setup_seed, count_parameters
from utils.logger import print_log
from synsbdd.models.syn2d import Syn2d
from synsbdd.data.dataset import Syn2dDataset
from trainers.syn2d_trainer import Syn2dTrainer 
from synsbdd.data.collate import collate_graph
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args):
    setup_seed(args.seed)
    model = Syn2d(args.model)
    model = torch.load('exps/syn2d/10w_2024_11_19__09_47_43/checkpoint/epoch6_step4578.ckpt')
    model.eval()

    ###dataset###
    for path in args.data.eval:
        evalset = Syn2dDataset(args.data.root + path)
        evaloader = DataLoader(evalset, batch_size=args.train.batch_size, shuffle=True,collate_fn=collate_graph)
        #evaloader = DataLoader(evalset, batch_size=256, shuffle=True, collate_fn=collate_graph)

    ###eval###
    
        device = torch.device('cuda:0')
        model = model.to(device)
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
        criterion = nn.CrossEntropyLoss()
        t_iter = tqdm(evaloader)

        loss_list = []
        acc_list = []
        for batch in t_iter:
            batch = Syn2dTrainer.to_device(batch, device)
            labels = batch['y'].squeeze().flatten().long()
            logits = model(batch['atoms'], batch['bonds'], batch['atom_padding_mask'])
        
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().sum()
            loss_list.append(loss.cpu().item())
            acc_list.append(accuracy.cpu().item())
            #print_log(f'Acc/{type}', accuracy/len(batch[0]))
    
        valid_metric = np.sum(acc_list) / len(evaloader.dataset)
        print_log(f'{np.sum(acc_list)}, {len(evaloader.dataset)}')
        print_log(f'Acc: {valid_metric}')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/2d_classifier.yml')
    parser.add_argument('--seed', type=int, default=59)
    #parser.add_argument('')
    args = parser.parse_args()
    config, config_name = load_config(args.config)
    #config.train.batch_size = 256
    main(config)


