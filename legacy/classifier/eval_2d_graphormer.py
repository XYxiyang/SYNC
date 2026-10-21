import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from utils.misc import load_config, setup_seed, count_parameters
from functools import partial
from utils.logger import print_log
from synsbdd.models.syn2dgraphormer import Syn2dG
from synsbdd.data.dataset import Syn2dDataset
from trainers.syn2dg_trainer import Syn2dGTrainer 
from synsbdd.data.collate import collator
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args, ckpt):
    print(ckpt)
    setup_seed(args.seed)
    model = Syn2dG(args.model)
    model = torch.load(ckpt)
    model.eval()

    ###dataset###
    for path in args.data.eval:
        evalset = Syn2dDataset(args.data.root + path)
        evaloader = DataLoader(evalset, batch_size=32, shuffle=False, num_workers=args.train.num_workers, pin_memory=True, collate_fn=partial(
                    collator,
                    max_node=155,
                    multi_hop_max_dist=20,
                    rel_pos_max=1024,
                ))
    ###eval###
    
        device = torch.device('cuda:0')
        model = model.to(device)
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
        criterion = nn.CrossEntropyLoss()
        t_iter = tqdm(evaloader)

        loss_list = []
        acc_list = []
        for batch in t_iter:
            batch = Syn2dGTrainer.to_device(batch, device)
            labels = batch.y.squeeze().flatten().long()
            logits = model(batch)
        
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
    parser.add_argument('--config', default='./configs/2d_classifier_graphormer.yml')
    parser.add_argument('--seed', type=int, default=59)
    parser.add_argument('--ckpt', type=str, default='')
    #parser.add_argument('')
    args = parser.parse_args()
    config, config_name = load_config(args.config)
    #config.train.batch_size = 256
    main(config, args.ckpt)


