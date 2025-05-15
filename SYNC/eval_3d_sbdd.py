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
from synsbdd.data.collate import collator_3d_knn, collator_3d_emb

import warnings
import time
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args, ckpt):
    setup_seed(args.seed)
    model = Syn3dlin(args.model)
    model = torch.load(ckpt)
    #model.eval()

    ###dataset###
    print_log(ckpt)
    for path in args.data.eval:
        evalset = Syn3dDataset(args.data.root + path)
        evaloader = DataLoader(evalset, batch_size=args.train.batch_size, shuffle=True, pin_memory=True, collate_fn=partial(
                    collator_3d_emb,
                    max_node=155
                ))
        #evaloader = DataLoader(evalset, batch_size=256, shuffle=True, collate_fn=collate_graph)

    ###eval###
        start_time = time.time()
        device = torch.device('cuda:0')
        model = model.to(device)
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
        criterion = nn.CrossEntropyLoss()
        t_iter = tqdm(evaloader)


        loss_list = []
        acc_list = []
        for batch in t_iter:
            batch = Syn2dTrainer.to_device(batch, device)
            #batch.x = batch.x + 1
            labels = batch.y.squeeze().flatten().long()
            logits = model(batch)
        
            loss = criterion(logits, labels)
            #grad_x = torch.autograd.grad(loss, batch.x, create_graph=True)[0]
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().sum()
            loss_list.append(loss.cpu().item())
            acc_list.append(accuracy.cpu().item())

            #print_log(f'Acc/{type}', accuracy/len(batch[0]))
    
        valid_metric = np.sum(acc_list) / len(evaloader.dataset)
        end_time = time.time()
        #print_log(f'Validation time: {end_time - start_time}')
        print_log(f'{np.sum(acc_list)}, {len(evaloader.dataset)}')
        print_log(f'Acc: {valid_metric}')

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/3d_classifier_sbdd.yml')
    parser.add_argument('--seed', type=int, default=59)
    parser.add_argument('--ckpt', type=str, default='exps/syn3d_sbdd/80w_1e-5_emb_diffsbdd_2025_04_08__04_22_06/checkpoint/epoch50_step121686.ckpt')
    #parser.add_argument('')
    args = parser.parse_args()
    config, config_name = load_config(args.config)
    #config.train.batch_size = 256
    main(config, args.ckpt)


