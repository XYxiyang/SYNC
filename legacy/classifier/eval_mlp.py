import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from utils.misc import load_config, setup_seed, count_parameters
from utils.logger import print_log
from synsbdd.models.synmlp import SynMLP
from synsbdd.data.dataset import SynFpDataset
from trainers.abs_trainer import Trainer 
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



def main(args):
    setup_seed(args.seed)
    model = SynMLP(args.model)
    model = torch.load('exps/fps/IL17_2025_02_18__13_44_35/checkpoint/epoch47_step480.ckpt')
    model.eval()

    for path in args.data.eval:
        print(path)
    ###dataset###

        evalset = SynFpDataset(args.data.root + path)
        evaloader = DataLoader(evalset, batch_size=args.train.batch_size, shuffle=False)

    ###eval###
    
        device = torch.device('cuda:0')
        model = model.to(device)
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
        criterion = nn.CrossEntropyLoss()
        t_iter = tqdm(evaloader)

        loss_list = []
        acc_list = []
        zeros = []
        for b_idx, batch in enumerate(t_iter):
            batch = Trainer.to_device(batch, device)
            labels = batch['y'].squeeze().flatten().long()
            logits = model(batch['fp'])
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().sum()
            loss_list.append(loss.cpu().item())
            acc_list.append(accuracy.cpu().item())
            #print_log(f'Acc/{type}', accuracy/len(batch[0]))
            
    # 保存标签为0的索引
            for i, label in enumerate(predictions):
                if label == 0:
                    zeros.append(i + b_idx * args.train.batch_size)


        valid_metric = np.sum(acc_list) / len(evaloader.dataset)
        print_log(f'{np.sum(acc_list)}, {len(evaloader.dataset)}')
        print_log(f'Acc: {valid_metric}')
        print_log(zeros)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/ic50.yml')
    parser.add_argument('--seed', type=int, default=1016)
    parser.add_argument('--valset', type=str, default='data/smiles/test.npz')
    #parser.add_argument('')
    args = parser.parse_args()
    config, config_name = load_config(args.config)
    main(config)


