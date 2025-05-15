import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from utils.misc import load_config, setup_seed, count_parameters
from utils.logger import print_log
from synsbdd.models.syn1d import Syn1d
from synsbdd.data.dataset import Syn1dDataset
from trainers.abs_trainer import Trainer 
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



def main(args):
    setup_seed(args.seed)
    model = Syn1d(args.model)
    model = torch.load('exps/syn1d/80w_2024_11_19__13_11_32/checkpoint/epoch9_step25000.ckpt')
    model.eval()

    ###dataset###
    for path in args.data.eval:
        evalset = Syn1dDataset(args.data.root + path)
        evaloader = DataLoader(evalset, batch_size=args.train.batch_size, shuffle=True)

    ###eval###
    
        device = torch.device('cuda:0')
        model = model.to(device)
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
        criterion = nn.CrossEntropyLoss()
        t_iter = tqdm(evaloader)

        loss_list = []
        acc_list = []
        for batch in t_iter:
            batch = Trainer.to_device(batch, device)
            labels = batch['y'].squeeze().flatten().long()
            logits = model(batch['smiles'].long())
        
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
    parser.add_argument('--config', default='./configs/1d_classifier.yml')
    parser.add_argument('--seed', type=int, default=1016)
    #parser.add_argument('')
    args = parser.parse_args()
    config, config_name = load_config(args.config)
    main(config)


