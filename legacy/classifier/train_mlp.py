import argparse
from torch.utils.data import DataLoader
import os
import json

from utils.misc import load_config, setup_seed, count_parameters
from utils.logger import print_log
from synsbdd.models.synmlp import SynMLP
from synsbdd.data.dataset import SynFpDataset   
from trainers.synmlp_trainer import SynMLPTrainer 
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



def main(args):
    setup_seed(args.seed)
    model = SynMLP(args.model)
    print(model)

    ###dataset###
    trainset = SynFpDataset(args.data.root + args.data.train)
    valset = SynFpDataset(args.data.root + args.data.val)
    testset = SynFpDataset(args.data.root + args.data.test)

    trainloader = DataLoader(trainset, batch_size=args.train.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.train.batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.train.batch_size, shuffle=False)

    ###train###
    print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
    trainer = SynMLPTrainer(model, trainloader, valloader, testloader, args)
    trainer.train(args.get('gpus', [0]), args.get('local_rank', -1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/ic50.yml')
    parser.add_argument('--seed', type=int, default=1016)
    parser.add_argument('--tag', type=str, default='IL17')
    #parser.add_argument('')
    args = parser.parse_args()
    config, config_name = load_config(args.config)
    print(config)
    config.train.tag = args.tag
    main(config)


