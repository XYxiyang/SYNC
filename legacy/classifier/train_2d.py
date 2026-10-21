import argparse
from torch.utils.data import DataLoader
import os
import json

from utils.misc import load_config, setup_seed, count_parameters
from utils.logger import print_log
from synsbdd.models.syn2d import Syn2d
from synsbdd.data.dataset import Syn2dDataset
from trainers.syn2d_trainer import Syn2dTrainer 
from synsbdd.data.collate import collate_graph
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)




def main(args):
    #setup_seed(args.seed)
    model = Syn2d(args.model)

    ###dataset###
    trainset = Syn2dDataset(args.data.root + args.data.train)
    valset = Syn2dDataset(args.data.root + args.data.val)
    testset = Syn2dDataset(args.data.root + args.data.test)

    trainloader = DataLoader(trainset, batch_size=args.train.batch_size, shuffle=True, collate_fn=collate_graph)
    #trainloader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=collate_graph)
    valloader = DataLoader(valset, batch_size=args.train.batch_size, shuffle=False, collate_fn=collate_graph)
    testloader = DataLoader(testset, batch_size=args.train.batch_size, shuffle=False, collate_fn=collate_graph)

    ###train###

    print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
    trainer = Syn2dTrainer(model, trainloader, valloader, testloader, args)
    trainer.train(args.get('gpus', [0]), args.get('local_rank', -1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/2d_classifier.yml')
    parser.add_argument('--seed', type=int, default=26515616416)
    parser.add_argument('--tag', type=str, default='80w')
    parser.add_argument('--layers', type=str, default='4')
    parser.add_argument('--dim', type=str, default='128')

    #parser.add_argument('')
    args = parser.parse_args()
    config, config_name = load_config(args.config)
    config.train.tag = args.tag
    config.model.encoder.dim = args.dim
    config.model.encoder.out_node_nf = args.dim
    config.model.encoder.hidden_nf = args.dim
    config.model.head.d_model = args.dim
    config.model.encoder.n_layers = args.layers
    config.train.tag = args.tag
    main(config)


