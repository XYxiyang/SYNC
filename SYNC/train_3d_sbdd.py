import argparse
from torch.utils.data import DataLoader


from functools import partial
from utils.misc import load_config, setup_seed, count_parameters
from utils.logger import print_log
from synsbdd.models.syn3d_lin import Syn3dlin
from synsbdd.data.dataset import Syn3dDataset
from trainers.syn2dg_trainer import Syn2dGTrainer 
from synsbdd.data.collate import collator_3d_emb, collator_3d_lin, collator_3d_rebuild, collator_3d_knn
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



def main(args):
    setup_seed(args.seed)
    model = Syn3dlin(args.model)


    ###dataset###
    trainset = Syn3dDataset(args.data.root + args.data.train)
    valset = Syn3dDataset(args.data.root + args.data.val)
    testset = Syn3dDataset(args.data.root + args.data.test)

    trainloader = DataLoader(trainset, batch_size=args.train.batch_size, shuffle=True, num_workers=args.train.num_workers, pin_memory=True, collate_fn=partial(
                    collator_3d_emb,
                    max_node=68, noisy=False
                ))
    #trainloader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=collate_graph)
    valloader = DataLoader(valset, batch_size=args.train.batch_size, shuffle=False, num_workers=args.train.num_workers, pin_memory=True, collate_fn=partial(
                    collator_3d_emb,
                    max_node=68, noisy=False
                ))
    testloader = DataLoader(testset, batch_size=args.train.batch_size, shuffle=False, num_workers=args.train.num_workers, pin_memory=True, collate_fn=partial(
                    collator_3d_emb,
                    max_node=68, noisy=False
                ))

    ###train###

    print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
    trainer = Syn2dGTrainer(model, trainloader, valloader, testloader, args)
    trainer.train(args.get('gpus', [0]), args.get('local_rank', -1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/3d_classifier_sbdd.yml')
    parser.add_argument('--seed', type=int, default=1016)
    parser.add_argument('--tag', type=str, default='80w_1e-5_emb_diffsbdd')
    parser.add_argument('--layers', type=int, default='4')
    parser.add_argument('--dim', type=int, default='128')
    
    #parser.add_argument('')
    args = parser.parse_args()
    config, config_name = load_config(args.config)
    config.train.tag = args.tag
    config.model.encoder.in_node_nf = args.dim
    config.model.encoder.out_node_nf = args.dim
    config.model.encoder.hidden_nf = args.dim
    config.model.head.d_model = args.dim
    config.model.encoder.n_layers = args.layers
    main(config)


