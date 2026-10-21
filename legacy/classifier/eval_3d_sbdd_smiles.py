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
from data.preprocessing.sdf2graph_tgtdiff import sdf_to_graph
from rdkit import Chem
from rdkit.Chem import AllChem

from torch_geometric.data import Data
import warnings
from utils.misc import read_smiles_from_txt
import time
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

def process_molecule(smile):
    try:
        m = Chem.MolFromSmiles(smile)
        if m is None:
            return None 
        m = Chem.AddHs(m) 
        params = AllChem.ETKDG()
        conformer_ids = AllChem.EmbedMultipleConfs(m, numConfs=1, params=params)

        # 对每个构象进行几何优化（可选）
        for conf_id in conformer_ids:
            AllChem.UFFOptimizeMolecule(m, confId=conf_id)
        return m
    except RuntimeError as e:
        print(f"Failed to generate conformer for SMILES: {smile}. Error: {e}")
    
def main(args):
    setup_seed(args.seed)
    model = Syn3dlin(args.model)
    model = torch.load('exps/classifier_noedge.ckpt')
    #model.eval()

    ###dataset###
        #evaloader = DataLoader(evalset, batch_size=256, shuffle=True, collate_fn=collate_graph)

    ###eval###
   
    #smiles = 'CCC(=O)C1C(C)CC(CC(=O)O)C1O'
    smile = read_smiles_from_txt('smiles.txt')
    syn = 0
    device = torch.device('cuda:0')
    model = model.to(device)
    smile = ['C12C3C1C23', 'CC(C)(C)C(=C(C(C)(C)C)C(C)(C)C)C(C)(C)C']
    for smiles in tqdm(smile):
        start = time.time()
        if '.' in smiles:
            continue
        conf = process_molecule(smiles)
        graph = sdf_to_graph([conf], 1, False)[0]
        graph.x = graph.x.long()
        graph.edge_attr = graph.edge_attr.long()
        graph.mask = torch.ones((graph.x.shape[0]), dtype=torch.bool)
        graph.y = torch.tensor(1, dtype=int)
        #graph = torch.load("./mol.pt")
        #graph = Data(x=graph['x'], edge_index=graph['edge_index'], edge_attr=graph['edge_attr'], mask=graph['mask'],
        #              positions=graph['positions'], y=torch.tensor(1, dtype=int))
        graph = graph.to(device)
        criterion = nn.CrossEntropyLoss()
        labels = graph.y.squeeze().flatten().long()
        logits = model.predict(graph).reshape(1, -1)
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        syn += predictions
        print(logits)

    print(syn)
    print(time.time() - start)
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


