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
import os
#from data.preprocessing.sdf2graph_sbdd import sdf_to_graph
from  data.preprocessing.sdf2graph_tgtdiff import mol_to_graph

import warnings
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL) 
from rdkit import Chem

warnings.simplefilter(action='ignore', category=FutureWarning)


def evaluate(model, path):
    
    sdf_files = []
    
    # 遍历路径下的所有文件和子目

    ###dataset###
    syn = 0
    total = 0
    sdf_files = [path]
    for file in sdf_files:
        supplier = Chem.SDMolSupplier(file, removeHs=True)
        for i, mol in enumerate(supplier):

            try:
                data = mol_to_graph(mol, 1)
                #print(data)

            except:
                continue
            if not data:
                continue
            total += 1
            data = data[0].to("cuda:0")
            print(data.x)
            data.positions[:, 2] -= 36.7458
            data.positions[:, 1] -= 27.1074
            data.positions[:, 0] -= 30.4465

            try:
                logits = model.predict(data)
                print(logits)
                predictions = torch.argmax(logits, dim=-1)
                synthesizable = predictions.sum().cpu()
                syn += synthesizable
            except:
                print(f"Error: {mol}")
                continue
            if i == 1000:
                break
    print(f"{path} Synthesizable: {syn}/{total} ({syn/total})\n")

def main(args):#1329
    setup_seed(args.seed)
    model = Syn3dlin(args.model)
    model = torch.load('exps/classifier_noedge.ckpt')


    #evaluate(model, './rebuttal_sdfs/compas_5.sdf')
    #evaluate(model, './rebuttal_sdfs/enamine_5.sdf')
    #evaluate(model, './rebuttal_sdfs/gasa_test1_neg_5.sdf')
    evaluate(model, './rebuttal_sdfs/sdfs_origin/55.sdf')
    #evaluate(model, './rebuttal_sdfs/gasa_test2_neg_5.sdf')
    #evaluate(model, './rebuttal_sdfs/gasa_test2_pos_5.sdf')
    #evaluate(model, './rebuttal_sdfs/gasa_test3_neg_5.sdf')
    #evaluate(model, './rebuttal_sdfs/gasa_test3_pos_5.sdf')
    #evaluate(model, './rebuttal_sdfs/decompdiff_guide2.sdf')
    #evaluate(model, './rebuttal_sdfs/4aaw_A_rec_10.sdf')
    #evaluate(model, './rebuttal_sdfs/4aaw_A_rec_20.sdf')
    #evaluate(model, './rebuttal_sdfs/3000.sdf')
    #evaluate(model, './rebuttal_sdfs/2z3h_A_rec_100.sdf')
    #evaluate(model, './rebuttal_sdfs/nonpher_pos_5.sdf')
    #for sdf in os.listdir("rebuttal_sdfs/guide_sdf"):
    #    
    #    evaluate(model, os.path.join("rebuttal_sdfs/guide_sdf", sdf))
    #
   #
    #print("origin")
    #for sdf in os.listdir("rebuttal_sdfs/origin_sdf"):
    #        evaluate(model, os.path.join("rebuttal_sdfs/origin_sdf", sdf))
    #    


# 初始化一个列表用于存储结果
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/3d_classifier_sbdd.yml')
    parser.add_argument('--seed', type=int, default=59)
    parser.add_argument('--device', type=str, default='cuda:0')
    #parser.add_argument('')
    args = parser.parse_args()
    config, config_name = load_config(args.config)
    #config.train.batch_size = 256
    main(config)


