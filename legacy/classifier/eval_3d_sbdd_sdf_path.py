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
from  data.preprocessing.sdf2graph_tgtdiff import sdf_to_graph, mol_to_graph

import warnings
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL) 
from rdkit import Chem

warnings.simplefilter(action='ignore', category=FutureWarning)

def evaluate(model, path):
    sdf_files = []
    
    # 遍历路径下的所有文件和子目录
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".sdf"):  # 检查文件后缀
                full_path = os.path.join(root, file)  # 获取文件的完整路径
                sdf_files.append(full_path)
    #print(sdf_files)

    #sdf_files = ["./data/processed/sdf/gasa_test1_pos.sdf"]
    ###dataset###
    syn = 0
    total = 0
    for file in tqdm(sdf_files):
        supplier = Chem.SDMolSupplier(file, removeHs=True)
        syns = 0
        for mol in supplier:
            '''try:
                data = sdf_to_graph(mol, 1)
            except:
                continue'''
            data = mol_to_graph(mol, 1)
            if not data:
                continue
            total += 1
            data = data[0].to("cuda:0")
            try:
                logits = model.predict(data)
                predictions = torch.argmax(logits, dim=-1)
                synthesizable = predictions.sum().cpu()
                syns += synthesizable
                #print(Chem.MolToSmiles(mol))
                #print(data.x)
            except:
                print(f"Error: {mol}")
                continue

        #print(f"{file}: {syns}")
        syn += syns

        

    print(f"{path} Synthesizable: {syn}/{total} ({syn/total})\n")


def main(args):#1329
    setup_seed(args.seed)
    model = Syn3dlin(args.model)
    model = torch.load('exps/classifier_noedge.ckpt') #_noedge
    #model = torch.load('exps/syn3d_sbdd/80w_1e-5_emb_diffsbdd_2025_04_08__04_22_06/checkpoint/epoch50_step121686.ckpt')
    #model = torch.load('exps/classifier.ckpt')

    #evaluate(model, "../DiffSBDD/3000")
    #evaluate(model, "../DiffSBDD/origin")
    #evaluate(model, "./rebuttal_sdfs/sdfs_0509")
    #evaluate(model, "./rebuttal_sdfs/sdfs_origin")
    #evaluate(model, "./rebuttal_sdfs/sdfs_0510")
    #
    #evaluate(model, "./rebuttal_sdfs/sdfs_dpo")
    #evaluate(model, "./rebuttal_sdfs/sdfs_dpo_2")
    evaluate(model, "./rebuttal_sdfs/sdfs_dpo_5")
    #evaluate(model, "./rebuttal_sdfs/origin_10")


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


