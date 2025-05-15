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
from  data.preprocessing.sdf2graph_tgtdiff import sdf_to_graph

import warnings
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL) 

warnings.simplefilter(action='ignore', category=FutureWarning)


def evaluate(model, path, threshold):
    out = open('syn.txt', 'a')
    sdf_files = []
    
    # 遍历路径下的所有文件和子目录
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".sdf"):  # 检查文件后缀
                full_path = os.path.join(root, file)  # 获取文件的完整路径
                sdf_files.append(full_path)


    ###dataset###
    syn = 0
    total = 0
    for mol in sdf_files:

        try:
            data = sdf_to_graph(mol, 1, isfile=True, threshold=threshold)
        except:
            continue
        if not data:
            continue
        total += 1
        data = data[0].to("cuda:0")
        try:
            logits = model.predict(data)
            predictions = torch.argmax(logits, dim=-1)
            synthesizable = predictions.sum().cpu()
            syn += synthesizable
        except:
            print(f"Error: {mol}")
            continue

    out.write(f"{path}_{threshold} Synthesizable: {syn}/{total} ({syn/total})\n")
    out.flush()
    out.close()

def main(args):#1329
    setup_seed(args.seed)
    model = Syn3dlin(args.model)
    model = torch.load('exps/classifier_noedge.ckpt')

    paths = ['3dsbdd', 'd3fg_linker', 'diffbp', 'decompdiff', 'diffsbdd', 'flag', 'graphbp', 'guide_aro', 'ligan', 'molcraft', 'pocket2mol', 'tgtdiff', 'voxbind', 'tgtdiff_dpo']

    paths = ['decompdiff']
    thresholds = [180, 210]
    for th in thresholds:
        for path in paths:
            print(path, th)
            evaluate(model, 'sdfs/' + path, th)
   


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


