import numpy as np
import random


from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import random

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

num_molecules = 1000

def read_smiles_from_txt(file_path):
    smiles_list = []
    with open(file_path, 'r') as file:
        for line in file:
            smiles = line.strip()
            if smiles:  
                smiles_list.append(smiles)
    return smiles_list
neg = read_smiles_from_txt("compas_smiles.txt")
pos = read_smiles_from_txt("pos_smiles_10w.txt")
#indices = list(range(len(neg)))
#random.shuffle(indices)
#first_80k = indices[:100000]
#molecules = [Chem.MolFromSmiles(neg[i])for i in first_80k]
molecules1 = [Chem.MolFromSmiles(i)for i in neg]
molecules2 = [Chem.MolFromSmiles(i)for i in pos[:1000]]


fingerprints1 = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in molecules1]
fingerprints2 = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in molecules2]


# 计算两两之间的 Tanimoto 相似度并计算平均值
total_similarity = 0
num_comparisons = 0

for i in range(1000):
    for j in range(0, len(fingerprints1)):
        similarity = DataStructs.TanimotoSimilarity(fingerprints2[i], fingerprints1[j])
        total_similarity += similarity
        num_comparisons += 1

average_similarity = total_similarity / num_comparisons if num_comparisons > 0 else 0
print(f"Average Tanimoto Similarity: {average_similarity:.4f}")
