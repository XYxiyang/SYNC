import json
import torch

# 读取TXT文件中的所有行
with open('9fl3_high.txt', 'r') as f:
    lines = f.readlines()

# 读取字典文件（假设是JSON格式）
smiles_dict = torch.load("9fl3.pt")

# 检查每个SMILES字符串在字典中的值是否为True
results = []
for line in lines:
    smiles = line.strip()
    if smiles in smiles_dict:
        results.append((smiles, smiles_dict[smiles]))

# 打印结果
for smiles, is_true in results:
    print(f'SMILES: {smiles}, Value in dict: {is_true}')

# 保存结果到新的TXT文件
with open('9fl3_final.txt', 'w') as f:
    for smiles, is_true in results:
        if is_true:
            f.write(smiles + '\n')

print("Results saved to results.txt")