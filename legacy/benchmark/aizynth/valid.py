import os
import re
import torch
def extract_smiles_and_status(file_path):
    results = {}
    # 读取文件内容
    with open(file_path, 'r') as file:
        for line in file:
            # 检测是否以 "Done" 开头
            if line.startswith("Done"):
                # 提取 SMILES 部分，Done 和 in 之间的内容
                match = re.search(r"Done\s+with\s+([^\s]+).*?in.*?(\d+\.?\d*)\s+s", line)
                if match:
                    smiles = match.group(1)  # SMILES
                    # 检测是否含有 "is solved" 或 "is not solved"
                    is_solved = "is solved" in line
                    results[smiles] = is_solved
    return results

def process_directory(directory_path):
    all_results = {}
    # 获取目录下所有的 txt 文件
    for filename in os.listdir(directory_path):
        if filename.endswith(".log"):
            file_path = os.path.join(directory_path, filename)
            file_results = extract_smiles_and_status(file_path)
            all_results.update(file_results)
    return all_results

# 使用示例
directory_path = "./9fl3_dpo"  # 修改为你的目录路径
results = process_directory(directory_path)

# 输出结果
torch.save(results, '9fl3_dpo.pt')
for k,v in results.items():
    print(f"SMILES: {k}, Solved: {v}")
