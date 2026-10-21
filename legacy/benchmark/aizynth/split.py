import pandas as pd

# 读取CSV文件
df = pd.read_csv('gasa_test3.csv')  # 替换成你的CSV文件路径

# 筛选符合synthetic_feasibility=feasible的行
feasible_smiles = df[df['labels'] == 0]['smiles']
# 筛选不符合synthetic_feasibility=feasible的行
non_feasible_smiles = df[df['labels'] == 1]['smiles']

# 将符合条件的smiles输出到feasible.txt
with open('gasa_test3_pos.txt', 'w') as f:
    for smile in feasible_smiles:
        f.write(smile + '\n')

# 将不符合条件的smiles输出到non_feasible.txt
with open('gasa_test3_neg.txt', 'w') as f:
    for smile in non_feasible_smiles:
        f.write(smile + '\n')

print("输出完成！")
