import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 生成示例数据

def norm(df):
    score = df['smiles_score']
    # 最小-最大归一化
    min_val = score.min()
    max_val = score.max()
    normalized_data = (score - min_val) / (max_val - min_val)
    return normalized_data.to_numpy()

model = "targetdiff"
fs = norm(pd.read_csv(f"../fsscore/{model}_scores.csv"))
ra = norm(pd.read_csv(f"../rascore/{model}_ra.csv"))
sa = norm(pd.read_csv(f"../SA/{model}_sa.csv"))
sc = norm(pd.read_csv(f"../scscore/{model}_sc.csv"))
syba = norm(pd.read_csv(f"../SYBA/{model}_syba.csv"))


np.random.seed(10)
data_dict = {
    'fs': fs,
    'ra': ra,
    'sa': sa,
    'sc': sc,
    'syba': syba,
}
data_melted = pd.DataFrame({k: pd.Series(v) for k, v in data_dict.items()}).melt(var_name='Metrics', value_name='Synsizability')

pastel_colors = ['#FF6961', '#FFDAB9', '#CFCFC4', '#FDFD96', '#D1E231'] #'#77DD77', '#FFB347', 'royalblue', '#FF9DA7']


# 创建小提琴图
plt.figure(figsize=(8,4.5))
plt.grid(ls='--', alpha=0.6)
sns.violinplot(x='Metrics', y='Synsizability', data=data_melted,  palette=pastel_colors[:len(data_dict)])

#plt.xticks(rotation=30)
#plt.subplots_adjust(bottom=0.2)
plt.savefig('lbe_adrb1.png')

plt.savefig('lbe_adrb1.pdf')

#plt.show()