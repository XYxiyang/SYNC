import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

# 全局字体设置（一次性调整）
plt.rcParams.update({
    'font.size': 20,            # 坐标轴数字、标签等基础字体
    'axes.titlesize': 24,       # 图标题
    'axes.labelsize': 22,       # X/Y轴标签
    'xtick.labelsize': 20,      # X轴刻度字体
    'ytick.labelsize': 20,      # Y轴刻度字体
    'legend.fontsize': 20,      # 图例字体
})
# 替换为你的文件路径
def distribution(metric, threshold, file):

    file1 = f'../{metric}/{file}_neg_{metric}.csv'
    file2 = f'../{metric}/{file}_pos_{metric}.csv'



    # 读取CSV文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 假设“数值”这一列的列名是 '数值'
    value_column = 'smiles_score'


    # 均分成 5 个区间
    all_values = pd.concat([df1[value_column], df2[value_column]])

    # 自动计算最小值和最大值
    min_val = all_values.min()
    max_val = all_values.max()

    bins = np.linspace(min_val, max_val, 21)  # 5段 = 6个边界
    bin_centers = (bins[:-1] + bins[1:]) / 2       # 用于画图的X坐标

    # 分箱处理
    df1['bin'] = pd.cut(df1[value_column], bins=bins, right=False, include_lowest=True)
    df2['bin'] = pd.cut(df2[value_column], bins=bins, right=False, include_lowest=True)

    # 统计数量并对齐顺序
    count1 = df1['bin'].value_counts().sort_index()
    count2 = df2['bin'].value_counts().sort_index()

    # 将 count 转换成 numpy 数组
    y1 = count1.values
    y2 = count2.values

    # 画图
    plt.figure(figsize=(12, 6))

    # 填充区域：使用 fill_between
    plt.fill_between(bin_centers, y1, color='skyblue', alpha=0.5, label='unsynthesizable')
    plt.plot(bin_centers, y1, color='skyblue')

    plt.fill_between(bin_centers, y2, color='salmon', alpha=0.5, label='synthesizable')
    plt.plot(bin_centers, y2, color='salmon')
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')

    plt.xlabel(f'{metric.upper()}')
    plt.ylabel('Frequency')
    plt.title(f'{file.upper()} Distribution over {metric.upper()}')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.savefig(f'dis_{file}_{metric}.pdf', format='pdf')

if __name__ == "__main__":
    distribution("sa", 0.61, "gasa_test1")
    distribution("syba", -18.6, "gasa_test1")
    distribution("qed", 0.25, "gasa_test1")
    distribution("fs", -13, "gasa_test1")
    distribution("ra", 0.5, "gasa_test1")
    distribution("sc", 3.1, "gasa_test1")
    distribution("fs", -13, "gasa_test2")
    distribution("fs", -13, "gasa_test3")
    distribution("fs", -13, "nonpher")





