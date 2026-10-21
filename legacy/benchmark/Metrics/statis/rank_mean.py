import pandas as pd

# 1. 读取数据，第一列作为行索引（Datasets）
df = pd.read_csv("rank.csv", index_col=0)

# 2. 按行计算排名：值越大排名越好（1 表示最高），method='min' 保证相同值并列同一名次
ranks = df.rank(axis=1, method='min', ascending=False)

# 3. 计算每个方法（每列）的平均排名
avg_rank = ranks.mean(axis=0)

# 4. 可选：把平均排名添加为一行
ranks.loc['Average Rank'] = avg_rank

# 5. 输出
print("各数据集上的排名：")
print(ranks)
print("\n各方法的平均排名：")
print(avg_rank)
