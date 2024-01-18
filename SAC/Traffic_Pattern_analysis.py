import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# 读取数据文件
file_path = 'D:/dataset/merge_test2.csv'
df = pd.read_csv(file_path)

# 选取所有特征 (除去标签列)
features = df.drop(columns=['Label'])

# 处理无穷大值和NaN值
features.replace([np.inf, -np.inf], np.nan, inplace=True)

# 分组计算每个标签的均值, 使用分组均值填充NaN值
group_means = df.groupby('Label').transform('mean')
df = df.fillna(group_means)

# 标准化特征
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

# 将标准化的特征添加回数据框中
df_normalized = pd.DataFrame(normalized_features, columns=features.columns)
df_normalized['Label'] = df['Label']

# 计算每个标签的特征平均值
label_means = df_normalized.groupby('Label').mean()

# 计算与Benign标签的距离
benign_mean = label_means.loc['Benign']
distances = euclidean_distances(label_means, [benign_mean]).flatten()

# 可视化
labels = label_means.index
plt.figure(figsize=(10, 2))
plt.scatter(distances, np.zeros_like(distances), c=range(len(labels)))
plt.xticks(distances, labels)
plt.xlabel('Distance from Benign Mean')
plt.yticks([])
plt.title('Average Similarity of Each Label to Benign')
plt.show()