import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# 读取数据文件
file_path = 'D:/dataset/merge_test2.csv'
df = pd.read_csv(file_path)

# 选取特征
features = df.iloc[:, :-1]
labels = df.iloc[:, -1].unique()

# 特征标准化
normalized_features = MinMaxScaler().fit_transform(features)

# 计算每个标签的中心点
centroids = {label: normalized_features[df.iloc[:, -1] == label].mean(axis=0) for label in labels}

# 计算距离并映射到一维空间
distances = np.array([min(euclidean_distances([sample], [centroids[label]])[0][0] for label in labels)
                      for sample in normalized_features])

# 绘图
plt.figure(figsize=(10, 2))
color_map = plt.get_cmap('tab10')
colors = {label: color_map(i) for i, label in enumerate(labels)}
plt.scatter(distances, np.zeros_like(distances), c=[colors[label] for label in df.iloc[:, -1]])
plt.xlabel('Minimum Distance from Centroids')
plt.yticks([])
plt.title('One-Dimensional Mapping of Network Traffic Based on Label Similarity')
plt.show()