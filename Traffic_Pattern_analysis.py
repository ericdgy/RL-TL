import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# 读取数据文件 (请确保文件路径正确)
file_path = 'D:/dataset/merge_test2.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 选取特征
selected_features = []  # 根据你的数据集调整特征名称
normalized_features = MinMaxScaler().fit_transform(df[selected_features])

# 计算每个标签的中心点
labels = ['Benign', 'Bot','DDOS attack-LOIC-UDP', 'DDOS attack-HOIC']  # 替换为你的标签名称
centroids = {label: normalized_features[df['Label'] == label].mean(axis=0) for label in labels}

# 计算距离并映射到一维空间
distances = np.array([min(euclidean_distances([sample], [centroids[label]])[0][0] for label in labels)
                      for sample in normalized_features])

# 绘图
plt.figure(figsize=(10, 2))
colors = {label: color for label, color in zip(labels, ['blue', 'green', 'red', 'purple'])}
plt.scatter(distances, np.zeros_like(distances), c=[colors[label] for label in df['Label']])
plt.xlabel('Minimum Distance from Centroids')
plt.yticks([])
plt.title('One-Dimensional Mapping of Network Traffic Based on Label Similarity')
plt.show()