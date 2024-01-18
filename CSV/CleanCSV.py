import pandas as pd
import numpy as np


file_path = 'D:/dataset/merge_test2.csv'


df = pd.read_csv(file_path)

# 处理无穷大值和NaN值
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 分组计算每个标签的均值, 使用分组均值填充NaN值
group_means = df.groupby('Label').transform('mean')
df = df.fillna(group_means)


# 保存修改后的数据回CSV文件
df.to_csv(file_path, index=False)

