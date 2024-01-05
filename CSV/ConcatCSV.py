import pandas as pd
import os

# 文件夹路径
folder_path = 'F:/dataset'

# 指定要合并的两个文件的名称
file1 = 'file1.csv'  # 修改为第一个文件的实际名称
file2 = 'file2.csv'  # 修改为第二个文件的实际名称

# 构建这两个文件的完整路径
file_path1 = os.path.join(folder_path, file1)
file_path2 = os.path.join(folder_path, file2)

# 读取这两个文件
data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)

# 使用pd.concat合并这两个数据集
merged_data = pd.concat([data1, data2], ignore_index=True)

# 如果CSV文件中包含时间列，将其转换为datetime对象以便排序
# 如果没有时间列，这两行可以删除
merged_data["Timestamp"] = pd.to_datetime(merged_data["Timestamp"], dayfirst=True)
merged_data = merged_data.sort_values(by="Timestamp")

# 保存合并后的数据到新的CSV文件
merged_data.to_csv('F:/dataset/merged_data.csv', index=False)
