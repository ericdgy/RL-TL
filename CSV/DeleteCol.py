import pandas as pd

# CSV文件路径
csv_file_path = 'D:/dataset/merge_test2.csv'  # 替换为您的CSV文件的路径

# 要删除的列的名称
column_to_remove = 'Timestamp'  # 替换为您要删除的列的名称

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 删除指定的列
df.drop(column_to_remove, axis=1, inplace=True)

# 保存修改后的数据回CSV文件
df.to_csv(csv_file_path, index=False)