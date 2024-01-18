import pandas as pd

# CSV文件路径
csv_file_path = 'D:/dataset/merge_test2.csv'

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 统计行数和列数
num_rows, num_cols = df.shape

# 获取列名（特征名）
column_names = df.columns.tolist()

# 假设标签列名为'Label'，根据实际情况修改
label_column = 'Label'

# 统计不同的label数量
unique_labels = df[label_column].nunique()

# 打印结果
print(f'行数: {num_rows}')
print(f'列数: {num_cols}')
print(f'特征名: {column_names}')
print(f'不同的label数量: {unique_labels}')