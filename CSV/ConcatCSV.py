import pandas as pd
import os

# 文件夹路径
folder_path = 'D:/dataset'

# 指定要合并的两个文件的名称
file1 = 'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv'  # 修改为第一个文件的实际名称
file2 = 'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv'  # 修改为第二个文件的实际名称

# 构建这两个文件的完整路径
file_path1 = os.path.join(folder_path, file1)
file_path2 = os.path.join(folder_path, file2)

# 读取这两个文件
data1 = pd.read_csv(file_path1, low_memory=False)
data2 = pd.read_csv(file_path2, low_memory=False)

# 使用pd.concat合并这两个数据集
merged_data = pd.concat([data1, data2], ignore_index=True)

# 如果CSV文件中包含时间列
try:
    # 尝试将时间列转换为datetime对象，不指定具体格式
    merged_data["Timestamp"] = pd.to_datetime(merged_data["Timestamp"], dayfirst=True)
    merged_data = merged_data.sort_values(by="Timestamp")
except KeyError:
    # 如果没有时间列，则忽略这个步骤
    pass
except ValueError as e:
    # 输出错误信息，帮助诊断问题
    print(f"转换日期时出错: {e}")
    # 可能需要进一步检查时间数据的格式

# 保存合并后的数据到新的CSV文件
merged_data.to_csv('D:/dataset/merge_test.csv', index=False)
