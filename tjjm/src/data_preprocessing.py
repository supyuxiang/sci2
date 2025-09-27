import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import seaborn as sns

file_path_read = '../data/更完美.xlsx'

df = pd.read_excel(file_path_read,header=None)
print(f"Successfully read CSV file with shape: {df.shape}")

# 跳过元数据行，找到实际数据开始的位置
# 查找第一个数值行的索引
data_start_idx = None
for i in range(len(df)):
    try:
        # 尝试将第一列转换为数值
        float(df.iloc[i, 0])
        data_start_idx = i
        break
    except (ValueError, TypeError):
        continue

if data_start_idx is None:
    raise ValueError("No numeric data found in the file")


df = df.iloc[data_start_idx:,:]

# 确保所有数据都是数值类型
for col in [0, 1, 2, 3, 4, 5]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

data_start_idx_nan = None

for i in range(len(df)):
    if not pd.isna(df.iloc[i,5]):
        data_start_idx_nan = i
        break

df = df.iloc[0:data_start_idx_nan,:]

x,y,T = df.iloc[:,0],df.iloc[:,1],df.iloc[:,2]

###特征缩放
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 标准化（Z-score标准化）
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x.values.reshape(-1,1))
y_scaled = scaler.fit_transform(y.values.reshape(-1,1))
T_array = T.values if hasattr(T, 'values') else np.array(T)

x_scaled = x_scaled.flatten()
y_scaled = y_scaled.flatten()
T_array = T_array.flatten()

def data_preprocessing_0():
    return x_scaled,y_scaled,T_array


if __name__ == "__main__":
    pass



