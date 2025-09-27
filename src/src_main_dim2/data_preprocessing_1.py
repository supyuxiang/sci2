import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import seaborn as sns

file_path_read = '/data1/chzhang/sci826/data/更完美.xlsx'

df = pd.read_excel(file_path_read,header=None)

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

'''print(f"Data starts at row: {data_start_idx}")'''

df = df.iloc[data_start_idx:,:]

# 确保所有数据都是数值类型
for col in [0, 1, 2, 3, 4, 5]:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# 删除包含NaN的行
df = df.dropna()

x,y,T,atm,u,v = df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],df.iloc[:,3],df.iloc[:,4],df.iloc[:,5]



###特征缩放
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# 标准化（Z-score标准化）
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x.values.reshape(-1,1))
y_scaled = scaler.fit_transform(y.values.reshape(-1,1))


T_array = T.values if hasattr(T, 'values') else np.array(T)
atm_array = atm.values if hasattr(atm, 'values') else np.array(atm)
u_array = u.values if hasattr(u, 'values') else np.array(u)
v_array = v.values if hasattr(v, 'values') else np.array(v)

T_reshaped = T_array.reshape(-1, 1) if T_array.ndim == 1 else T_array
atm_reshaped = atm_array.reshape(-1, 1) if atm_array.ndim == 1 else atm_array
u_reshaped = u_array.reshape(-1, 1) if u_array.ndim == 1 else u_array
v_reshaped = v_array.reshape(-1, 1) if v_array.ndim == 1 else v_array


def data_preprocessing_1():
    return x_scaled,y_scaled,T_reshaped,atm_reshaped,u_reshaped,v_reshaped


if __name__ == "__main__":
    pass
