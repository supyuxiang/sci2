import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import seaborn as sns
from dataclasses import dataclass

from data_preprocessing_1 import Datapreprocessing1
from data_preprocessing_0 import Datapreprocessing0

_dp1 = Datapreprocessing1(file_path='/data1/chzhang/sci826/data/正确的二维（无裂解,3MPa).xlsx')
_dp0 = Datapreprocessing0(file_path='/data1/chzhang/sci826/data/正确的二维（无裂解,3MPa).xlsx')

# 先定义占位，类定义后再赋值（避免顺序问题）
x_scaled_1 = None
y_scaled_1 = None
T_1 = None
atm_1 = None
u_1 = None
v_1 = None

x_scaled_0 = None
y_scaled_0 = None
T_0 = None


@dataclass
class config_train:
    # 数据_1（在实例化后填充）
    x_scaled_1 = x_scaled_1
    y_scaled_1 = y_scaled_1
    T_1 = T_1
    p_1 = atm_1
    u_1 = u_1
    v_1 = v_1

    # 数据_0
    x_scaled_0 = x_scaled_0
    y_scaled_0 = y_scaled_0
    T_array_0 = T_0

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_data_parallel = True
    gpu_device_ids = [0, 1]
    use_mixed_precision = True
    loss_function = "MSELoss"
    scheduler = "ReduceLROnPlateau"
    early_stopping_patience = 150

    # 训练参数_1
    epochs_1 = 600
    batch_size_1 = 256
    learning_rate_1 = 0.0001
    weight_decay_1 = 0.0001
    momentum_1 = 0.9
    patience_1 = 10
    optimizer_1 = "AdamW"
    scheduler_1 = "ReduceLROnPlateau"
    early_stopping_patience_1 = 150
    loss_function_1 = "MSELoss"
    gpu_device_ids_1 = [0, 1]
    method_1 = "standard" # standard, minmax, robust, maxabs, normalize, quantile, power


    # 训练参数_0
    epochs_0 = 600
    batch_size_0 = 256
    learning_rate_0 = 0.00001
    weight_decay_0 = 0.0001
    momentum_0 = 0.9
    patience_0 = 10
    loss_function_0 = "MSELoss"
    scheduler_0 = "ReduceLROnPlateau"
    optimizer_0 = "AdamW"
    early_stopping_patience_0 = 150
    method_0 = "standard" # standard, minmax, robust, maxabs, normalize, quantile, power

    #model
    model_0 = "EnhancedCARNet_v2"
    model_1 = "EnhancedCARNet_v2"

    # lsss权重_1
    loss_T_weight = 1/12002
    loss_atm_weight = 1000/12002
    loss_u_weight = 1000/12002
    loss_v_weight = 10000/12002
    divergence_loss_weight = 0.2
    loss_physics_weight = 0.5
    loss_physics_weight_equation_1 = 1
    loss_physics_weight_equation_2 = 1
    loss_physics_weight_equation_3 = 1
    loss_physics_weight_equation_4 = 1
    loss_physics_weight_equation_5 = 1
    loss_physics_weight_equation_6 = 1
    loss_physics_weight_equation_7 = 1

    # 模型参数
    hidden_size = 512
    num_heads = 8
    dropout = 0.2
    

    # 训练日志
  


config_train = config_train()

# 基于配置的方法，运行类预处理以填充数据
x_scaled_1, y_scaled_1, T_1, atm_1, u_1, v_1 = _dp1.run(method=config_train.method_1)
x_scaled_0, y_scaled_0, T_0 = _dp0.run(method=config_train.method_0)

# 回填到配置对象（保持现有使用方式不变）
config_train.x_scaled_1 = x_scaled_1
config_train.y_scaled_1 = y_scaled_1
config_train.T_1 = T_1
config_train.p_1 = atm_1
config_train.u_1 = u_1
config_train.v_1 = v_1

config_train.x_scaled_0 = x_scaled_0
config_train.y_scaled_0 = y_scaled_0
config_train.T_array_0 = T_0




