from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_squared_error
import torch
import torch.nn as nn
from torch import Tensor

from typing import Dict, List, Optional, Union
from pathlib import Path
import os
import yaml
import logging

class MetricsManager(nn.Module):
    def __init__(self):
        super(MetricsManager,self).__init__()
        self.r2_score = r2_score
        self.mae = mean_absolute_error
        self.rmse = root_mean_squared_error
        self.mse = mean_squared_error 
        self.metrics = {
            'loss': [],
            'r2': [],
            'mae': [],
            'rmse': [],
            'mse': []
        }
    
    def cal_r2(self,output,target):
        return self.r2_score(output,target)
    
    def cal_mae(self,output,target):
        return self.mae(output,target)
    
    def cal_rmse(self,output,target):
        return self.rmse(output,target)
    
    def cal_mse(self,output,target):
        return self.mse(output,target)


    def calculate_metrics(self,output,target,loss):
        # 旧实现：直接把 GPU Tensor 传给 sklearn，会触发 cuda Tensor 转 numpy 的错误
        # assert output.shape == target.shape,'output and target must have the same shape'
        # self.metrics_current = {
        # 'loss': loss,
        # 'r2': self.cal_r2(output,target),
        # 'mae': self.cal_mae(output,target),
        # 'rmse':self.cal_rmse(output,target),
        # 'mse': self.cal_mse(output,target)
        # }
        # return self.metrics_current

        # 新实现：先对齐形状，再将 Tensor 转为 CPU numpy 数组供 sklearn 使用,展平成一维，适配 sklearn 指标
        assert output.shape == target.shape, 'output and target must have the same shape'
        y_pred = output.detach().cpu().numpy().reshape(-1)
        y_true = target.detach().cpu().numpy().reshape(-1)

        self.metrics_current = {
            'loss': float(loss.detach().cpu().item()),
            'r2': self.r2_score(y_true, y_pred),
            'mae': self.mae(y_true, y_pred),
            'rmse': self.rmse(y_true, y_pred),
            'mse': self.mse(y_true, y_pred),
        }
        return self.metrics_current

    def update_metrics(self):
        for key,value in self.metrics_current.items():
            self.metrics[key].append(value)
        