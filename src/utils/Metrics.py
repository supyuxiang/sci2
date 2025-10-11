import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import os
import yaml
import logging

class MetricsManager(nn.Module):
    def __init__(self,logger:logging.Logger,save_dir:str):
        super(MetricsManager,self).__init__()
        self.logger = logger
        self.save_dir = save_dir
        self.init_metrics()
    
    def init_metrics(self):
        self.metrics = {
            'total_loss':[],
            'physics_loss':[],
            'original_loss':[],
            'r2':[],
            'mae':[],
            'rmse':[],
            'mse':[]
        }
        
        Path(self.save_dir).mkdir(parents=True,exist_ok=True)
        self.logger.info(f"Metrics manager initialized, save_dir: {self.save_dir}")

    def cal_r2(self,output,target):
        try:
            from sklearn.metrics import r2_score
        except Exception as e:
            self.logger.error(f"Failed to import r2_score: {e}")
            raise e
        return r2_score(output,target)
    
    def cal_mae(self,output,target):
        try:
            from sklearn.metrics import mean_absolute_error
        except Exception as e:
            self.logger.error(f"Failed to import mean_absolute_error: {e}")
            raise e
        return mean_absolute_error(output,target)
    
    def cal_rmse(self,output,target):
        try:
            from sklearn.metrics import root_mean_squared_error
        except Exception as e:
            self.logger.error(f"Failed to import root_mean_squared_error: {e}")
            raise e
        return root_mean_squared_error(output,target)
    
    def cal_mse(self,output,target):
        try:
            from sklearn.metrics import mean_squared_error
        except Exception as e:
            self.logger.error(f"Failed to import mean_squared_error: {e}")
            raise e
        return mean_squared_error(output,target)


    def calculate_metrics(self,output,target,loss,physics_loss,original_loss,step:int):

        # 新实现：先对齐形状，再将 Tensor 转为 CPU numpy 数组供 sklearn 使用, 展平成一维，且对 NaN/Inf 做掩码
        assert output.shape == target.shape, 'output and target must have the same shape'
        y_pred = output.detach().cpu().numpy().reshape(-1)
        y_true = target.detach().cpu().numpy().reshape(-1)

        # 根据实际数据列重新映射
        pred_T = y_pred[:,0]      # T (K)
        pred_spfU = y_pred[:,1]   # spf. U (m/s) - 速度大小
        pred_u = y_pred[:,2]      # u (m/s) - x方向速度分量
        pred_p = y_pred[:,3]      # p (Pa) - 压力
        true_T = y_true[:,0]
        true_spfU = y_true[:,1]
        true_u = y_true[:,2]
        true_p = y_true[:,3]

        self.metrics_current = {
            'total_loss':float(loss.detach().cpu().item()),
            'physics_loss':float(physics_loss.detach().cpu().item()),
            'original_loss':float(original_loss.detach().cpu().item()),
            'mae':{
                'T':self.cal_mae(pred_T,true_T),
                'spfU':self.cal_mae(pred_spfU,true_spfU),
                'u':self.cal_mae(pred_u,true_u),
                'p':self.cal_mae(pred_p,true_p),
                'step':step,
            },
            'r2':{
                'T':self.cal_r2(pred_T,true_T),
                'spfU':self.cal_r2(pred_spfU,true_spfU),
                'u':self.cal_r2(pred_u,true_u),
                'p':self.cal_r2(pred_p,true_p),
                'step':step,
            },
            'rmse':{
                'T':self.cal_rmse(pred_T,true_T),
                'spfU':self.cal_rmse(pred_spfU,true_spfU),
                'u':self.cal_rmse(pred_u,true_u),
                'p':self.cal_rmse(pred_p,true_p),
                'step':step,
            },
            'mse':{
                'T':self.cal_mse(pred_T,true_T),
                'spfU':self.cal_mse(pred_spfU,true_spfU),
                'u':self.cal_mse(pred_u,true_u),
                'p':self.cal_mse(pred_p,true_p),
                'step':step,
            }
        }
        return self.metrics_current

    def update_metrics(self):
        for key,value in self.metrics_current.items():
            self.metrics[key].append(value)

    def save_metrics(self,save_path:str,metrics:dict[str,Any]):
        save_path = Path(save_path)
        with open(save_path,'w') as f:
            f.write(yaml.dump(metrics))
        