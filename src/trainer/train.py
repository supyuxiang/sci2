import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from typing import Dict
from pathlib import Path
from tqdm import tqdm
import time

from utils.timer import Timer
from utils.optimizer import build_optimizer
from utils.loss_function import build_loss_function
from utils.scheduler import build_scheduler
from utils.Metrics import MetricsManager
from utils.logger import Logger


class Trainer:

    '''
    Trainer
    Args:
        config: Dict
        model: nn.Module
        dataloader_train: DataLoader
        logger: Logger

    Attributes:
        dataloader_train: DataLoader
        model: nn.Module
        config: Dict
        logger: Logger
        optimizer: Optimizer
        loss_function: nn.Module
        scheduler: Scheduler
        metrics_manager: MetricsManager
        device: torch.device
    '''

    def __init__(self,config:Dict,model,dataloader_train,logger:Logger):
        self.dataloader_train = dataloader_train
        self.model = model
        self.config = config.get('Train',{})
        self.logger = logger
        self.config_check()
        self._set_device()
        self._build_optimizer()
        self._build_loss_function()
        self._build_scheduler()
        self._build_metrics_manager()
        self.train()


    def config_check(self):
        '''
        check config
        '''
        assert self.config.get('epochs',None) is not None,'epochs is not set'
        assert self.config.get('optimizer',None) is not None,'optimizer is not set'
        assert self.config.get('loss_function',None) is not None,'loss_function is not set'
        assert self.config.get('scheduler',None) is not None,'scheduler is not set'
        assert self.config.get('save_model_path',None) is not None,'save_model_path is not set'
        assert self.config.get('save_best_model_only_path',None) is not None,'save_best_model_only_path is not set'
        assert self.config.get('load_model_path',None) is not None,'load_model_path is not set'
        assert self.config.get('log_interval',None) is not None,'log_interval is not set'
        assert self.config.get('is_log',None) is not None,'is_log is not set'
        assert self.config.get('is_save',None) is not None,'is_save is not set'
        assert self.config.get('save_freq',None) is not None,'save_freq is not set'
        assert self.config.get('log_dir',None) is not None,'log_dir is not set'
        assert self.config.get('device',None) is not None,'device is not set'
        assert self.config.get('metrics_dir',None) is not None,'metrics_dir is not set'
        assert self.config.get('early_stopping',None) is not None,'early_stopping is not set'
        assert self.config.get('patience',None) is not None,'patience is not set'
        assert self.config.get('min_delta',None) is not None,'min_delta is not set'
        assert self.config.get('save_best_only',None) is not None,'save_best_only is not set'
        assert self.config.get('log_freq',None) is not None,'log_freq is not set'
        assert self.config.get('optimizer_config',None) is not None,'optimizer_config is not set'
        assert self.config.get('loss_function_config',None) is not None,'loss_function_config is not set'
        assert self.config.get('scheduler_config',None) is not None,'scheduler_config is not set'
        assert self.config.get('metrics_dir',None) is not None,'metrics_dir is not set'
        self.logger.info(f"Config checked, config: {self.config}")
        

    def _set_device(self):
        self.device = self.config['device']
        self.model = self.model.to(self.device)
        self.logger.info(f"Device set, device: {self.device}")

    def _build_optimizer(self):
        optimizer_name = self.config.get('optimizer', 'adam')
        self.optimizer = build_optimizer(optimizer_name, self.model.parameters(), self.config.get('optimizer_config',{}))  
        self.logger.info(f"Optimizer set, optimizer: {self.optimizer}")
        

    def _build_loss_function(self):
        loss_function_name = self.config.get('loss_function', 'mse')
        self.loss_function = build_loss_function(loss_function_name)
        self.logger.info(f"Loss function set, loss_function: {self.loss_function}")

    def _build_scheduler(self):
        scheduler_name = self.config.get('scheduler', 'cosine')
        self.scheduler = build_scheduler(scheduler_name, self.optimizer, self.config.get('scheduler_config',{}))
        self.logger.info(f"Scheduler set, scheduler: {self.scheduler}")

    def _build_metrics_manager(self):
        self.metrics_manager = MetricsManager()
        self.logger.info(f"Metrics manager set, metrics_manager: {self.metrics_manager}")

    @Timer
    def train(self):
        '''
        train

        '''
        self.logger.info(f"Training started, epochs: {self.config.get('epochs',None)}")
        for epoch in tqdm(range(self.config.get('epochs',None))):
            start_time = time.time()
            for idx,(batch_data,batch_target) in enumerate(tqdm(self.dataloader_train,desc=f'Epoch {epoch}')):
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()
                output = self.model(batch_data)
                loss = self.loss_function(output,batch_target)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                # Calculate metrics for this batch, 要改
                with torch.no_grad():
                    self.metrics_current = self.metrics_manager.calculate_metrics(output,batch_target,loss)
                    self.metrics_manager.update_metrics()

            if self.config.get('is_log',True):
                if idx % self.config.get('log_freq',10) == 0:
                    self.logger.info(f"Epoch {epoch}, Batch {idx}, Loss: {loss.item()}, Time: {time.time() - start_time}")

            if self.config.get('is_save',True) and self.config.get('save_freq',False) and epoch % self.config['save_freq'] == 0:
                    if self.config.get('save_best_only',True) and self.config.get('save_best_only_path',None):
                        if self.metrics_manager.metrics['loss'][-1] < min(self.metrics_manager.metrics['loss']):
                            self._save_model(epoch)
                            self._save_metrics(epoch)
                    else:
                        self._save_model(epoch)
                        self._save_metrics(epoch)

            if self.config.get('early_stopping', True):
                loss_history = self.metrics_manager.metrics['loss']
                n = len(loss_history)
                # 旧实现：直接从配置读取，若为字符串会导致一元负号失败
                # patience = self.config.get('patience',5)
                # min_delta = self.config.get('min_delta',1e-3)
                # 新实现：强制解析类型
                patience = int(self.config.get('patience', 5))
                try:
                    min_delta = float(self.config.get('min_delta', 1e-3))
                except (TypeError, ValueError):
                    min_delta = 1e-3
                if n > patience and (max(loss_history[-patience:]) - loss_history[-1]) < (-min_delta):
                    self.logger.info(f'Early stopping triggered, epoch: {epoch}')
                    break

            self.logger.info(f"Epoch {epoch} completed, loss: {loss.item()}, Time: {time.time() - start_time}")
        self.logger.info(f"Training completed, epochs: {self.config.get('epochs',None)}")
        self._save_model(self.config['epochs'])
        # expose metrics snapshot for external access
        self.metrics = self.metrics_manager.metrics
        return self.metrics

    def _validate(self,dataloader_val):
        '''
        validate
        '''
        self.model.eval()
        total_dict = {
            'total_pred':[],
            'total_loss':[],
            'total_r2':0,
            'total_mae':0,
            'total_rmse':0,
            'total_mse':0
        }
        with torch.no_grad():
            for batch_idx,(batch_in,batch_out) in enumerate(tqdm(dataloader_val),desc=f'Validating'):
                batch_in = batch_in.to(self.device)
                batch_out = batch_out.to(self.device)
                output = self.model(batch_in)
                loss = self.loss_function(output,batch_out)
                total_dict['total_pred'].append(output)
                total_dict['total_loss'].append(loss)
        self.logger.info(f"Validating completed, total_loss: {total_dict['total_loss']}")
        return total_dict

    # public validation API for external callers (e.g., scripts/test.py)
    def validate(self, dataloader_val):
        results = self._validate(dataloader_val)
        # keep latest validation results under a common attribute
        self.metrics = results
        return self

                
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        pass
    

    def _save_metrics(self,epoch:int):
        '''
        save metrics
        Args:
            epoch: int
        '''
        metrics_dir = Path(self.config.get('metrics_dir', 'metrics'))
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"{self.model.phase}_{self.model.name}_metrics_{epoch}.yaml"
        with open(metrics_path, 'w') as f:
            yaml.dump(self.metrics_manager.metrics, f)
        self.logger.info(f"Metrics saved, metrics_path: {metrics_path}")
    

    def _save_model(self,epoch:int,save_best_only:bool=True):
        '''
        save model
        Args:
            epoch: int
            save_best_only: bool
        '''
        # 旧实现：对目录使用 .parent.mkdir，导致目录本身未创建而保存失败
        # save_model_dir = self.config.get('save_model_path',None)
        # save_best_model_only_dir = self.config.get('save_best_model_only_path',None)
        # if not save_best_only:
        #     Path(save_model_dir).parent.mkdir(parents=True,exist_ok=True)
        #     torch.save(self.model.state_dict(),save_model_dir + f"/model_{epoch}.pth")
        #     self.logger.info(f"Model saved, save_path: {save_model_dir + f'/model_{epoch}.pth'}")
        # else:
        #     Path(save_best_model_only_dir).parent.mkdir(parents=True,exist_ok=True)
        #     torch.save(self.model.state_dict(),save_best_model_only_dir + f"/model_{epoch}.pth")
        #     self.logger.info(f"Model saved, save_path: {save_best_model_only_dir + f'/model_{epoch}.pth'}")

        # 新实现：显式创建目标目录本身，再保存
        save_model_dir = Path(self.config.get('save_model_path', 'models'))
        save_best_model_only_dir = Path(self.config.get('save_best_model_only_path', 'models/best'))
        if not save_best_only:
            save_model_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_model_dir / f"{self.model.phase}_{self.model.name}_model_{epoch}.pth"
            torch.save(self.model.state_dict(), save_path)
            self.logger.info(f"Model saved, save_path: {save_path}")
        else:
            save_best_model_only_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_best_model_only_dir / f"{self.model.phase}_{self.model.name}_model_{epoch}.pth"
            torch.save(self.model.state_dict(), save_path)
            self.logger.info(f"Model saved, save_path: {save_path}")
    
    


    
        
        
        