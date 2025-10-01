import yaml
from typing import Dict, Any,List,Optional,Tuple,Union
import logging
import time
from pathlib import Path
import re
import os
import random
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from src.data.data_manager import DataManager
from src.utils.logger import Logger
from src.models.models import ModelFactory
from src.trainer.train import Trainer
from src.utils.timer import Timer
from src.utils.set_seed import set_seed


def set_logger(config):
    logger = Logger(name='main',config=config)
    return logger


def set_args(config):
    """
    解析命令行参数
    """
    args = argparse.ArgumentParser()
    args.add_argument('--config_path', type=str, default='src/config.yaml', help='config file path')
    args.add_argument('--mode', type=str, default='train', help='mode')
    args.add_argument('--model_name', type=str, default='mlp', help='model name')
    args.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    args.add_argument('--device_id', type=int, default=0, help='device id')
    args.add_argument('--seed', type=int, default=42, help='seed')
    args.add_argument('--save_dir', type=str, default='./logs', help='save dir')
    return args.parse_args()

def load_config(args):
    config_path = Path('src/config.yaml') if not args.config_path else Path(args.config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
    

def create_model(logger,model_name:str,phase:int):
    """
    创建模型
    
    Returns:
        模型对象
    """
    model = ModelFactory(phase=phase,logger=logger,model_name=model_name).model
    return model



def create_dataloaders(config: Dict[str, Any]):
    """
    创建数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        (训练加载器, 测试加载器)
    """
    data_manager = DataManager(config)
    # align with DataManager attributes
    return (
        data_manager.dataloader_phase1_train,
        data_manager.dataloader_phase1_test,
        data_manager.dataloader_phase2_train,
        data_manager.dataloader_phase2_test,
    )



def train_model(model, train_loader, device, config, logger):
    """
    model.train()
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 设备
        config: 配置字典
        logger: 日志记录器
        experiment_dir: 实验目录
    """
    trainer = Trainer(config=config, model=model, dataloader_train=train_loader, logger=logger)
    return trainer


def test_model(model, test_loader, device, config, logger):
    """
    model.eval()
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        config: 配置字典
        logger: 日志记录器
    """
    trainer = Trainer(config=config, model=model, dataloader_train=test_loader, logger=logger)
    trainer.validate(test_loader)
    return trainer

def phase1_process(model, train_loader, test_loader, device, config, logger, is_val=True):
    trainer = Trainer(config=config, model=model, dataloader_train=train_loader, logger=logger)
    if is_val:
        trainer.validate(test_loader)
    return trainer
    
def phase2_process(model, train_loader, test_loader, device, config, logger, is_val=True):
    trainer = Trainer(config=config, model=model, dataloader_train=train_loader, logger=logger)
    if is_val:
        trainer.validate(test_loader)
    return trainer


@Timer
def main():
    start_time = time.time()
    set_seed()
    args = set_args()
    config = load_config(args)
    logger = set_logger(config)
    phase_ls = config.get('Train',{}).get('phase_ls',[1,2])
    dataloader_train_phase1, dataloader_test_phase1, dataloader_train_phase2, dataloader_test_phase2 = create_dataloaders(config)
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    if 1 in phase_ls:
        model1 = create_model(logger,config.get('Model',{}).get('model_name',{}).get('phase1',{}),1)
        phase1_process(model1,dataloader_train_phase1,dataloader_test_phase1,device,config,logger,is_val=True)
    if 2 in phase_ls:
        model2 = create_model(logger,config.get('Model',{}).get('model_name',{}).get('phase2',{}),2)
        phase2_process(model2,dataloader_train_phase2,dataloader_test_phase2,device,config,logger,is_val=True)
    else:
        raise ValueError(f"Invalid phase: {phase_ls}")
    logger.info(f"Main completed, time: {time.time() - start_time:.2f}s")
    

    
    
    


if __name__ == '__main__':
    main()

