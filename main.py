from typing import Dict, Any,List,Optional,Tuple,Union
import logging
import time
from pathlib import Path
import re
import os
import random
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml


from src.data.data_manager import DataManager
from src.utils.logger import Logger
from src.models.models import ModelFactory
from src.trainer.train import Trainer
from src.utils.timer import Timer
from src.utils.set_seed import set_seed



def set_logger(config):
    logger = Logger(name='main',config=config)
    return logger


# 移除set_args和load_config函数，因为Hydra会自动处理配置加载
    

def create_model(logger,model_name:str,phase:int):
    """
    create model
    
    Args:
        logger: logger
        model_name: model name
        phase: phase
        
    Returns:
        model object
    """
    model = ModelFactory(phase=phase,logger=logger,model_name=model_name).model
    return model



def create_dataloaders(config: Dict[str, Any]):
    """
    create data loaders
    
    Args:
        config: config dictionary
        
    Returns:
        (train loader, test loader)
    """
    data_manager = DataManager(config)
    return (
        data_manager.dataloader_phase1_train,
        data_manager.dataloader_phase1_test,
        data_manager.dataloader_phase2_train,
        data_manager.dataloader_phase2_test,
    )



def train_model(model, train_loader, save_model_dir, config, logger):
    """
    model.train()
    
    Args:
        model: model
        train_loader: train loader
        save_model_dir: save model directory
        config: config dictionary
        logger: logger
    """
    trainer = Trainer(config=config, model=model, dataloader_train=train_loader, logger=logger, save_model_dir=save_model_dir)
    return trainer


def test_model(model, test_loader,config, logger):
    """
    model.eval()
    
    Args:
        model: model
        test_loader: test loader
        config: config dictionary
        logger: logger
    """
    trainer = Trainer(config=config, model=model, dataloader_train=test_loader, logger=logger)
    trainer.validate(test_loader)
    return trainer

def phase1_pipline(model, train_loader, test_loader, save_model_dir, config, logger, is_val=True):
    trainer = Trainer(config=config, model=model, dataloader_train=train_loader, logger=logger, save_model_dir=save_model_dir)
    if is_val:
        trainer.validate(test_loader)
    return trainer
    
def phase2_pipline(model, train_loader, test_loader, save_model_dir, config, logger, is_val=True):
    trainer = Trainer(config=config, model=model, dataloader_train=train_loader, logger=logger, save_model_dir=save_model_dir)
    if is_val:
        trainer.validate(test_loader)
    return trainer



@Timer
@hydra.main(version_base=None, config_path="src", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    主函数 - 使用Hydra配置管理
    
    Args:
        cfg: Hydra配置对象
    """
    start_time = time.time()
    set_seed()
    
    # 打印配置信息（可选）
    print("=== 当前配置 ===")
    print(OmegaConf.to_yaml(cfg))
    
    # 使用Hydra配置对象
    logger = set_logger(cfg)
    phase_ls = cfg.get('Train', {}).get('phase_ls', None)
    assert phase_ls is not None, f"phase_ls is not set"
    
    dataloader_train_phase1, dataloader_test_phase1, dataloader_train_phase2, dataloader_test_phase2 = create_dataloaders(cfg)
    
    if 1 in phase_ls:
        print(f"Training phase 1")
        model1 = create_model(logger, cfg.get('Model', {}).get('model_name', {}), 1)
        phase1_pipline(model1, dataloader_train_phase1, dataloader_test_phase1, 
                      cfg.get('Train', {}).get('save_model_path1', {}), cfg, logger, is_val=True)
        print(f"Training phase 1 completed")
        
    if 2 in phase_ls:
        print(f"Training phase 2")
        model2 = create_model(logger, cfg.get('Model', {}).get('model_name', {}), 2)
        phase2_pipline(model2, dataloader_train_phase2, dataloader_test_phase2, 
                      cfg.get('Train', {}).get('save_model_path2', {}), cfg, logger, is_val=True)
        print(f"Training phase 2 completed")
    else:
        raise ValueError(f"Invalid phase: {phase_ls}")
        
    logger.info(f"Main completed, time: {time.time() - start_time:.2f}s")
    

    
    
    


if __name__ == '__main__':
    main()

