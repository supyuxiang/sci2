import yaml
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


def set_args():
    """
    parse command line arguments
    """
    args = argparse.ArgumentParser()
    args.add_argument('--config_path', type=str, default='src/config.yaml', help='config file path')
    args.add_argument('--device_id', type=int, default=0, help='device id')
    args.add_argument('--seed', type=int, default=42, help='seed')
    args.add_argument('--model_name', type=str, default='mlp', help='model name')
    args.add_argument('--epochs', type=int, default=100, help='epochs')
    return args.parse_args()

def load_config(args):
    config_path = str(Path(__file__).parent) + '/src/config.yaml' if not args.config_path else Path(args.config_path)
    with open(config_path, 'r') as f:
        assert f is not None, f"config file {config_path} not found"
        config = yaml.safe_load(f)
    if args.model_name:
        config['Model']['model_name'] = args.model_name
    if args.epochs:
        config['Train']['epochs'] = args.epochs
    return config
    

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
def main():
    start_time = time.time()
    set_seed()
    args = set_args()
    config = load_config(args)
    logger = set_logger(config)
    phase_ls = config.get('Train',{}).get('phase_ls',None)
    assert phase_ls is not None, f"phase_ls is not set"
    dataloader_train_phase1, dataloader_test_phase1, dataloader_train_phase2, dataloader_test_phase2 = create_dataloaders(config)
    if 1 in phase_ls:
        print(f"Training phase 1")
        model1 = create_model(logger,config.get('Model',{}).get('model_name',{}),1)
        phase1_pipline(model1,dataloader_train_phase1,dataloader_test_phase1,config.get('Train',{}).get('save_model_path1',{}),config,logger,is_val=True)
        print(f"Training phase 1 completed")
    if 2 in phase_ls:
        print(f"Training phase 2")
        model2 = create_model(logger,config.get('Model',{}).get('model_name',{}),2)
        phase2_pipline(model2,dataloader_train_phase2,dataloader_test_phase2,config.get('Train',{}).get('save_model_path2',{}),config,logger,is_val=True)
        print(f"Training phase 2 completed")
    else:
        raise ValueError(f"Invalid phase: {phase_ls}")
    logger.info(f"Main completed, time: {time.time() - start_time:.2f}s")
    

    
    
    


if __name__ == '__main__':
    main()

