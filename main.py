from typing import Dict, Any, List, Optional, Tuple, Union
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
    

def create_model(logger: Logger, model_name: str, phase: int):
    """
    Create model using ModelFactory
    
    Args:
        logger: logger instance
        model_name: model name from config
        phase: training phase (currently only phase 1 is supported)
        
    Returns:
        model object
    """
    try:
        model_factory = ModelFactory(logger=logger, phase=phase, model_name=model_name)
        model = model_factory.model
        logger.info(f"Model created successfully: {model_name}, phase: {phase}")
        return model
    except Exception as e:
        logger.error(f"Failed to create model {model_name}: {e}")
        raise e



def create_dataloaders(config: Dict[str, Any], logger: Logger):
    """
    Create data loaders using DataManager
    
    Args:
        config: config dictionary
        logger: logger instance
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    try:
        data_manager = DataManager(config, logger)
        logger.info("Data loaders created successfully")
        logger.info(f"Train loader: {len(data_manager.dataloader_train.dataset)} samples")
        logger.info(f"Test loader: {len(data_manager.dataloader_test.dataset)} samples")
        return (
            data_manager.dataloader_train,
            data_manager.dataloader_test,
        )
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        raise e



def train_model(model, train_loader, eval_loader, save_model_dir, config, logger):
    """
    Train the model using Trainer
    
    Args:
        model: model instance
        train_loader: training data loader
        eval_loader: evaluation data loader
        save_model_dir: directory to save model
        config: config dictionary
        logger: logger instance
        
    Returns:
        trainer: trained trainer instance
    """
    try:
        trainer = Trainer(
            config=config, 
            model=model, 
            dataloader_train=train_loader,
            dataloader_eval=eval_loader,
            logger=logger, 
            save_dir=save_model_dir
        )
        logger.info("Model training completed successfully")
        return trainer
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise e


def load_checkpoint(checkpoint_path: str, logger: Logger):
    """
    Load model checkpoint and extract model information
    
    Args:
        checkpoint_path: path to the checkpoint file
        logger: logger instance
        
    Returns:
        tuple: (model_state_dict, model_name, epoch, step, total_steps)
    """
    try:
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        
        # Extract model name from filename
        # Expected format: ep{epoch}_steps_{step}_{model_name}_model.pth
        # or: ep{epoch}/{epochs}_steps{step}/{total_steps}_{model_name}_model.pth
        filename = checkpoint_path.stem  # Remove .pth extension
        
        # Parse filename to extract information
        model_name = None
        epoch = None
        step = None
        total_steps = None
        
        # Try different filename patterns
        patterns = [
            r'ep(\d+)_steps_(\d+)_(.+)_model',  # ep1000_steps_5000_grmlp_model
            r'(\d+)_steps(\d+)_(.+)_model',     # 1000_steps5000_grmlp_model
            r'ep(\d+)/(\d+)_steps(\d+)/(\d+)_(.+)_model',  # ep1000/1000_steps5000/5000_grmlp_model
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                if len(match.groups()) == 3:  # First two patterns
                    epoch, step, model_name = match.groups()
                    step = int(step)
                    epoch = int(epoch)
                elif len(match.groups()) == 5:  # Third pattern
                    epoch, epochs, step, total_steps, model_name = match.groups()
                    epoch = int(epoch)
                    step = int(step)
                    total_steps = int(total_steps)
                break
        
        if model_name is None:
            # Fallback: try to extract from the last part before '_model'
            parts = filename.split('_')
            if len(parts) >= 2 and parts[-1] == 'model':
                model_name = parts[-2]
            else:
                model_name = 'unknown_model'
                logger.warning(f"Could not extract model name from filename: {filename}")
        
        logger.info(f"Extracted info - Model: {model_name}, Epoch: {epoch}, Step: {step}, Total Steps: {total_steps}")
        
        return checkpoint, model_name, epoch, step, total_steps
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise e


def load_model_from_checkpoint(checkpoint_path: str, logger: Logger, phase: int = 1):
    """
    Load a complete model from checkpoint
    
    Args:
        checkpoint_path: path to the checkpoint file
        logger: logger instance
        phase: model phase (default: 1)
        
    Returns:
        tuple: (model, model_name, epoch, step, total_steps)
    """
    try:
        # Load checkpoint and extract info
        model_state_dict, model_name, epoch, step, total_steps = load_checkpoint(checkpoint_path, logger)
        
        # Create model using ModelFactory
        model = create_model(logger, model_name, phase)
        
        # Load state dict
        model.load_state_dict(model_state_dict)
        logger.info(f"Model state dict loaded successfully")
        
        return model, model_name, epoch, step, total_steps
        
    except Exception as e:
        logger.error(f"Failed to load model from checkpoint: {e}")
        raise e


def test_model(model, test_loader, config, logger):
    """
    Test the model using Trainer
    
    Args:
        model: model instance
        test_loader: test data loader
        config: config dictionary
        logger: logger instance
        
    Returns:
        trainer: trainer instance with test results
    """
    try:
        # Create a dummy train loader for testing (not used in validation)
        dummy_train_loader = test_loader
        trainer = Trainer(
            config=config, 
            model=model, 
            dataloader_train=dummy_train_loader,
            dataloader_eval=test_loader,
            logger=logger,
            save_dir=config.get('Train', {}).get('save_dir', './outputs')
        )
        
        # Run validation
        val_results = trainer.validate(step=0)
        logger.info("Model testing completed successfully")
        return trainer
    except Exception as e:
        logger.error(f"Failed to test model: {e}")
        raise e



@Timer
@hydra.main(version_base=None, config_path="src", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function - using Hydra configuration management
    
    Args:
        cfg: Hydra configuration object
    """
    start_time = time.time()
    
    # Set random seed for reproducibility
    set_seed()
    
    # Print configuration information
    print("=== Current Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize logger
    logger = set_logger(cfg)
    logger.info("Starting main execution...")
    
    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        dataloader_train, dataloader_test = create_dataloaders(cfg, logger)
        
        # Get training configuration
        train_config = cfg.get('Train', {})
        save_dir = train_config.get('save_dir', './outputs')
        load_model_path = train_config.get('load_model_path', None)
        
        # Create output directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Load model from checkpoint or create new model
        if load_model_path and Path(load_model_path).exists():
            logger.info(f"Loading model from checkpoint: {load_model_path}")
            model, model_name, epoch, step, total_steps = load_model_from_checkpoint(
                load_model_path, logger, phase=1
            )
            logger.info(f"Model loaded - Name: {model_name}, Epoch: {epoch}, Step: {step}")
        else:
            # Get model configuration
            model_config = cfg.get('Model', {})
            model_name = model_config.get('model_name', 'mlp')
            
            # Create new model
            logger.info(f"Creating new model: {model_name}")
            model = create_model(logger, model_name, phase=1)
        
        # Train the model
        logger.info("Starting model training...")
        trainer = train_model(
            model=model,
            train_loader=dataloader_train,
            eval_loader=dataloader_test,
            save_model_dir=save_dir,
            config=cfg,
            logger=logger
        )
        
        # Test the model (optional)
        test_enabled = train_config.get('test_after_training', False)
        if test_enabled:
            logger.info("Running model testing...")
            test_trainer = test_model(
                model=model,
                test_loader=dataloader_test,
                config=cfg,
                logger=logger
            )
        
        logger.info(f"Main execution completed successfully in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise e
    

    
    
    


if __name__ == '__main__':
    main()

