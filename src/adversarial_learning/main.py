"""
对抗学习增强预测器主程序
"""

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

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.data_manager import DataManager
from src.utils.logger import Logger
from src.utils.timer import Timer
from src.utils.set_seed import set_seed
from .adversarial_models import create_adversarial_model
from .adversarial_trainer import AdversarialTrainer


def set_logger(config):
    """设置日志器"""
    logger = Logger(name='adversarial_main', config=config)
    return logger


def create_adversarial_model_wrapper(logger: Logger, model_config: Dict[str, Any]):
    """
    创建对抗学习模型
    
    Args:
        logger: logger实例
        model_config: 模型配置
        
    Returns:
        对抗学习模型
    """
    try:
        model = create_adversarial_model(model_config)
        logger.info(f"Adversarial model created successfully: {model_config.get('name', 'adversarial_model')}")
        return model
    except Exception as e:
        logger.error(f"Failed to create adversarial model: {e}")
        raise e


def create_dataloaders(config: Dict[str, Any], logger: Logger):
    """
    创建数据加载器
    
    Args:
        config: 配置字典
        logger: logger实例
        
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


def train_adversarial_model(model, train_loader, eval_loader, save_model_dir, config, logger):
    """
    训练对抗学习模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        eval_loader: 评估数据加载器
        save_model_dir: 模型保存目录
        config: 配置字典
        logger: logger实例
        
    Returns:
        trainer: 训练好的训练器实例
    """
    try:
        trainer = AdversarialTrainer(
            config=config, 
            model=model, 
            dataloader_train=train_loader,
            dataloader_eval=eval_loader,
            logger=logger, 
            save_dir=save_model_dir
        )
        logger.info("Adversarial model training completed successfully")
        return trainer
    except Exception as e:
        logger.error(f"Failed to train adversarial model: {e}")
        raise e


def load_checkpoint(checkpoint_path: str, logger: Logger):
    """
    加载模型检查点并提取模型信息
    
    Args:
        checkpoint_path: 检查点文件路径
        logger: logger实例
        
    Returns:
        tuple: (model_state_dict, model_name, epoch, step, total_steps)
    """
    try:
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        
        # 从文件名提取模型名称
        filename = checkpoint_path.stem  # 移除.pth扩展名
        
        # 解析文件名以提取信息
        model_name = None
        epoch = None
        step = None
        total_steps = None
        
        # 尝试不同的文件名模式
        patterns = [
            r'adversarial_model_epoch_(\d+)_(.+)',  # adversarial_model_epoch_1000_best
            r'adversarial_model_(\d+)_(.+)',        # adversarial_model_1000_best
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                if len(match.groups()) == 2:
                    epoch, model_name = match.groups()
                    epoch = int(epoch)
                break
        
        if model_name is None:
            # 回退：尝试从最后一部分提取
            parts = filename.split('_')
            if len(parts) >= 2:
                model_name = parts[-1]
            else:
                model_name = 'adversarial_model'
                logger.warning(f"Could not extract model name from filename: {filename}")
        
        logger.info(f"Extracted info - Model: {model_name}, Epoch: {epoch}, Step: {step}")
        
        return checkpoint, model_name, epoch, step, total_steps
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise e


def load_model_from_checkpoint(checkpoint_path: str, logger: Logger, config: Dict[str, Any]):
    """
    从检查点加载完整模型
    
    Args:
        checkpoint_path: 检查点文件路径
        logger: logger实例
        config: 配置字典
        
    Returns:
        tuple: (model, model_name, epoch, step, total_steps)
    """
    try:
        # 加载检查点并提取信息
        model_state_dict, model_name, epoch, step, total_steps = load_checkpoint(checkpoint_path, logger)
        
        # 创建模型
        model_config = config.get('AdversarialModel', {})
        model = create_adversarial_model_wrapper(logger, model_config)
        
        # 加载状态字典
        model.load_state_dict(model_state_dict['model_state_dict'])
        logger.info(f"Model state dict loaded successfully")
        
        return model, model_name, epoch, step, total_steps
        
    except Exception as e:
        logger.error(f"Failed to load model from checkpoint: {e}")
        raise e


def test_adversarial_model(model, test_loader, config, logger):
    """
    测试对抗学习模型
    
    Args:
        model: 模型实例
        test_loader: 测试数据加载器
        config: 配置字典
        logger: logger实例
        
    Returns:
        trainer: 带有测试结果的训练器实例
    """
    try:
        # 创建虚拟训练加载器用于测试（在验证中不使用）
        dummy_train_loader = test_loader
        trainer = AdversarialTrainer(
            config=config, 
            model=model, 
            dataloader_train=dummy_train_loader,
            dataloader_eval=test_loader,
            logger=logger,
            save_dir=config.get('AdversarialTrain', {}).get('save_dir', './outputs')
        )
        
        # 运行验证
        val_results = trainer.validate()
        logger.info("Adversarial model testing completed successfully")
        return trainer
    except Exception as e:
        logger.error(f"Failed to test adversarial model: {e}")
        raise e


def evaluate_adversarial_robustness(model, test_loader, config, logger):
    """
    评估对抗学习模型的鲁棒性
    
    Args:
        model: 模型实例
        test_loader: 测试数据加载器
        config: 配置字典
        logger: logger实例
        
    Returns:
        dict: 鲁棒性评估结果
    """
    try:
        model.eval()
        device = torch.device(config.get('AdversarialTrain', {}).get('device', 'cuda:2'))
        
        clean_accuracy = 0.0
        adversarial_accuracy = 0.0
        total_samples = 0
        
        attack_config = config.get('AdversarialAttack', {})
        epsilon_range = attack_config.get('epsilon_range', [0.01, 0.1, 0.2])
        
        results = {}
        
        for epsilon in epsilon_range:
            epsilon_clean_accuracy = 0.0
            epsilon_adversarial_accuracy = 0.0
            epsilon_total_samples = 0
            
            with torch.no_grad():
                for batch_data, batch_target in test_loader:
                    batch_data = batch_data.to(device)
                    batch_target = batch_target.to(device)
                    
                    # 干净样本预测
                    clean_predictions = model(batch_data, mode='predict')['predictions']
                    clean_loss = F.mse_loss(clean_predictions, batch_target)
                    epsilon_clean_accuracy += clean_loss.item()
                    
                    # 生成对抗样本
                    adversarial_data = model.generate_adversarial_samples(
                        batch_data, batch_target, attack_type='fgsm'
                    )
                    
                    # 对抗样本预测
                    adversarial_predictions = model(adversarial_data, mode='predict')['predictions']
                    adversarial_loss = F.mse_loss(adversarial_predictions, batch_target)
                    epsilon_adversarial_accuracy += adversarial_loss.item()
                    
                    epsilon_total_samples += 1
                    
            epsilon_clean_accuracy /= epsilon_total_samples
            epsilon_adversarial_accuracy /= epsilon_total_samples
            
            results[f'epsilon_{epsilon}'] = {
                'clean_accuracy': epsilon_clean_accuracy,
                'adversarial_accuracy': epsilon_adversarial_accuracy,
                'robustness_gap': epsilon_clean_accuracy - epsilon_adversarial_accuracy
            }
            
            logger.info(f"Epsilon {epsilon}: Clean Loss: {epsilon_clean_accuracy:.6f}, "
                       f"Adversarial Loss: {epsilon_adversarial_accuracy:.6f}, "
                       f"Robustness Gap: {epsilon_clean_accuracy - epsilon_adversarial_accuracy:.6f}")
        
        model.train()
        return results
        
    except Exception as e:
        logger.error(f"Failed to evaluate adversarial robustness: {e}")
        raise e


@Timer
@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """
    主函数 - 使用Hydra配置管理
    
    Args:
        cfg: Hydra配置对象
    """
    start_time = time.time()
    
    # 设置随机种子以确保可重现性
    set_seed()
    
    # 打印配置信息
    print("=== Current Adversarial Learning Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    
    # 初始化日志器
    logger = set_logger(cfg)
    logger.info("Starting adversarial learning main execution...")
    
    try:
        # 创建数据加载器
        logger.info("Creating data loaders...")
        dataloader_train, dataloader_test = create_dataloaders(cfg, logger)
        
        # 获取训练配置
        train_config = cfg.get('AdversarialTrain', {})
        save_dir = train_config.get('save_dir', './outputs')
        load_model_path = train_config.get('load_model_path', None)
        
        # 创建输出目录
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 从检查点加载模型或创建新模型
        if load_model_path and Path(load_model_path).exists():
            logger.info(f"Loading adversarial model from checkpoint: {load_model_path}")
            model, model_name, epoch, step, total_steps = load_model_from_checkpoint(
                load_model_path, logger, cfg
            )
            logger.info(f"Model loaded - Name: {model_name}, Epoch: {epoch}, Step: {step}")
        else:
            # 获取模型配置
            model_config = cfg.get('AdversarialModel', {})
            
            # 创建新模型
            logger.info(f"Creating new adversarial model: {model_config.get('name', 'adversarial_model')}")
            model = create_adversarial_model_wrapper(logger, model_config)
        
        # 训练模型
        logger.info("Starting adversarial model training...")
        trainer = train_adversarial_model(
            model=model,
            train_loader=dataloader_train,
            eval_loader=dataloader_test,
            save_model_dir=save_dir,
            config=cfg,
            logger=logger
        )
        
        # 测试模型（可选）
        test_enabled = train_config.get('test_after_training', False)
        if test_enabled:
            logger.info("Running adversarial model testing...")
            test_trainer = test_adversarial_model(
                model=model,
                test_loader=dataloader_test,
                config=cfg,
                logger=logger
            )
        
        # 评估对抗鲁棒性（可选）
        robustness_test_enabled = cfg.get('Evaluation', {}).get('adversarial_evaluation', {}).get('robustness_test', False)
        if robustness_test_enabled:
            logger.info("Running adversarial robustness evaluation...")
            robustness_results = evaluate_adversarial_robustness(
                model=model,
                test_loader=dataloader_test,
                config=cfg,
                logger=logger
            )
            logger.info(f"Robustness evaluation results: {robustness_results}")
        
        logger.info(f"Adversarial learning main execution completed successfully in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Adversarial learning main execution failed: {e}")
        raise e


if __name__ == '__main__':
    main()
