"""
对比学习增强预测器主程序
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import Logger
from src.utils.timer import Timer
from src.utils.set_seed import set_seed
from src.data.data_manager import DataManager
from .contrastive_model import create_contrastive_model
from .contrastive_trainer import ContrastiveTrainer


def set_logger(config: DictConfig) -> Logger:
    """设置日志器"""
    logger = Logger(name='contrastive_main', config=config)
    return logger


def create_contrastive_dataloaders(config: DictConfig, logger: Logger):
    """创建对比学习数据加载器"""
    try:
        # 使用主配置中的数据配置
        data_config = config.get('Data', {})
        data_manager = DataManager(data_config, logger, data_config.get('data_path'))
        
        logger.info("Contrastive learning data loaders created successfully")
        logger.info(f"Train loader: {len(data_manager.dataloader_train.dataset)} samples")
        logger.info(f"Test loader: {len(data_manager.dataloader_test.dataset)} samples")
        
        return data_manager.dataloader_train, data_manager.dataloader_test
        
    except Exception as e:
        logger.error(f"Failed to create contrastive learning data loaders: {e}")
        raise e


def create_contrastive_model_from_config(config: DictConfig, logger: Logger):
    """从配置创建对比学习模型"""
    try:
        model_config = config.get('ContrastiveModel', {})
        model = create_contrastive_model(model_config)
        
        logger.info(f"Contrastive learning model created successfully: {model.name}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to create contrastive learning model: {e}")
        raise e


def train_contrastive_model(model, train_loader, eval_loader, config: DictConfig, logger: Logger):
    """训练对比学习模型"""
    try:
        # 获取保存目录
        save_dir = config.get('ContrastiveTrain', {}).get('save_dir', './outputs/contrastive')
        
        # 创建训练器
        trainer = ContrastiveTrainer(
            config=config,
            model=model,
            dataloader_train=train_loader,
            dataloader_eval=eval_loader,
            logger=logger,
            save_dir=save_dir
        )
        
        logger.info("Contrastive learning training started...")
        
        # 开始训练
        results = trainer.train()
        
        logger.info("Contrastive learning training completed successfully")
        return trainer, results
        
    except Exception as e:
        logger.error(f"Failed to train contrastive learning model: {e}")
        raise e


@Timer
@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    对比学习增强预测器主函数
    
    Args:
        cfg: Hydra配置对象
    """
    start_time = time.time()
    
    # 设置随机种子
    set_seed()
    
    # 打印配置信息
    print("=== Contrastive Learning Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    
    # 初始化日志器
    logger = set_logger(cfg)
    logger.info("Starting contrastive learning main execution...")
    
    try:
        # 创建数据加载器
        logger.info("Creating contrastive learning data loaders...")
        dataloader_train, dataloader_test = create_contrastive_dataloaders(cfg, logger)
        
        # 创建对比学习模型
        logger.info("Creating contrastive learning model...")
        model = create_contrastive_model_from_config(cfg, logger)
        
        # 训练模型
        logger.info("Starting contrastive learning model training...")
        trainer, results = train_contrastive_model(
            model=model,
            train_loader=dataloader_train,
            eval_loader=dataloader_test,
            config=cfg,
            logger=logger
        )
        
        # 最终验证
        logger.info("Running final validation...")
        final_metrics = trainer.validate()
        logger.info(f"Final validation metrics: {final_metrics}")
        
        # 保存最终结果
        results['final_validation'] = final_metrics
        logger.info(f"Contrastive learning results: {results}")
        
        logger.info(f"Contrastive learning main execution completed successfully in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Contrastive learning main execution failed: {e}")
        raise e


if __name__ == '__main__':
    main()
