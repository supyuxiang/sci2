"""
对比学习模块测试脚本
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import Logger
from .contrastive_model import create_contrastive_model, test_contrastive_model
from .contrastive_loss import test_contrastive_loss
from .augmentation import test_augmentation
from .contrastive_trainer import test_contrastive_trainer


def test_all_modules():
    """测试所有对比学习模块"""
    print("=" * 60)
    print("开始测试对比学习模块")
    print("=" * 60)
    
    # 创建测试配置
    config = {
        'name': 'test_contrastive_model',
        'base_model': 'mlp',
        'input_dim': 3,
        'output_dim': 4,
        'hidden_dim': 64,
        'num_blocks': 2,
        'dropout': 0.1,
        'embedding_dim': 64,
        'projection_dim': 32,
        'temperature': 0.07,
        'projection_head': {
            'hidden_dim': 32,
            'output_dim': 32,
            'num_layers': 2,
            'dropout': 0.1
        },
        'prediction_head': {
            'hidden_dim': 32,
            'output_dim': 4,
            'num_layers': 2,
            'dropout': 0.1
        }
    }
    
    # 创建日志器
    logger = Logger('test_contrastive', config)
    
    try:
        # 测试1: 数据增强模块
        print("\n1. 测试数据增强模块...")
        test_augmentation()
        print("✅ 数据增强模块测试通过")
        
        # 测试2: 对比学习损失函数
        print("\n2. 测试对比学习损失函数...")
        test_contrastive_loss()
        print("✅ 对比学习损失函数测试通过")
        
        # 测试3: 对比学习模型
        print("\n3. 测试对比学习模型...")
        test_contrastive_model()
        print("✅ 对比学习模型测试通过")
        
        # 测试4: 对比学习训练器
        print("\n4. 测试对比学习训练器...")
        test_contrastive_trainer()
        print("✅ 对比学习训练器测试通过")
        
        print("\n" + "=" * 60)
        print("🎉 所有对比学习模块测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        logger.error(f"Contrastive learning test failed: {e}")
        raise e


def test_integration():
    """集成测试"""
    print("\n" + "=" * 60)
    print("开始集成测试")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 16
    input_dim = 3
    output_dim = 4
    num_samples = 100
    
    # 生成测试数据
    x = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, output_dim)
    
    # 创建数据集
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建配置
    config = {
        'ContrastiveModel': {
            'name': 'integration_test_model',
            'base_model': 'mlp',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_dim': 32,
            'num_blocks': 2,
            'dropout': 0.1,
            'embedding_dim': 32,
            'projection_dim': 16
        },
        'ContrastiveTrain': {
            'epochs': 5,
            'pretrain_epochs': 2,
            'finetune_epochs': 3,
            'optimizer': 'adam',
            'optimizer_config': {'lr': 1e-3},
            'scheduler': 'cosine',
            'scheduler_config': {},
            'device': 'cpu',
            'log_interval': 1,
            'save_freq': 2,
            'is_pinn': False
        },
        'Augmentation': {
            'noise': {'enabled': True, 'noise_std': 0.01},
            'spatial': {'enabled': True, 'spatial_scale': 0.05}
        },
        'swanlab': {'use_swanlab': False}
    }
    
    # 创建日志器
    logger = Logger('integration_test', config)
    
    try:
        # 创建模型
        model = create_contrastive_model(config['ContrastiveModel'])
        logger.info(f"Model created: {model.name}")
        
        # 测试前向传播
        test_batch = next(iter(dataloader))
        x_batch, y_batch = test_batch
        
        # 测试不同模式
        outputs = model(x_batch, mode='predict')
        logger.info(f"Predict mode output shape: {outputs['predictions'].shape}")
        
        outputs = model(x_batch, mode='contrastive')
        logger.info(f"Contrastive mode output shape: {outputs['projections'].shape}")
        
        outputs = model(x_batch, mode='both')
        logger.info(f"Both mode - Predictions: {outputs['predictions'].shape}, "
                   f"Projections: {outputs['projections'].shape}")
        
        # 测试训练模式切换
        model.set_training_mode('pretrain')
        logger.info("Switched to pretrain mode")
        
        model.set_training_mode('finetune')
        logger.info("Switched to finetune mode")
        
        print("✅ 集成测试通过")
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        logger.error(f"Integration test failed: {e}")
        raise e


def test_performance():
    """性能测试"""
    print("\n" + "=" * 60)
    print("开始性能测试")
    print("=" * 60)
    
    import time
    
    # 创建测试数据
    batch_size = 64
    input_dim = 3
    output_dim = 4
    num_samples = 1000
    
    x = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, output_dim)
    
    # 创建配置
    config = {
        'name': 'performance_test_model',
        'base_model': 'mlp',
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dim': 128,
        'num_blocks': 4,
        'dropout': 0.1,
        'embedding_dim': 128,
        'projection_dim': 64
    }
    
    try:
        # 创建模型
        model = create_contrastive_model(config)
        
        # 测试推理速度
        num_iterations = 100
        
        # 预热
        for _ in range(10):
            _ = model(x[:batch_size], mode='both')
        
        # 测试预测模式速度
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model(x[:batch_size], mode='predict')
        predict_time = time.time() - start_time
        
        # 测试对比学习模式速度
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model(x[:batch_size], mode='contrastive')
        contrastive_time = time.time() - start_time
        
        # 测试两种模式速度
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model(x[:batch_size], mode='both')
        both_time = time.time() - start_time
        
        print(f"预测模式平均时间: {predict_time/num_iterations*1000:.2f} ms")
        print(f"对比学习模式平均时间: {contrastive_time/num_iterations*1000:.2f} ms")
        print(f"两种模式平均时间: {both_time/num_iterations*1000:.2f} ms")
        
        # 测试内存使用
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建更大的模型进行内存测试
        large_config = config.copy()
        large_config['hidden_dim'] = 512
        large_config['num_blocks'] = 8
        
        large_model = create_contrastive_model(large_config)
        _ = large_model(x[:batch_size], mode='both')
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        print(f"大模型内存使用: {memory_used:.2f} MB")
        
        print("✅ 性能测试完成")
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        raise e


def main():
    """主测试函数"""
    print("对比学习模块完整测试")
    print("=" * 60)
    
    try:
        # 运行所有测试
        test_all_modules()
        test_integration()
        test_performance()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成！对比学习模块工作正常！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
