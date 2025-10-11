"""
对抗学习模块测试文件
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import Logger
from .adversarial_models import create_adversarial_model, test_adversarial_model
from .adversarial_losses import test_adversarial_loss
from .adversarial_augmentation import test_adversarial_augmentation
from .adversarial_trainer import test_adversarial_trainer


def test_adversarial_learning_module():
    """测试对抗学习模块的完整功能"""
    print("=" * 60)
    print("Testing Adversarial Learning Module")
    print("=" * 60)
    
    # 创建测试配置
    config = {
        'AdversarialModel': {
            'name': 'test_adversarial_model',
            'input_dim': 3,
            'output_dim': 4,
            'hidden_dim': 64,
            'num_blocks': 4,
            'dropout': 0.1,
            'generator': {
                'hidden_dim': 64,
                'num_layers': 4,
                'dropout': 0.1,
                'noise_dim': 16,
                'output_activation': 'tanh'
            },
            'discriminator': {
                'hidden_dim': 64,
                'num_layers': 3,
                'dropout': 0.1,
                'output_activation': 'sigmoid'
            },
            'predictor': {
                'hidden_dim': 64,
                'num_layers': 4,
                'dropout': 0.1,
                'output_activation': 'none'
            },
            'adversarial': {
                'epsilon': 0.1,
                'num_adv_steps': 5,
                'adv_lr': 0.01,
                'pgd_steps': 3,
                'fgsm_alpha': 0.01
            }
        },
        'AdversarialTrain': {
            'epochs': 10,
            'pretrain_epochs': 3,
            'adversarial_epochs': 4,
            'finetune_epochs': 3,
            'optimizer': 'adam',
            'optimizer_config': {'lr': 1e-3},
            'scheduler': 'cosine',
            'scheduler_config': {},
            'device': 'cpu',
            'log_interval': 1,
            'save_freq': 5,
            'is_pinn': False,
            'gan_training': {
                'd_steps': 1,
                'g_steps': 1,
                'd_lr': 1e-4,
                'g_lr': 1e-4
            }
        },
        'Augmentation': {
            'enabled': True,
            'noise': {'enabled': True, 'noise_std': 0.01, 'noise_type': 'gaussian'},
            'spatial': {'enabled': True, 'spatial_scale': 0.05, 'rotation_angle': 0.1},
            'dropout': {'enabled': True, 'dropout_rate': 0.1},
            'physics_augmentation': {
                'enabled': True,
                'temperature_scale': 0.1,
                'velocity_scale': 0.05,
                'pressure_scale': 0.1
            }
        },
        'AdversarialAttack': {
            'enabled': True,
            'attack_types': ['fgsm', 'pgd'],
            'fgsm': {'epsilon': 0.1, 'targeted': False},
            'pgd': {'epsilon': 0.1, 'alpha': 0.01, 'num_iter': 3, 'targeted': False}
        },
        'AdversarialLoss': {
            'gan_loss': {'type': 'bce', 'label_smoothing': 0.1},
            'adversarial_loss': {'type': 'mse', 'epsilon': 0.1},
            'physics_loss': {'enabled': True, 'weight': 0.2, 'constraint_types': ['conservation', 'boundary']},
            'regularization': {'l1_weight': 0.01, 'l2_weight': 0.01, 'gradient_penalty_weight': 10.0}
        },
        'swanlab': {'use_swanlab': False}
    }
    
    # 创建日志器
    logger = Logger('test_adversarial_learning', config)
    
    print("\n1. Testing Adversarial Models...")
    try:
        test_adversarial_model()
        print("✓ Adversarial models test passed")
    except Exception as e:
        print(f"✗ Adversarial models test failed: {e}")
        return False
    
    print("\n2. Testing Adversarial Losses...")
    try:
        test_adversarial_loss()
        print("✓ Adversarial losses test passed")
    except Exception as e:
        print(f"✗ Adversarial losses test failed: {e}")
        return False
    
    print("\n3. Testing Adversarial Augmentation...")
    try:
        test_adversarial_augmentation()
        print("✓ Adversarial augmentation test passed")
    except Exception as e:
        print(f"✗ Adversarial augmentation test failed: {e}")
        return False
    
    print("\n4. Testing Adversarial Trainer...")
    try:
        test_adversarial_trainer()
        print("✓ Adversarial trainer test passed")
    except Exception as e:
        print(f"✗ Adversarial trainer test failed: {e}")
        return False
    
    print("\n5. Testing Model Creation...")
    try:
        model = create_adversarial_model(config['AdversarialModel'])
        print(f"✓ Model created successfully: {model.name}")
        
        # 测试模型前向传播
        batch_size = 16
        input_dim = 3
        x = torch.randn(batch_size, input_dim)
        
        # 测试不同模式
        outputs = model(x, mode='predict')
        print(f"✓ Predict mode: {outputs['predictions'].shape}")
        
        outputs = model(x, mode='generate')
        print(f"✓ Generate mode: {outputs['generated'].shape}")
        
        outputs = model(x, mode='discriminate')
        print(f"✓ Discriminate mode: {outputs['discriminator_output'].shape}")
        
        outputs = model(x, mode='adversarial')
        print(f"✓ Adversarial mode: {len(outputs)} outputs")
        
        # 测试训练模式切换
        model.set_training_mode('pretrain')
        print("✓ Switched to pretrain mode")
        
        model.set_training_mode('adversarial')
        print("✓ Switched to adversarial mode")
        
        model.set_training_mode('finetune')
        print("✓ Switched to finetune mode")
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False
    
    print("\n6. Testing Adversarial Attacks...")
    try:
        # 创建测试数据
        batch_size = 16
        input_dim = 3
        output_dim = 4
        x = torch.randn(batch_size, input_dim)
        y = torch.randn(batch_size, output_dim)
        
        # 测试对抗攻击
        x_adv_fgsm = model.generate_adversarial_samples(x, y, attack_type='fgsm')
        print(f"✓ FGSM attack: {x_adv_fgsm.shape}")
        
        x_adv_pgd = model.generate_adversarial_samples(x, y, attack_type='pgd')
        print(f"✓ PGD attack: {x_adv_pgd.shape}")
        
        x_adv_cw = model.generate_adversarial_samples(x, y, attack_type='cw')
        print(f"✓ CW attack: {x_adv_cw.shape}")
        
    except Exception as e:
        print(f"✗ Adversarial attacks test failed: {e}")
        return False
    
    print("\n7. Testing Loss Functions...")
    try:
        from .adversarial_losses import CombinedAdversarialLoss
        
        loss_fn = CombinedAdversarialLoss(config)
        
        # 创建测试数据
        batch_size = 16
        input_dim = 3
        output_dim = 4
        x = torch.randn(batch_size, input_dim)
        y = torch.randn(batch_size, output_dim)
        y_adv = torch.randn(batch_size, output_dim)
        
        # 测试生成器损失
        discriminator_output_fake = torch.randn(batch_size, 1)
        generator_losses = loss_fn.compute_generator_loss(
            discriminator_output_fake, y, y_adv
        )
        print(f"✓ Generator losses: {len(generator_losses)} components")
        
        # 测试判别器损失
        discriminator_output_real = torch.randn(batch_size, 1)
        discriminator_losses = loss_fn.compute_discriminator_loss(
            discriminator_output_real, discriminator_output_fake, x, y_adv, model.discriminator
        )
        print(f"✓ Discriminator losses: {len(discriminator_losses)} components")
        
        # 测试预测器损失
        predictions = torch.randn(batch_size, output_dim)
        adversarial_predictions = torch.randn(batch_size, output_dim)
        predictor_losses = loss_fn.compute_predictor_loss(
            predictions, y, adversarial_predictions, x
        )
        print(f"✓ Predictor losses: {len(predictor_losses)} components")
        
    except Exception as e:
        print(f"✗ Loss functions test failed: {e}")
        return False
    
    print("\n8. Testing Data Augmentation...")
    try:
        from .adversarial_augmentation import create_adversarial_augmentation_pipeline
        
        pipeline = create_adversarial_augmentation_pipeline(config)
        
        # 创建测试数据
        batch_size = 16
        input_dim = 3
        output_dim = 4
        x = torch.randn(batch_size, input_dim)
        y = torch.randn(batch_size, output_dim)
        
        # 测试数据增强
        augmented_data = pipeline.augment(x, y, model)
        print(f"✓ Data augmentation: {len(augmented_data)} augmented samples")
        
        # 测试对抗样本生成
        x_adv, y_adv = pipeline.generate_adversarial_samples(x, y, model, num_samples=50)
        print(f"✓ Adversarial sample generation: {x_adv.shape}, {y_adv.shape}")
        
    except Exception as e:
        print(f"✗ Data augmentation test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All Adversarial Learning Module Tests Passed! ✓")
    print("=" * 60)
    
    return True


def test_integration_with_main_project():
    """测试与主项目的集成"""
    print("\n" + "=" * 60)
    print("Testing Integration with Main Project")
    print("=" * 60)
    
    try:
        # 测试导入主项目的模块
        from src.data.data_manager import DataManager
        from src.utils.logger import Logger
        from src.utils.optimizer import build_optimizer
        from src.utils.scheduler import build_scheduler
        from src.utils.Metrics import MetricsManager
        
        print("✓ Successfully imported main project modules")
        
        # 测试配置兼容性
        config = {
            'Data': {
                'data_path': "/home/yxfeng/project2/sci2/data/数据100mm流体域.xlsx",
                'sheet_name': "数据100mm流体域",
                'format': "auto",
                'sheet': 0,
                'index_start': 8,
                'features': ["% x", "y", "z"],
                'targets': ["T (K)", "spf.U (m/s)", "u (m/s)", "p (Pa)"],
                'test_ratio': 0.2,
                'random_state': 42,
                'scaler_name': "standardscaler",
                'batch_size': 64
            },
            'AdversarialModel': {
                'name': 'integration_test_model',
                'input_dim': 3,
                'output_dim': 4,
                'hidden_dim': 64,
                'num_blocks': 4,
                'dropout': 0.1
            }
        }
        
        logger = Logger('integration_test', config)
        print("✓ Successfully created logger with main project integration")
        
        # 测试模型创建
        model = create_adversarial_model(config['AdversarialModel'])
        print(f"✓ Successfully created adversarial model: {model.name}")
        
        print("\n✓ Integration with main project successful!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    # 运行所有测试
    success = True
    
    # 测试对抗学习模块
    success &= test_adversarial_learning_module()
    
    # 测试与主项目的集成
    success &= test_integration_with_main_project()
    
    if success:
        print("\n🎉 All tests passed! Adversarial learning module is ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
