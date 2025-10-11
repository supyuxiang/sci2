"""
对比学习数据增强模块
实现多种数据增强策略，包括噪声注入、空间变换、物理约束增强等
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import random


class BaseAugmentation:
    """基础数据增强类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用数据增强"""
        if not self.enabled:
            return x, y
        return self.augment(x, y)
    
    def augment(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """子类需要实现的具体增强方法"""
        raise NotImplementedError


class NoiseAugmentation(BaseAugmentation):
    """噪声注入增强"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.noise_std = config.get('noise_std', 0.01)
        
    def augment(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """添加高斯噪声"""
        if not self.enabled:
            return x, y
            
        # 为输入特征添加噪声
        noise = torch.randn_like(x) * self.noise_std
        x_aug = x + noise
        
        return x_aug, y


class SpatialAugmentation(BaseAugmentation):
    """空间变换增强"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scale = config.get('spatial_scale', 0.05)
        
    def augment(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """对空间坐标进行小幅变换"""
        if not self.enabled:
            return x, y
            
        # 假设前3列是空间坐标 (x, y, z)
        x_aug = x.clone()
        if x.shape[1] >= 3:
            # 对空间坐标添加随机扰动
            spatial_noise = torch.randn(x.shape[0], 3, device=x.device) * self.scale
            x_aug[:, :3] += spatial_noise
            
        return x_aug, y


class DropoutAugmentation(BaseAugmentation):
    """随机丢弃增强"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
    def augment(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机将部分特征置零"""
        if not self.enabled:
            return x, y
            
        x_aug = x.clone()
        mask = torch.rand_like(x) > self.dropout_rate
        x_aug = x_aug * mask.float()
        
        return x_aug, y


class PhysicsAugmentation(BaseAugmentation):
    """物理约束增强"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.temperature_scale = config.get('temperature_scale', 0.1)
        self.velocity_scale = config.get('velocity_scale', 0.05)
        self.pressure_scale = config.get('pressure_scale', 0.1)
        
    def augment(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于物理定律的增强"""
        if not self.enabled or y is None:
            return x, y
            
        # 假设目标变量的顺序: [T, U, u, p] (温度, 速度分量, 压力)
        y_aug = y.clone()
        
        # 对温度添加小幅变化
        if y.shape[1] >= 1:
            temp_noise = torch.randn_like(y[:, 0:1]) * self.temperature_scale
            y_aug[:, 0:1] += temp_noise
            
        # 对速度添加小幅变化
        if y.shape[1] >= 3:
            vel_noise = torch.randn_like(y[:, 1:3]) * self.velocity_scale
            y_aug[:, 1:3] += vel_noise
            
        # 对压力添加小幅变化
        if y.shape[1] >= 4:
            press_noise = torch.randn_like(y[:, 3:4]) * self.pressure_scale
            y_aug[:, 3:4] += press_noise
            
        return x, y_aug


class TemporalAugmentation(BaseAugmentation):
    """时间序列增强"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.time_shift = config.get('time_shift', 0.1)
        self.time_scale = config.get('time_scale', 0.05)
        
    def augment(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """时间序列增强（如果数据是时间序列）"""
        if not self.enabled:
            return x, y
            
        # 这里可以实现时间序列特定的增强
        # 例如时间偏移、时间缩放等
        # 当前实现为简单的噪声添加
        x_aug = x.clone()
        if len(x.shape) > 2:  # 如果是时间序列数据
            time_noise = torch.randn_like(x) * self.time_scale
            x_aug += time_noise
            
        return x_aug, y


class ContrastiveAugmentation:
    """对比学习数据增强组合器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化各种增强方法
        self.noise_aug = NoiseAugmentation(config.get('noise', {}))
        self.spatial_aug = SpatialAugmentation(config.get('spatial', {}))
        self.dropout_aug = DropoutAugmentation(config.get('dropout', {}))
        self.physics_aug = PhysicsAugmentation(config.get('physics_augmentation', {}))
        self.temporal_aug = TemporalAugmentation(config.get('temporal_augmentation', {}))
        
        # 增强方法列表
        self.augmentations = [
            self.noise_aug,
            self.spatial_aug,
            self.dropout_aug,
            self.physics_aug,
            self.temporal_aug
        ]
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor = None, 
                 num_views: int = 2) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        生成多个增强视图
        
        Args:
            x: 输入特征
            y: 目标变量
            num_views: 生成的视图数量
            
        Returns:
            List of (augmented_x, augmented_y) tuples
        """
        views = []
        
        # 原始视图
        views.append((x, y))
        
        # 生成增强视图
        for i in range(num_views - 1):
            x_aug = x.clone()
            y_aug = y.clone() if y is not None else None
            
            # 随机选择和应用增强方法
            for aug in self.augmentations:
                if random.random() < 0.5:  # 50%概率应用每种增强
                    x_aug, y_aug = aug(x_aug, y_aug)
                    
            views.append((x_aug, y_aug))
            
        return views
    
    def generate_positive_pairs(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成正样本对"""
        views = self(x, y, num_views=2)
        return views[0], views[1]
    
    def generate_negative_pairs(self, x: torch.Tensor, y: torch.Tensor = None, 
                               batch_size: int = 64) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """生成负样本对"""
        # 随机打乱批次中的样本
        indices = torch.randperm(x.shape[0])
        x_neg = x[indices]
        y_neg = y[indices] if y is not None else None
        
        # 生成增强视图
        views = self(x_neg, y_neg, num_views=1)
        return [(x, views[0][0])]  # 返回 (anchor, negative) 对


class MultiScaleAugmentation(BaseAugmentation):
    """多尺度增强"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scales = config.get('scales', [1.0, 0.5, 2.0])
        self.weights = config.get('weights', [0.5, 0.3, 0.2])
        
    def augment(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """多尺度增强"""
        if not self.enabled:
            return x, y
            
        # 随机选择一个尺度
        scale = random.choices(self.scales, weights=self.weights)[0]
        
        # 应用尺度变换
        x_aug = x * scale
        y_aug = y * scale if y is not None else y
        
        return x_aug, y_aug


def create_augmentation_pipeline(config: Dict[str, Any]) -> ContrastiveAugmentation:
    """创建数据增强管道"""
    return ContrastiveAugmentation(config)


def test_augmentation():
    """测试数据增强功能"""
    # 创建测试数据
    batch_size = 32
    input_dim = 3
    output_dim = 4
    
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, output_dim)
    
    # 创建增强配置
    config = {
        'noise': {'enabled': True, 'noise_std': 0.01},
        'spatial': {'enabled': True, 'spatial_scale': 0.05},
        'dropout': {'enabled': True, 'dropout_rate': 0.1},
        'physics_augmentation': {'enabled': True, 'temperature_scale': 0.1},
        'temporal_augmentation': {'enabled': False}
    }
    
    # 创建增强器
    augmenter = ContrastiveAugmentation(config)
    
    # 测试生成多个视图
    views = augmenter(x, y, num_views=3)
    print(f"Generated {len(views)} views")
    print(f"Original shape: {x.shape}")
    print(f"Augmented shape: {views[1][0].shape}")
    
    # 测试正样本对生成
    pos_pair = augmenter.generate_positive_pairs(x, y)
    print(f"Positive pair shapes: {pos_pair[0].shape}, {pos_pair[1].shape}")
    
    print("Augmentation test completed successfully!")


if __name__ == "__main__":
    test_augmentation()
