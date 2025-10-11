"""
对抗数据增强模块
实现对抗样本生成、数据增强策略等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
import random
from abc import ABC, abstractmethod


class AdversarialAugmentation(ABC):
    """对抗数据增强基类"""
    
    @abstractmethod
    def generate_adversarial_samples(self, x: torch.Tensor, y: torch.Tensor, 
                                   model: nn.Module) -> torch.Tensor:
        """生成对抗样本"""
        pass


class FGSMAugmentation(AdversarialAugmentation):
    """FGSM对抗数据增强"""
    
    def __init__(self, epsilon: float = 0.1, targeted: bool = False):
        self.epsilon = epsilon
        self.targeted = targeted
        
    def generate_adversarial_samples(self, x: torch.Tensor, y: torch.Tensor, 
                                   model: nn.Module) -> torch.Tensor:
        """生成FGSM对抗样本"""
        x_adv = x.clone().detach().requires_grad_(True)
        
        # 前向传播
        predictions = model(x_adv)
        loss = F.mse_loss(predictions, y)
        
        # 计算梯度
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        
        # 生成对抗样本
        if self.targeted:
            x_adv = x_adv - self.epsilon * grad.sign()
        else:
            x_adv = x_adv + self.epsilon * grad.sign()
            
        return x_adv.detach()


class PGDAugmentation(AdversarialAugmentation):
    """PGD对抗数据增强"""
    
    def __init__(self, epsilon: float = 0.1, alpha: float = 0.01, 
                 num_iter: int = 7, targeted: bool = False):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.targeted = targeted
        
    def generate_adversarial_samples(self, x: torch.Tensor, y: torch.Tensor, 
                                   model: nn.Module) -> torch.Tensor:
        """生成PGD对抗样本"""
        x_adv = x.clone().detach()
        
        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            
            # 前向传播
            predictions = model(x_adv)
            loss = F.mse_loss(predictions, y)
            
            # 计算梯度
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            
            # 更新对抗样本
            if self.targeted:
                x_adv = x_adv - self.alpha * grad.sign()
            else:
                x_adv = x_adv + self.alpha * grad.sign()
                
            # 投影到epsilon球内
            delta = x_adv - x
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            x_adv = x + delta
            
            x_adv = x_adv.detach()
            
        return x_adv


class CWAugmentation(AdversarialAugmentation):
    """CW对抗数据增强"""
    
    def __init__(self, c: float = 1.0, kappa: float = 0.0, 
                 num_iter: int = 1000, lr: float = 0.01):
        self.c = c
        self.kappa = kappa
        self.num_iter = num_iter
        self.lr = lr
        
    def generate_adversarial_samples(self, x: torch.Tensor, y: torch.Tensor, 
                                   model: nn.Module) -> torch.Tensor:
        """生成CW对抗样本"""
        x_adv = x.clone().detach().requires_grad_(True)
        
        # 使用Adam优化器
        optimizer = torch.optim.Adam([x_adv], lr=self.lr)
        
        for _ in range(self.num_iter):
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(x_adv)
            
            # CW损失
            f_loss = torch.clamp(predictions - y + self.kappa, min=0.0)
            cw_loss = torch.sum(f_loss)
            
            # L2正则化
            l2_loss = torch.norm(x_adv - x, p=2)
            total_loss = cw_loss + self.c * l2_loss
            
            total_loss.backward()
            optimizer.step()
            
        return x_adv.detach()


class NoiseAugmentation:
    """噪声数据增强"""
    
    def __init__(self, noise_std: float = 0.01, noise_type: str = 'gaussian'):
        self.noise_std = noise_std
        self.noise_type = noise_type
        
    def augment(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """添加噪声增强"""
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(x) * self.noise_std
        elif self.noise_type == 'uniform':
            noise = (torch.rand_like(x) - 0.5) * 2 * self.noise_std
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")
            
        x_aug = x + noise
        return x_aug, y


class SpatialAugmentation:
    """空间变换数据增强"""
    
    def __init__(self, spatial_scale: float = 0.05, rotation_angle: float = 0.1):
        self.spatial_scale = spatial_scale
        self.rotation_angle = rotation_angle
        
    def augment(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """空间变换增强"""
        batch_size = x.shape[0]
        
        # 随机缩放
        scale = 1.0 + (torch.rand(batch_size, 1, device=x.device) - 0.5) * 2 * self.spatial_scale
        x_aug = x * scale
        
        # 随机旋转（简化版本）
        if x.shape[1] >= 2:
            angle = (torch.rand(batch_size, 1, device=x.device) - 0.5) * 2 * self.rotation_angle
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            
            # 2D旋转矩阵
            rotation_matrix = torch.stack([
                torch.stack([cos_a, -sin_a], dim=2),
                torch.stack([sin_a, cos_a], dim=2)
            ], dim=2)
            
            # 应用旋转
            x_2d = x_aug[:, :2].unsqueeze(2)
            x_rotated = torch.bmm(rotation_matrix, x_2d).squeeze(2)
            x_aug[:, :2] = x_rotated
            
        return x_aug, y


class DropoutAugmentation:
    """Dropout数据增强"""
    
    def __init__(self, dropout_rate: float = 0.1):
        self.dropout_rate = dropout_rate
        
    def augment(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dropout增强"""
        mask = torch.rand_like(x) > self.dropout_rate
        x_aug = x * mask.float()
        return x_aug, y


class PhysicsAugmentation:
    """物理约束数据增强"""
    
    def __init__(self, temperature_scale: float = 0.1, velocity_scale: float = 0.05, 
                 pressure_scale: float = 0.1):
        self.temperature_scale = temperature_scale
        self.velocity_scale = velocity_scale
        self.pressure_scale = pressure_scale
        
    def augment(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """物理约束增强"""
        y_aug = y.clone()
        
        if y.shape[1] >= 4:  # 假设有温度、速度、压力等
            # 温度增强
            temp_noise = (torch.rand_like(y[:, 0:1]) - 0.5) * 2 * self.temperature_scale
            y_aug[:, 0:1] = y[:, 0:1] + temp_noise
            
            # 速度增强
            vel_noise = (torch.rand_like(y[:, 1:3]) - 0.5) * 2 * self.velocity_scale
            y_aug[:, 1:3] = y[:, 1:3] + vel_noise
            
            # 压力增强
            pressure_noise = (torch.rand_like(y[:, 3:4]) - 0.5) * 2 * self.pressure_scale
            y_aug[:, 3:4] = y[:, 3:4] + pressure_noise
            
        return x, y_aug


class GenerativeAugmentation:
    """生成式数据增强"""
    
    def __init__(self, generator: nn.Module, num_samples: int = 1000, 
                 mix_ratio: float = 0.4):
        self.generator = generator
        self.num_samples = num_samples
        self.mix_ratio = mix_ratio
        
    def augment(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成式增强"""
        batch_size = x.shape[0]
        num_generated = int(batch_size * self.mix_ratio)
        
        if num_generated > 0:
            # 生成新样本
            with torch.no_grad():
                generated_y = self.generator(x[:num_generated])
                
            # 混合原始样本和生成样本
            x_mixed = torch.cat([x, x[:num_generated]], dim=0)
            y_mixed = torch.cat([y, generated_y], dim=0)
            
            return x_mixed, y_mixed
        else:
            return x, y


class AdversarialAugmentationPipeline:
    """对抗数据增强流水线"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.augmentations = []
        self.adversarial_attacks = []
        
        # 构建增强策略
        self._build_augmentations()
        self._build_adversarial_attacks()
        
    def _build_augmentations(self):
        """构建数据增强策略"""
        aug_config = self.config.get('Augmentation', {})
        
        # 噪声增强
        if aug_config.get('enabled', True):
            noise_config = aug_config.get('noise', {})
            if noise_config.get('enabled', True):
                self.augmentations.append(NoiseAugmentation(
                    noise_std=noise_config.get('noise_std', 0.01),
                    noise_type=noise_config.get('noise_type', 'gaussian')
                ))
        
        # 空间变换增强
        spatial_config = aug_config.get('spatial', {})
        if spatial_config.get('enabled', True):
            self.augmentations.append(SpatialAugmentation(
                spatial_scale=spatial_config.get('spatial_scale', 0.05),
                rotation_angle=spatial_config.get('rotation_angle', 0.1)
            ))
        
        # Dropout增强
        dropout_config = aug_config.get('dropout', {})
        if dropout_config.get('enabled', True):
            self.augmentations.append(DropoutAugmentation(
                dropout_rate=dropout_config.get('dropout_rate', 0.1)
            ))
        
        # 物理约束增强
        physics_config = aug_config.get('physics_augmentation', {})
        if physics_config.get('enabled', True):
            self.augmentations.append(PhysicsAugmentation(
                temperature_scale=physics_config.get('temperature_scale', 0.1),
                velocity_scale=physics_config.get('velocity_scale', 0.05),
                pressure_scale=physics_config.get('pressure_scale', 0.1)
            ))
    
    def _build_adversarial_attacks(self):
        """构建对抗攻击策略"""
        attack_config = self.config.get('AdversarialAttack', {})
        
        if attack_config.get('enabled', True):
            attack_types = attack_config.get('attack_types', ['fgsm', 'pgd'])
            
            for attack_type in attack_types:
                if attack_type == 'fgsm':
                    fgsm_config = attack_config.get('fgsm', {})
                    self.adversarial_attacks.append(FGSMAugmentation(
                        epsilon=fgsm_config.get('epsilon', 0.1),
                        targeted=fgsm_config.get('targeted', False)
                    ))
                elif attack_type == 'pgd':
                    pgd_config = attack_config.get('pgd', {})
                    self.adversarial_attacks.append(PGDAugmentation(
                        epsilon=pgd_config.get('epsilon', 0.1),
                        alpha=pgd_config.get('alpha', 0.01),
                        num_iter=pgd_config.get('num_iter', 7),
                        targeted=pgd_config.get('targeted', False)
                    ))
                elif attack_type == 'cw':
                    cw_config = attack_config.get('cw', {})
                    self.adversarial_attacks.append(CWAugmentation(
                        c=cw_config.get('c', 1.0),
                        kappa=cw_config.get('kappa', 0.0),
                        num_iter=cw_config.get('num_iter', 1000),
                        lr=cw_config.get('lr', 0.01)
                    ))
    
    def augment(self, x: torch.Tensor, y: torch.Tensor, 
                model: Optional[nn.Module] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        数据增强
        
        Args:
            x: 输入特征
            y: 目标值
            model: 模型（用于对抗攻击）
            
        Returns:
            增强后的数据列表
        """
        augmented_data = [(x, y)]  # 原始数据
        
        # 应用数据增强
        for augmentation in self.augmentations:
            x_aug, y_aug = augmentation.augment(x, y)
            augmented_data.append((x_aug, y_aug))
        
        # 应用对抗攻击
        if model is not None:
            for attack in self.adversarial_attacks:
                x_adv = attack.generate_adversarial_samples(x, y, model)
                augmented_data.append((x_adv, y))
        
        return augmented_data
    
    def generate_adversarial_samples(self, x: torch.Tensor, y: torch.Tensor, 
                                   model: nn.Module, num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成对抗样本
        
        Args:
            x: 输入特征
            y: 目标值
            model: 模型
            num_samples: 生成样本数量
            
        Returns:
            对抗样本
        """
        all_x_adv = []
        all_y_adv = []
        
        for _ in range(num_samples // x.shape[0] + 1):
            # 随机选择攻击方法
            if self.adversarial_attacks:
                attack = random.choice(self.adversarial_attacks)
                x_adv = attack.generate_adversarial_samples(x, y, model)
                all_x_adv.append(x_adv)
                all_y_adv.append(y)
        
        # 合并所有对抗样本
        if all_x_adv:
            x_adv_all = torch.cat(all_x_adv, dim=0)[:num_samples]
            y_adv_all = torch.cat(all_y_adv, dim=0)[:num_samples]
            return x_adv_all, y_adv_all
        else:
            return x, y


def create_adversarial_augmentation_pipeline(config: Dict[str, Any]) -> AdversarialAugmentationPipeline:
    """创建对抗数据增强流水线"""
    return AdversarialAugmentationPipeline(config)


def test_adversarial_augmentation():
    """测试对抗数据增强"""
    # 创建测试数据
    batch_size = 32
    input_dim = 3
    output_dim = 4
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, output_dim)
    
    # 创建测试模型
    class TestModel(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], output_dim)
    
    model = TestModel()
    
    # 创建配置
    config = {
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
        }
    }
    
    # 创建增强流水线
    pipeline = create_adversarial_augmentation_pipeline(config)
    
    # 测试数据增强
    augmented_data = pipeline.augment(x, y, model)
    print(f"Generated {len(augmented_data)} augmented samples")
    
    # 测试对抗样本生成
    x_adv, y_adv = pipeline.generate_adversarial_samples(x, y, model, num_samples=100)
    print(f"Generated adversarial samples: {x_adv.shape}, {y_adv.shape}")
    
    # 测试各种增强方法
    noise_aug = NoiseAugmentation(noise_std=0.01)
    x_noise, y_noise = noise_aug.augment(x, y)
    print(f"Noise augmentation: {x_noise.shape}, {y_noise.shape}")
    
    spatial_aug = SpatialAugmentation(spatial_scale=0.05)
    x_spatial, y_spatial = spatial_aug.augment(x, y)
    print(f"Spatial augmentation: {x_spatial.shape}, {y_spatial.shape}")
    
    dropout_aug = DropoutAugmentation(dropout_rate=0.1)
    x_dropout, y_dropout = dropout_aug.augment(x, y)
    print(f"Dropout augmentation: {x_dropout.shape}, {y_dropout.shape}")
    
    physics_aug = PhysicsAugmentation()
    x_physics, y_physics = physics_aug.augment(x, y)
    print(f"Physics augmentation: {x_physics.shape}, {y_physics.shape}")
    
    # 测试对抗攻击
    fgsm_attack = FGSMAugmentation(epsilon=0.1)
    x_fgsm = fgsm_attack.generate_adversarial_samples(x, y, model)
    print(f"FGSM attack: {x_fgsm.shape}")
    
    pgd_attack = PGDAugmentation(epsilon=0.1, alpha=0.01, num_iter=3)
    x_pgd = pgd_attack.generate_adversarial_samples(x, y, model)
    print(f"PGD attack: {x_pgd.shape}")
    
    print("Adversarial augmentation test completed successfully!")


if __name__ == "__main__":
    test_adversarial_augmentation()
