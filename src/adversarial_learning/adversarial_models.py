"""
对抗学习增强预测器模型
实现生成器、判别器和对抗增强预测器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.mlp import MLP
from src.models.lstm import LSTM
from src.models.gru import GRURegressor
from src.models.transformer import TransformerEncoderRegressor
from src.models.cnn1d import CNNRegressor1D


class Generator(nn.Module):
    """生成器：生成对抗样本或增强数据"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.name = 'generator'
        
        # 生成器配置
        gen_config = config.get('generator', {})
        input_dim = config.get('input_dim', 3)
        output_dim = config.get('output_dim', 4)
        hidden_dim = gen_config.get('hidden_dim', 128)
        num_layers = gen_config.get('num_layers', 6)
        dropout = gen_config.get('dropout', 0.1)
        noise_dim = gen_config.get('noise_dim', 32)
        output_activation = gen_config.get('output_activation', 'tanh')
        
        # 输入维度：原始特征 + 噪声
        total_input_dim = input_dim + noise_dim
        
        # 构建生成器网络
        layers = []
        current_dim = total_input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        
        # 输出激活函数
        if output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        # 'none' 表示不使用激活函数
            
        self.generator = nn.Sequential(*layers)
        self.noise_dim = noise_dim
        
    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            noise: 噪声 [batch_size, noise_dim]
            
        Returns:
            生成的输出 [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # 生成噪声
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=x.device)
            
        # 拼接输入和噪声
        combined_input = torch.cat([x, noise], dim=1)
        
        # 生成输出
        generated = self.generator(combined_input)
        
        return generated
    
    def generate_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """生成随机噪声"""
        return torch.randn(batch_size, self.noise_dim, device=device)


class Discriminator(nn.Module):
    """判别器：区分真实样本和生成样本"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.name = 'discriminator'
        
        # 判别器配置
        disc_config = config.get('discriminator', {})
        input_dim = config.get('input_dim', 3)
        output_dim = config.get('output_dim', 4)
        hidden_dim = disc_config.get('hidden_dim', 128)
        num_layers = disc_config.get('num_layers', 4)
        dropout = disc_config.get('dropout', 0.1)
        output_activation = disc_config.get('output_activation', 'sigmoid')
        
        # 输入维度：特征 + 目标
        total_input_dim = input_dim + output_dim
        
        # 构建判别器网络
        layers = []
        current_dim = total_input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(current_dim, 1))
        
        # 输出激活函数
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        # 'none' 表示不使用激活函数
            
        self.discriminator = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            y: 目标值 [batch_size, output_dim]
            
        Returns:
            判别结果 [batch_size, 1]
        """
        # 拼接特征和目标
        combined_input = torch.cat([x, y], dim=1)
        
        # 判别
        output = self.discriminator(combined_input)
        
        return output


class AdversarialPredictor(nn.Module):
    """对抗增强预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.name = 'adversarial_predictor'
        
        # 预测器配置
        pred_config = config.get('predictor', {})
        input_dim = config.get('input_dim', 3)
        output_dim = config.get('output_dim', 4)
        hidden_dim = pred_config.get('hidden_dim', 128)
        num_layers = pred_config.get('num_layers', 6)
        dropout = pred_config.get('dropout', 0.1)
        output_activation = pred_config.get('output_activation', 'none')
        
        # 构建预测器网络
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        
        # 输出激活函数
        if output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        # 'none' 表示不使用激活函数
            
        self.predictor = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            预测结果 [batch_size, output_dim]
        """
        return self.predictor(x)


class AdversarialEnhancedPredictor(nn.Module):
    """对抗学习增强预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.name = config.get('name', 'adversarial_enhanced_predictor')
        
        # 创建生成器、判别器和预测器
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.predictor = AdversarialPredictor(config)
        
        # 对抗训练参数
        adv_config = config.get('adversarial', {})
        self.epsilon = adv_config.get('epsilon', 0.1)
        self.num_adv_steps = adv_config.get('num_adv_steps', 10)
        self.adv_lr = adv_config.get('adv_lr', 0.01)
        self.pgd_steps = adv_config.get('pgd_steps', 7)
        self.fgsm_alpha = adv_config.get('fgsm_alpha', 0.01)
        
        # 训练模式
        self.training_mode = 'pretrain'  # 'pretrain', 'adversarial', 'finetune'
        
    def forward(self, x: torch.Tensor, mode: str = 'predict') -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征
            mode: 模式 ('predict', 'generate', 'discriminate', 'adversarial')
            
        Returns:
            输出字典
        """
        outputs = {}
        
        if mode == 'predict':
            # 预测模式
            predictions = self.predictor(x)
            outputs['predictions'] = predictions
            
        elif mode == 'generate':
            # 生成模式
            generated = self.generator(x)
            outputs['generated'] = generated
            
        elif mode == 'discriminate':
            # 判别模式
            predictions = self.predictor(x)
            discriminator_output = self.discriminator(x, predictions)
            outputs['predictions'] = predictions
            outputs['discriminator_output'] = discriminator_output
            
        elif mode == 'adversarial':
            # 对抗模式
            predictions = self.predictor(x)
            generated = self.generator(x)
            discriminator_output = self.discriminator(x, predictions)
            outputs['predictions'] = predictions
            outputs['generated'] = generated
            outputs['discriminator_output'] = discriminator_output
            
        return outputs
    
    def generate_adversarial_samples(self, x: torch.Tensor, y: torch.Tensor, 
                                   attack_type: str = 'fgsm') -> torch.Tensor:
        """
        生成对抗样本
        
        Args:
            x: 原始输入
            y: 目标标签
            attack_type: 攻击类型 ('fgsm', 'pgd', 'cw')
            
        Returns:
            对抗样本
        """
        if attack_type == 'fgsm':
            return self._fgsm_attack(x, y)
        elif attack_type == 'pgd':
            return self._pgd_attack(x, y)
        elif attack_type == 'cw':
            return self._cw_attack(x, y)
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")
    
    def _fgsm_attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """FGSM攻击"""
        x_adv = x.clone().detach().requires_grad_(True)
        
        # 前向传播
        predictions = self.predictor(x_adv)
        loss = F.mse_loss(predictions, y)
        
        # 计算梯度
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        
        # 生成对抗样本
        x_adv = x_adv + self.epsilon * grad.sign()
        
        return x_adv.detach()
    
    def _pgd_attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """PGD攻击"""
        x_adv = x.clone().detach()
        
        for _ in range(self.pgd_steps):
            x_adv.requires_grad_(True)
            
            # 前向传播
            predictions = self.predictor(x_adv)
            loss = F.mse_loss(predictions, y)
            
            # 计算梯度
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            
            # 更新对抗样本
            x_adv = x_adv + self.fgsm_alpha * grad.sign()
            
            # 投影到epsilon球内
            delta = x_adv - x
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            x_adv = x + delta
            
            x_adv = x_adv.detach()
            
        return x_adv
    
    def _cw_attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """CW攻击"""
        # 简化的CW攻击实现
        x_adv = x.clone().detach().requires_grad_(True)
        
        # 使用Adam优化器
        optimizer = torch.optim.Adam([x_adv], lr=0.01)
        
        for _ in range(100):  # 简化版本，减少迭代次数
            optimizer.zero_grad()
            
            # 前向传播
            predictions = self.predictor(x_adv)
            loss = F.mse_loss(predictions, y)
            
            # 添加L2正则化
            l2_loss = torch.norm(x_adv - x, p=2)
            total_loss = loss + 0.1 * l2_loss
            
            total_loss.backward()
            optimizer.step()
            
            # 投影到epsilon球内
            delta = x_adv - x
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            x_adv.data = x + delta
            
        return x_adv.detach()
    
    def set_training_mode(self, mode: str):
        """设置训练模式"""
        assert mode in ['pretrain', 'adversarial', 'finetune'], f"Invalid training mode: {mode}"
        self.training_mode = mode
        
        if mode == 'pretrain':
            # 预训练模式：只训练预测器
            for param in self.generator.parameters():
                param.requires_grad = False
            for param in self.discriminator.parameters():
                param.requires_grad = False
            for param in self.predictor.parameters():
                param.requires_grad = True
        elif mode == 'adversarial':
            # 对抗训练模式：训练所有组件
            for param in self.generator.parameters():
                param.requires_grad = True
            for param in self.discriminator.parameters():
                param.requires_grad = True
            for param in self.predictor.parameters():
                param.requires_grad = True
        else:  # finetune
            # 微调模式：冻结生成器和判别器，训练预测器
            for param in self.generator.parameters():
                param.requires_grad = False
            for param in self.discriminator.parameters():
                param.requires_grad = False
            for param in self.predictor.parameters():
                param.requires_grad = True


class PhysicsAwareAdversarialPredictor(nn.Module):
    """物理感知对抗学习预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.name = 'physics_aware_adversarial_predictor'
        
        # 基础对抗预测器
        self.base_predictor = AdversarialEnhancedPredictor(config)
        
        # 物理约束层
        physics_config = config.get('physics_constraints', {})
        self.physics_enabled = physics_config.get('enabled', True)
        
        if self.physics_enabled:
            # 物理约束网络
            input_dim = config.get('input_dim', 3)
            output_dim = config.get('output_dim', 4)
            self.physics_head = nn.Sequential(
                nn.Linear(input_dim + output_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)  # 物理约束损失
            )
            
    def forward(self, x: torch.Tensor, mode: str = 'predict') -> Dict[str, torch.Tensor]:
        """物理感知前向传播"""
        outputs = self.base_predictor(x, mode)
        
        if self.physics_enabled and mode in ['predict', 'adversarial']:
            # 计算物理约束
            predictions = outputs['predictions']
            combined_input = torch.cat([x, predictions], dim=1)
            physics_constraint = self.physics_head(combined_input)
            outputs['physics_constraint'] = physics_constraint
            
        return outputs
    
    def compute_physics_loss(self, predictions: torch.Tensor, 
                           targets: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """计算物理约束损失"""
        if not self.physics_enabled:
            return torch.tensor(0.0, device=predictions.device)
            
        # 物理约束：预测值应该满足某些物理关系
        if predictions.shape[1] >= 4:  # 假设有温度、速度、压力等
            temp_pred = predictions[:, 0:1]  # 温度
            vel_pred = predictions[:, 1:3]   # 速度分量
            
            # 计算速度大小
            vel_magnitude = torch.norm(vel_pred, dim=1, keepdim=True)
            
            # 物理约束：温度与速度平方的关系
            physics_loss = F.mse_loss(temp_pred, vel_magnitude ** 2)
            
            return physics_loss
        else:
            return torch.tensor(0.0, device=predictions.device)


class MultiScaleAdversarialPredictor(nn.Module):
    """多尺度对抗学习预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.name = 'multi_scale_adversarial_predictor'
        
        # 多尺度配置
        scales = config.get('multi_scale', {}).get('scales', [1.0, 0.5, 2.0])
        self.scales = scales
        
        # 为每个尺度创建预测器
        self.predictors = nn.ModuleList([
            AdversarialEnhancedPredictor(config) for _ in scales
        ])
        
        # 融合层
        input_dim = config.get('input_dim', 3)
        output_dim = config.get('output_dim', 4)
        self.fusion_layer = nn.Linear(input_dim * len(scales), output_dim)
        
    def forward(self, x: torch.Tensor, mode: str = 'predict') -> Dict[str, torch.Tensor]:
        """多尺度前向传播"""
        outputs = {}
        
        # 获取各尺度的特征
        all_features = []
        all_predictions = []
        all_generated = []
        all_discriminator_outputs = []
        
        for i, (scale, predictor) in enumerate(zip(self.scales, self.predictors)):
            # 应用尺度变换
            x_scaled = x * scale
            
            # 前向传播
            pred_outputs = predictor(x_scaled, mode='adversarial')
            
            all_features.append(x_scaled)
            all_predictions.append(pred_outputs['predictions'])
            all_generated.append(pred_outputs['generated'])
            all_discriminator_outputs.append(pred_outputs['discriminator_output'])
            
        # 融合特征
        fused_features = torch.cat(all_features, dim=1)
        fused_predictions = self.fusion_layer(fused_features)
        
        outputs['features'] = fused_features
        outputs['predictions'] = fused_predictions
        outputs['multi_scale_features'] = all_features
        outputs['multi_scale_predictions'] = all_predictions
        outputs['multi_scale_generated'] = all_generated
        outputs['multi_scale_discriminator_outputs'] = all_discriminator_outputs
        
        return outputs


def create_adversarial_model(config: Dict[str, Any]) -> nn.Module:
    """创建对抗学习模型"""
    model_type = config.get('model_type', 'adversarial_enhanced')
    
    if model_type == 'adversarial_enhanced':
        return AdversarialEnhancedPredictor(config)
    elif model_type == 'multi_scale':
        return MultiScaleAdversarialPredictor(config)
    elif model_type == 'physics_aware':
        return PhysicsAwareAdversarialPredictor(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def test_adversarial_model():
    """测试对抗学习模型"""
    # 创建测试配置
    config = {
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
    }
    
    # 创建模型
    model = AdversarialEnhancedPredictor(config)
    
    # 创建测试数据
    batch_size = 32
    input_dim = 3
    output_dim = 4
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, output_dim)
    
    # 测试预测模式
    outputs = model(x, mode='predict')
    print(f"Predictions shape: {outputs['predictions'].shape}")
    
    # 测试生成模式
    outputs = model(x, mode='generate')
    print(f"Generated shape: {outputs['generated'].shape}")
    
    # 测试判别模式
    outputs = model(x, mode='discriminate')
    print(f"Discriminator output shape: {outputs['discriminator_output'].shape}")
    
    # 测试对抗模式
    outputs = model(x, mode='adversarial')
    print(f"Adversarial mode - Predictions: {outputs['predictions'].shape}, "
          f"Generated: {outputs['generated'].shape}, "
          f"Discriminator: {outputs['discriminator_output'].shape}")
    
    # 测试对抗攻击
    x_adv = model.generate_adversarial_samples(x, y, attack_type='fgsm')
    print(f"Adversarial samples shape: {x_adv.shape}")
    
    # 测试训练模式切换
    model.set_training_mode('pretrain')
    print("Switched to pretrain mode")
    
    model.set_training_mode('adversarial')
    print("Switched to adversarial mode")
    
    model.set_training_mode('finetune')
    print("Switched to finetune mode")
    
    print("Adversarial model test completed successfully!")


if __name__ == "__main__":
    test_adversarial_model()
