"""
对抗学习损失函数模块
实现GAN损失、对抗损失、物理约束损失等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List


class GANLoss(nn.Module):
    """GAN损失函数"""
    
    def __init__(self, loss_type: str = 'bce', label_smoothing: float = 0.0):
        super().__init__()
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        
        if loss_type == 'bce':
            self.loss_fn = nn.BCELoss()
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'wgan':
            self.loss_fn = None  # WGAN使用特殊损失
        else:
            raise ValueError(f"Unsupported GAN loss type: {loss_type}")
            
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算GAN损失
        
        Args:
            predictions: 预测值 [batch_size, 1]
            targets: 目标值 [batch_size, 1]
            
        Returns:
            GAN损失值
        """
        if self.loss_type == 'wgan':
            # WGAN损失：判别器损失为真实样本和生成样本的差值
            return -torch.mean(predictions * targets)
        else:
            # 应用标签平滑
            if self.label_smoothing > 0:
                targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            return self.loss_fn(predictions, targets)


class AdversarialLoss(nn.Module):
    """对抗损失函数"""
    
    def __init__(self, loss_type: str = 'mse', epsilon: float = 0.1):
        super().__init__()
        self.loss_type = loss_type
        self.epsilon = epsilon
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported adversarial loss type: {loss_type}")
            
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                adversarial_predictions: torch.Tensor) -> torch.Tensor:
        """
        计算对抗损失
        
        Args:
            predictions: 原始预测值
            targets: 目标值
            adversarial_predictions: 对抗样本预测值
            
        Returns:
            对抗损失值
        """
        # 原始预测损失
        original_loss = self.loss_fn(predictions, targets)
        
        # 对抗预测损失
        adversarial_loss = self.loss_fn(adversarial_predictions, targets)
        
        # 对抗损失：鼓励模型对对抗样本和原始样本产生相似的预测
        consistency_loss = self.loss_fn(predictions, adversarial_predictions)
        
        # 总对抗损失
        total_loss = original_loss + self.epsilon * consistency_loss
        
        return total_loss


class PhysicsLoss(nn.Module):
    """物理约束损失函数"""
    
    def __init__(self, weight: float = 0.2, constraint_types: List[str] = None):
        super().__init__()
        self.weight = weight
        self.constraint_types = constraint_types or ["conservation", "boundary"]
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                x: torch.Tensor) -> torch.Tensor:
        """
        计算物理约束损失
        
        Args:
            predictions: 预测值 [batch_size, output_dim]
            targets: 目标值 [batch_size, output_dim]
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            物理约束损失值
        """
        physics_loss = torch.tensor(0.0, device=predictions.device)
        
        if "conservation" in self.constraint_types:
            # 守恒定律约束
            conservation_loss = self._conservation_constraint(predictions, x)
            physics_loss += conservation_loss
            
        if "boundary" in self.constraint_types:
            # 边界条件约束
            boundary_loss = self._boundary_constraint(predictions, x)
            physics_loss += boundary_loss
            
        return self.weight * physics_loss
    
    def _conservation_constraint(self, predictions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """守恒定律约束"""
        if predictions.shape[1] >= 4:  # 假设有温度、速度、压力等
            temp_pred = predictions[:, 0:1]  # 温度
            vel_pred = predictions[:, 1:3]   # 速度分量
            pressure_pred = predictions[:, 3:4]  # 压力
            
            # 计算速度大小
            vel_magnitude = torch.norm(vel_pred, dim=1, keepdim=True)
            
            # 能量守恒：温度与速度平方的关系
            energy_conservation = F.mse_loss(temp_pred, vel_magnitude ** 2)
            
            # 动量守恒：速度梯度约束
            if x.shape[1] >= 3:  # 假设有空间坐标
                # 简化的动量守恒约束
                momentum_conservation = torch.mean(torch.abs(vel_pred))
            else:
                momentum_conservation = torch.tensor(0.0, device=predictions.device)
                
            return energy_conservation + 0.1 * momentum_conservation
        else:
            return torch.tensor(0.0, device=predictions.device)
    
    def _boundary_constraint(self, predictions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """边界条件约束"""
        if x.shape[1] >= 3:  # 假设有空间坐标
            # 边界条件：在边界处某些物理量应该满足特定条件
            # 这里实现一个简化的边界约束
            boundary_mask = (torch.abs(x[:, 0]) > 0.9) | (torch.abs(x[:, 1]) > 0.9) | (torch.abs(x[:, 2]) > 0.9)
            
            if boundary_mask.any():
                boundary_predictions = predictions[boundary_mask]
                # 边界处温度应该为常数
                boundary_temp = boundary_predictions[:, 0:1]
                boundary_loss = torch.var(boundary_temp)
                return boundary_loss
            else:
                return torch.tensor(0.0, device=predictions.device)
        else:
            return torch.tensor(0.0, device=predictions.device)


class GradientPenaltyLoss(nn.Module):
    """梯度惩罚损失函数（用于WGAN-GP）"""
    
    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight
        
    def forward(self, discriminator: nn.Module, real_data: torch.Tensor, 
                fake_data: torch.Tensor) -> torch.Tensor:
        """
        计算梯度惩罚损失
        
        Args:
            discriminator: 判别器模型
            real_data: 真实数据
            fake_data: 生成数据
            
        Returns:
            梯度惩罚损失值
        """
        batch_size = real_data.shape[0]
        device = real_data.device
        
        # 生成随机插值
        alpha = torch.rand(batch_size, 1, device=device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # 计算判别器输出
        d_interpolated = discriminator(interpolated)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 计算梯度惩罚
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return self.weight * gradient_penalty


class FeatureMatchingLoss(nn.Module):
    """特征匹配损失函数"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.loss_fn = nn.L1Loss()
        
    def forward(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> torch.Tensor:
        """
        计算特征匹配损失
        
        Args:
            real_features: 真实数据的特征
            fake_features: 生成数据的特征
            
        Returns:
            特征匹配损失值
        """
        return self.weight * self.loss_fn(fake_features, real_features)


class PerceptualLoss(nn.Module):
    """感知损失函数"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.loss_fn = nn.MSELoss()
        
    def forward(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> torch.Tensor:
        """
        计算感知损失
        
        Args:
            real_features: 真实数据的特征
            fake_features: 生成数据的特征
            
        Returns:
            感知损失值
        """
        return self.weight * self.loss_fn(fake_features, real_features)


class CombinedAdversarialLoss(nn.Module):
    """组合对抗学习损失函数"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 损失权重
        self.generator_weight = config.get('generator_weight', 0.3)
        self.discriminator_weight = config.get('discriminator_weight', 0.3)
        self.predictor_weight = config.get('predictor_weight', 0.4)
        self.physics_weight = config.get('physics_weight', 0.2)
        
        # 创建各种损失函数
        self._build_loss_functions()
        
    def _build_loss_functions(self):
        """构建损失函数"""
        # GAN损失
        gan_config = self.config.get('AdversarialLoss', {}).get('gan_loss', {})
        self.gan_loss = GANLoss(
            loss_type=gan_config.get('type', 'bce'),
            label_smoothing=gan_config.get('label_smoothing', 0.1)
        )
        
        # 对抗损失
        adv_config = self.config.get('AdversarialLoss', {}).get('adversarial_loss', {})
        self.adversarial_loss = AdversarialLoss(
            loss_type='mse',
            epsilon=adv_config.get('epsilon', 0.1)
        )
        
        # 物理约束损失
        physics_config = self.config.get('AdversarialLoss', {}).get('physics_loss', {})
        if physics_config.get('enabled', True):
            self.physics_loss = PhysicsLoss(
                weight=physics_config.get('weight', 0.2),
                constraint_types=physics_config.get('constraint_types', ["conservation", "boundary"])
            )
        else:
            self.physics_loss = None
            
        # 梯度惩罚损失
        reg_config = self.config.get('AdversarialLoss', {}).get('regularization', {})
        self.gradient_penalty = GradientPenaltyLoss(
            weight=reg_config.get('gradient_penalty_weight', 10.0)
        )
        
        # 特征匹配损失
        self.feature_matching_loss = FeatureMatchingLoss(weight=1.0)
        
        # 预测损失
        self.prediction_loss = nn.MSELoss()
        
    def compute_generator_loss(self, discriminator_output_fake: torch.Tensor,
                              real_features: torch.Tensor, fake_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算生成器损失"""
        losses = {}
        
        # GAN损失
        fake_labels = torch.ones_like(discriminator_output_fake)
        gan_loss = self.gan_loss(discriminator_output_fake, fake_labels)
        losses['gan_loss'] = gan_loss
        
        # 特征匹配损失
        feature_matching = self.feature_matching_loss(real_features, fake_features)
        losses['feature_matching_loss'] = feature_matching
        
        # 总生成器损失
        total_loss = gan_loss + feature_matching
        losses['total_generator_loss'] = total_loss
        
        return losses
    
    def compute_discriminator_loss(self, discriminator_output_real: torch.Tensor,
                                  discriminator_output_fake: torch.Tensor,
                                  real_data: torch.Tensor, fake_data: torch.Tensor,
                                  discriminator: nn.Module) -> Dict[str, torch.Tensor]:
        """计算判别器损失"""
        losses = {}
        
        # 真实样本损失
        real_labels = torch.ones_like(discriminator_output_real)
        real_loss = self.gan_loss(discriminator_output_real, real_labels)
        losses['real_loss'] = real_loss
        
        # 生成样本损失
        fake_labels = torch.zeros_like(discriminator_output_fake)
        fake_loss = self.gan_loss(discriminator_output_fake, fake_labels)
        losses['fake_loss'] = fake_loss
        
        # 梯度惩罚损失
        gradient_penalty = self.gradient_penalty(discriminator, real_data, fake_data)
        losses['gradient_penalty'] = gradient_penalty
        
        # 总判别器损失
        total_loss = real_loss + fake_loss + gradient_penalty
        losses['total_discriminator_loss'] = total_loss
        
        return losses
    
    def compute_predictor_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                              adversarial_predictions: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算预测器损失"""
        losses = {}
        
        # 预测损失
        prediction_loss = self.prediction_loss(predictions, targets)
        losses['prediction_loss'] = prediction_loss
        
        # 对抗损失
        adversarial_loss = self.adversarial_loss(predictions, targets, adversarial_predictions)
        losses['adversarial_loss'] = adversarial_loss
        
        # 物理约束损失
        if self.physics_loss is not None:
            physics_loss = self.physics_loss(predictions, targets, x)
            losses['physics_loss'] = physics_loss
        else:
            losses['physics_loss'] = torch.tensor(0.0, device=predictions.device)
            
        # 总预测器损失
        total_loss = (prediction_loss + 
                     self.config.get('AdversarialLoss', {}).get('adversarial_loss', {}).get('epsilon', 0.1) * adversarial_loss +
                     self.physics_weight * losses['physics_loss'])
        losses['total_predictor_loss'] = total_loss
        
        return losses
    
    def compute_total_loss(self, generator_losses: Dict[str, torch.Tensor],
                          discriminator_losses: Dict[str, torch.Tensor],
                          predictor_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算总损失"""
        total_losses = {}
        
        # 组合所有损失
        total_loss = (self.generator_weight * generator_losses['total_generator_loss'] +
                     self.discriminator_weight * discriminator_losses['total_discriminator_loss'] +
                     self.predictor_weight * predictor_losses['total_predictor_loss'])
        
        total_losses['total_loss'] = total_loss
        total_losses.update(generator_losses)
        total_losses.update(discriminator_losses)
        total_losses.update(predictor_losses)
        
        return total_losses


class AdversarialLossFactory:
    """对抗学习损失函数工厂"""
    
    @staticmethod
    def create_loss(loss_type: str, config: Dict[str, Any]) -> nn.Module:
        """
        创建对抗学习损失函数
        
        Args:
            loss_type: 损失函数类型
            config: 配置参数
            
        Returns:
            损失函数实例
        """
        if loss_type == "gan":
            return GANLoss(
                loss_type=config.get('type', 'bce'),
                label_smoothing=config.get('label_smoothing', 0.0)
            )
        elif loss_type == "adversarial":
            return AdversarialLoss(
                loss_type=config.get('type', 'mse'),
                epsilon=config.get('epsilon', 0.1)
            )
        elif loss_type == "physics":
            return PhysicsLoss(
                weight=config.get('weight', 0.2),
                constraint_types=config.get('constraint_types', ["conservation", "boundary"])
            )
        elif loss_type == "gradient_penalty":
            return GradientPenaltyLoss(
                weight=config.get('weight', 10.0)
            )
        elif loss_type == "feature_matching":
            return FeatureMatchingLoss(
                weight=config.get('weight', 1.0)
            )
        elif loss_type == "perceptual":
            return PerceptualLoss(
                weight=config.get('weight', 1.0)
            )
        elif loss_type == "combined":
            return CombinedAdversarialLoss(config)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")


def test_adversarial_loss():
    """测试对抗学习损失函数"""
    batch_size = 32
    input_dim = 3
    output_dim = 4
    
    # 创建测试数据
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, output_dim)
    y_adv = torch.randn(batch_size, output_dim)
    discriminator_output_real = torch.randn(batch_size, 1)
    discriminator_output_fake = torch.randn(batch_size, 1)
    
    # 测试GAN损失
    gan_loss = GANLoss(loss_type='bce', label_smoothing=0.1)
    loss1 = gan_loss(discriminator_output_real, torch.ones_like(discriminator_output_real))
    print(f"GAN Loss: {loss1.item():.4f}")
    
    # 测试对抗损失
    adversarial_loss = AdversarialLoss(loss_type='mse', epsilon=0.1)
    loss2 = adversarial_loss(y, y, y_adv)
    print(f"Adversarial Loss: {loss2.item():.4f}")
    
    # 测试物理约束损失
    physics_loss = PhysicsLoss(weight=0.2, constraint_types=["conservation", "boundary"])
    loss3 = physics_loss(y, y, x)
    print(f"Physics Loss: {loss3.item():.4f}")
    
    # 测试梯度惩罚损失
    class DummyDiscriminator(nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=1, keepdim=True)
    
    discriminator = DummyDiscriminator()
    gradient_penalty = GradientPenaltyLoss(weight=10.0)
    loss4 = gradient_penalty(discriminator, x, x)
    print(f"Gradient Penalty Loss: {loss4.item():.4f}")
    
    # 测试特征匹配损失
    feature_matching_loss = FeatureMatchingLoss(weight=1.0)
    loss5 = feature_matching_loss(y, y_adv)
    print(f"Feature Matching Loss: {loss5.item():.4f}")
    
    print("Adversarial loss test completed successfully!")


if __name__ == "__main__":
    test_adversarial_loss()
