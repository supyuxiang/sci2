"""
对比学习损失函数模块
实现InfoNCE、SimCLR、MoCo等对比学习损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List


class InfoNCELoss(nn.Module):
    """InfoNCE对比学习损失函数"""
    
    def __init__(self, temperature: float = 0.07, negative_samples: int = 64):
        super().__init__()
        self.temperature = temperature
        self.negative_samples = negative_samples
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negatives: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算InfoNCE损失
        
        Args:
            anchor: 锚点特征 [batch_size, feature_dim]
            positive: 正样本特征 [batch_size, feature_dim]
            negatives: 负样本特征 [batch_size, negative_samples, feature_dim]
            
        Returns:
            InfoNCE损失值
        """
        batch_size = anchor.shape[0]
        
        # 计算锚点与正样本的相似度
        pos_sim = F.cosine_similarity(anchor, positive, dim=1) / self.temperature
        
        # 如果没有提供负样本，使用批次中的其他样本作为负样本
        if negatives is None:
            # 使用批次中的其他样本作为负样本
            all_features = torch.cat([anchor, positive], dim=0)
            neg_sim = torch.matmul(anchor, all_features.T) / self.temperature
            
            # 移除对角线元素（自己与自己的相似度）
            mask = torch.eye(batch_size, device=anchor.device).bool()
            neg_sim = neg_sim.masked_fill(mask, float('-inf'))
        else:
            # 使用提供的负样本
            neg_sim = torch.matmul(anchor, negatives.view(-1, negatives.shape[-1]).T) / self.temperature
        
        # 计算logits
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # 标签：第一个位置是正样本
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss


class SimCLRLoss(nn.Module):
    """SimCLR对比学习损失函数"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算SimCLR损失
        
        Args:
            features: 特征向量 [2*batch_size, feature_dim]
                     前batch_size个是原始样本，后batch_size个是对应的增强样本
                     
        Returns:
            SimCLR损失值
        """
        batch_size = features.shape[0] // 2
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建掩码：正样本对的对角线
        mask = torch.zeros(2 * batch_size, 2 * batch_size, device=features.device)
        mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = 1
        mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = 1
        
        # 计算正样本相似度
        pos_sim = sim_matrix * mask
        pos_sim = pos_sim.sum(dim=1)
        
        # 计算负样本相似度
        neg_sim = sim_matrix * (1 - mask)
        neg_sim = neg_sim.sum(dim=1)
        
        # 计算损失
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
        
        return loss.mean()


class MoCoLoss(nn.Module):
    """MoCo对比学习损失函数"""
    
    def __init__(self, temperature: float = 0.07, queue_size: int = 65536):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        
        # 初始化队列
        self.register_buffer("queue", torch.randn(queue_size, 128))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """更新队列"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # 替换队列中的样本
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.queue_size
        else:
            # 队列已满，替换部分样本
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]
            ptr = batch_size - remaining
            
        self.queue_ptr[0] = ptr
        
    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        计算MoCo损失
        
        Args:
            query: 查询特征 [batch_size, feature_dim]
            key: 键特征 [batch_size, feature_dim]
            
        Returns:
            MoCo损失值
        """
        batch_size = query.shape[0]
        
        # 计算查询与键的相似度
        pos_sim = torch.sum(query * key, dim=1, keepdim=True) / self.temperature
        
        # 计算查询与队列中负样本的相似度
        neg_sim = torch.matmul(query, self.queue.T) / self.temperature
        
        # 组合正样本和负样本
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        # 标签：第一个位置是正样本
        labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)
        
        # 计算损失
        loss = F.cross_entropy(logits, labels)
        
        # 更新队列
        self._dequeue_and_enqueue(key)
        
        return loss


class PhysicsContrastiveLoss(nn.Module):
    """物理约束对比学习损失函数"""
    
    def __init__(self, temperature: float = 0.07, weight: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.base_loss = InfoNCELoss(temperature)
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                anchor_physics: torch.Tensor, positive_physics: torch.Tensor) -> torch.Tensor:
        """
        计算物理约束对比学习损失
        
        Args:
            anchor: 锚点特征
            positive: 正样本特征
            anchor_physics: 锚点物理量
            positive_physics: 正样本物理量
            
        Returns:
            物理约束对比学习损失
        """
        # 基础对比学习损失
        contrastive_loss = self.base_loss(anchor, positive)
        
        # 物理约束损失
        physics_loss = F.mse_loss(anchor_physics, positive_physics)
        
        # 组合损失
        total_loss = contrastive_loss + self.weight * physics_loss
        
        return total_loss


class MultiScaleContrastiveLoss(nn.Module):
    """多尺度对比学习损失函数"""
    
    def __init__(self, scales: List[float], weights: List[float], temperature: float = 0.07):
        super().__init__()
        self.scales = scales
        self.weights = weights
        self.temperature = temperature
        
        # 为每个尺度创建损失函数
        self.losses = nn.ModuleList([
            InfoNCELoss(temperature) for _ in scales
        ])
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negatives: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算多尺度对比学习损失
        
        Args:
            anchor: 锚点特征
            positive: 正样本特征
            negatives: 负样本特征
            
        Returns:
            多尺度对比学习损失
        """
        total_loss = 0.0
        
        for i, (scale, weight, loss_fn) in enumerate(zip(self.scales, self.weights, self.losses)):
            # 应用尺度变换
            anchor_scaled = anchor * scale
            positive_scaled = positive * scale
            
            # 计算该尺度的损失
            scale_loss = loss_fn(anchor_scaled, positive_scaled, negatives)
            
            # 加权累加
            total_loss += weight * scale_loss
            
        return total_loss


class ContrastiveLossFactory:
    """对比学习损失函数工厂"""
    
    @staticmethod
    def create_loss(loss_type: str, config: Dict[str, Any]) -> nn.Module:
        """
        创建对比学习损失函数
        
        Args:
            loss_type: 损失函数类型
            config: 配置参数
            
        Returns:
            损失函数实例
        """
        if loss_type == "infonce":
            return InfoNCELoss(
                temperature=config.get('temperature', 0.07),
                negative_samples=config.get('negative_samples', 64)
            )
        elif loss_type == "simclr":
            return SimCLRLoss(
                temperature=config.get('temperature', 0.07)
            )
        elif loss_type == "moco":
            return MoCoLoss(
                temperature=config.get('temperature', 0.07),
                queue_size=config.get('queue_size', 65536)
            )
        elif loss_type == "physics_contrastive":
            return PhysicsContrastiveLoss(
                temperature=config.get('temperature', 0.07),
                weight=config.get('weight', 0.3)
            )
        elif loss_type == "multi_scale":
            return MultiScaleContrastiveLoss(
                scales=config.get('scales', [1.0, 0.5, 2.0]),
                weights=config.get('weights', [0.5, 0.3, 0.2]),
                temperature=config.get('temperature', 0.07)
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")


class CombinedContrastiveLoss(nn.Module):
    """组合对比学习损失函数"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 对比学习损失权重
        self.contrastive_weight = config.get('contrastive_weight', 0.5)
        self.prediction_weight = config.get('prediction_weight', 0.5)
        self.physics_weight = config.get('physics_weight', 0.3)
        
        # 创建对比学习损失函数
        contrastive_config = config.get('ContrastiveLoss', {})
        self.contrastive_loss = ContrastiveLossFactory.create_loss(
            contrastive_config.get('type', 'infonce'),
            contrastive_config
        )
        
        # 预测损失函数
        self.prediction_loss = nn.MSELoss()
        
        # 物理约束损失函数
        if config.get('is_pinn', False):
            from src.utils.loss_function import PhysicsLoss
            self.physics_loss = PhysicsLoss()
        else:
            self.physics_loss = None
            
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                predictions: torch.Tensor, targets: torch.Tensor,
                physics_predictions: Optional[torch.Tensor] = None,
                physics_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算组合损失
        
        Args:
            anchor: 锚点特征
            positive: 正样本特征
            predictions: 预测结果
            targets: 真实标签
            physics_predictions: 物理预测结果
            physics_targets: 物理真实标签
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 对比学习损失
        contrastive_loss = self.contrastive_loss(anchor, positive)
        losses['contrastive_loss'] = contrastive_loss
        
        # 预测损失
        prediction_loss = self.prediction_loss(predictions, targets)
        losses['prediction_loss'] = prediction_loss
        
        # 物理约束损失
        if self.physics_loss is not None and physics_predictions is not None:
            physics_loss = self.physics_loss.compute_physics_loss(physics_predictions, physics_targets)
            losses['physics_loss'] = physics_loss
        else:
            losses['physics_loss'] = torch.tensor(0.0, device=anchor.device)
            
        # 总损失
        total_loss = (self.contrastive_weight * contrastive_loss + 
                     self.prediction_weight * prediction_loss + 
                     self.physics_weight * losses['physics_loss'])
        losses['total_loss'] = total_loss
        
        return losses


def test_contrastive_loss():
    """测试对比学习损失函数"""
    batch_size = 32
    feature_dim = 128
    
    # 创建测试数据
    anchor = torch.randn(batch_size, feature_dim)
    positive = torch.randn(batch_size, feature_dim)
    negative = torch.randn(batch_size, 64, feature_dim)
    
    # 测试InfoNCE损失
    infonce_loss = InfoNCELoss(temperature=0.07)
    loss1 = infonce_loss(anchor, positive, negative)
    print(f"InfoNCE Loss: {loss1.item():.4f}")
    
    # 测试SimCLR损失
    features = torch.cat([anchor, positive], dim=0)
    simclr_loss = SimCLRLoss(temperature=0.07)
    loss2 = simclr_loss(features)
    print(f"SimCLR Loss: {loss2.item():.4f}")
    
    # 测试MoCo损失
    moco_loss = MoCoLoss(temperature=0.07)
    loss3 = moco_loss(anchor, positive)
    print(f"MoCo Loss: {loss3.item():.4f}")
    
    # 测试多尺度损失
    multi_scale_loss = MultiScaleContrastiveLoss(
        scales=[1.0, 0.5, 2.0],
        weights=[0.5, 0.3, 0.2]
    )
    loss4 = multi_scale_loss(anchor, positive)
    print(f"Multi-Scale Loss: {loss4.item():.4f}")
    
    print("Contrastive loss test completed successfully!")


if __name__ == "__main__":
    test_contrastive_loss()
