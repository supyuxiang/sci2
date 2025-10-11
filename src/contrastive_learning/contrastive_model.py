"""
对比学习增强预测器模型
实现基于对比学习的特征表示学习和预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.mlp import MLP
from src.models.lstm import LSTM
from src.models.gru import GRURegressor
from src.models.transformer import TransformerEncoderRegressor
from src.models.cnn1d import CNNRegressor1D


class ProjectionHead(nn.Module):
    """投影头：将特征映射到对比学习空间"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
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
            
        # 最后一层不使用激活函数
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class PredictionHead(nn.Module):
    """预测头：基于学习到的特征进行预测"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
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
            
        # 最后一层不使用激活函数
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.prediction = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prediction(x)


class ContrastiveEncoder(nn.Module):
    """对比学习编码器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 基础模型配置
        base_model = config.get('base_model', 'mlp')
        input_dim = config.get('input_dim', 3)
        hidden_dim = config.get('hidden_dim', 128)
        num_blocks = config.get('num_blocks', 8)
        dropout = config.get('dropout', 0.2)
        
        # 创建基础编码器
        if base_model == 'mlp':
            self.encoder = MLP(
                indim=input_dim,
                outdim=hidden_dim,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                dropout=dropout
            )
        elif base_model == 'lstm':
            self.encoder = LSTM(
                indim=input_dim,
                outdim=hidden_dim,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                dropout=dropout
            )
        elif base_model == 'gru':
            self.encoder = GRURegressor(
                indim=input_dim,
                outdim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=num_blocks,
                dropout=dropout
            )
        elif base_model == 'transformer':
            self.encoder = TransformerEncoderRegressor(
                indim=input_dim,
                outdim=hidden_dim,
                hidden_dim=hidden_dim,
                depth=num_blocks,
                num_heads=8,
                dropout=dropout
            )
        elif base_model == 'cnn1d':
            self.encoder = CNNRegressor1D(
                indim=input_dim,
                outdim=hidden_dim,
                channels=64,
                depth=4,
                kernel_size=3,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
            
        self.embedding_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码输入特征"""
        return self.encoder(x)


class ContrastiveEnhancedPredictor(nn.Module):
    """对比学习增强预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.name = config.get('name', 'contrastive_enhanced_predictor')
        
        # 模型参数
        input_dim = config.get('input_dim', 3)
        output_dim = config.get('output_dim', 4)
        embedding_dim = config.get('embedding_dim', 128)
        projection_dim = config.get('projection_dim', 64)
        temperature = config.get('temperature', 0.07)
        
        # 创建编码器
        self.encoder = ContrastiveEncoder(config)
        
        # 创建投影头
        projection_config = config.get('projection_head', {})
        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=projection_config.get('hidden_dim', 64),
            output_dim=projection_config.get('output_dim', 64),
            num_layers=projection_config.get('num_layers', 2),
            dropout=projection_config.get('dropout', 0.1)
        )
        
        # 创建预测头
        prediction_config = config.get('prediction_head', {})
        self.prediction_head = PredictionHead(
            input_dim=embedding_dim,
            hidden_dim=prediction_config.get('hidden_dim', 64),
            output_dim=output_dim,
            num_layers=prediction_config.get('num_layers', 2),
            dropout=prediction_config.get('dropout', 0.1)
        )
        
        # 温度参数
        self.temperature = temperature
        
        # 训练模式标志
        self.training_mode = 'pretrain'  # 'pretrain' or 'finetune'
        
    def forward(self, x: torch.Tensor, mode: str = 'predict') -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征
            mode: 模式 ('predict', 'contrastive', 'both')
            
        Returns:
            输出字典
        """
        # 编码特征
        features = self.encoder(x)
        
        outputs = {}
        
        if mode in ['predict', 'both']:
            # 预测模式
            predictions = self.prediction_head(features)
            outputs['predictions'] = predictions
            
        if mode in ['contrastive', 'both']:
            # 对比学习模式
            projections = self.projection_head(features)
            # L2归一化
            projections = F.normalize(projections, p=2, dim=1)
            outputs['projections'] = projections
            
        outputs['features'] = features
        
        return outputs
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码输入特征"""
        return self.encoder(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测模式"""
        outputs = self.forward(x, mode='predict')
        return outputs['predictions']
    
    def get_contrastive_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取对比学习特征"""
        outputs = self.forward(x, mode='contrastive')
        return outputs['projections']
    
    def set_training_mode(self, mode: str):
        """设置训练模式"""
        assert mode in ['pretrain', 'finetune'], f"Invalid training mode: {mode}"
        self.training_mode = mode
        
        if mode == 'pretrain':
            # 预训练模式：冻结预测头，训练编码器和投影头
            for param in self.prediction_head.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.projection_head.parameters():
                param.requires_grad = True
        else:
            # 微调模式：冻结投影头，训练编码器和预测头
            for param in self.projection_head.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.prediction_head.parameters():
                param.requires_grad = True


class MultiScaleContrastivePredictor(nn.Module):
    """多尺度对比学习预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.name = 'multi_scale_contrastive_predictor'
        
        # 多尺度配置
        scales = config.get('multi_scale', {}).get('scales', [1.0, 0.5, 2.0])
        self.scales = scales
        
        # 为每个尺度创建预测器
        self.predictors = nn.ModuleList([
            ContrastiveEnhancedPredictor(config) for _ in scales
        ])
        
        # 融合层
        embedding_dim = config.get('embedding_dim', 128)
        output_dim = config.get('output_dim', 4)
        self.fusion_layer = nn.Linear(embedding_dim * len(scales), output_dim)
        
    def forward(self, x: torch.Tensor, mode: str = 'predict') -> Dict[str, torch.Tensor]:
        """多尺度前向传播"""
        outputs = {}
        
        # 获取各尺度的特征
        all_features = []
        all_projections = []
        all_predictions = []
        
        for i, (scale, predictor) in enumerate(zip(self.scales, self.predictors)):
            # 应用尺度变换
            x_scaled = x * scale
            
            # 前向传播
            pred_outputs = predictor(x_scaled, mode='both')
            
            all_features.append(pred_outputs['features'])
            all_projections.append(pred_outputs['projections'])
            all_predictions.append(pred_outputs['predictions'])
            
        # 融合特征
        fused_features = torch.cat(all_features, dim=1)
        fused_predictions = self.fusion_layer(fused_features)
        
        outputs['features'] = fused_features
        outputs['predictions'] = fused_predictions
        outputs['multi_scale_features'] = all_features
        outputs['multi_scale_projections'] = all_projections
        outputs['multi_scale_predictions'] = all_predictions
        
        return outputs


class PhysicsAwareContrastivePredictor(nn.Module):
    """物理感知对比学习预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.name = 'physics_aware_contrastive_predictor'
        
        # 基础预测器
        self.base_predictor = ContrastiveEnhancedPredictor(config)
        
        # 物理约束层
        physics_config = config.get('physics_constraints', {})
        self.physics_enabled = physics_config.get('enabled', True)
        
        if self.physics_enabled:
            # 物理约束网络
            embedding_dim = config.get('embedding_dim', 128)
            self.physics_head = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)  # 物理约束损失
            )
            
    def forward(self, x: torch.Tensor, mode: str = 'predict') -> Dict[str, torch.Tensor]:
        """物理感知前向传播"""
        outputs = self.base_predictor(x, mode)
        
        if self.physics_enabled and mode in ['predict', 'both']:
            # 计算物理约束
            features = outputs['features']
            physics_constraint = self.physics_head(features)
            outputs['physics_constraint'] = physics_constraint
            
        return outputs
    
    def compute_physics_loss(self, predictions: torch.Tensor, 
                           targets: torch.Tensor) -> torch.Tensor:
        """计算物理约束损失"""
        if not self.physics_enabled:
            return torch.tensor(0.0, device=predictions.device)
            
        # 这里可以实现具体的物理约束损失
        # 例如：质量守恒、动量守恒、能量守恒等
        
        # 简单的物理约束：预测值应该满足某些物理关系
        # 这里使用一个示例约束：温度应该与速度的平方成正比
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


def create_contrastive_model(config: Dict[str, Any]) -> nn.Module:
    """创建对比学习模型"""
    model_type = config.get('model_type', 'contrastive_enhanced')
    
    if model_type == 'contrastive_enhanced':
        return ContrastiveEnhancedPredictor(config)
    elif model_type == 'multi_scale':
        return MultiScaleContrastivePredictor(config)
    elif model_type == 'physics_aware':
        return PhysicsAwareContrastivePredictor(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def test_contrastive_model():
    """测试对比学习模型"""
    # 创建测试配置
    config = {
        'name': 'test_contrastive_model',
        'base_model': 'mlp',
        'input_dim': 3,
        'output_dim': 4,
        'hidden_dim': 128,
        'num_blocks': 4,
        'dropout': 0.1,
        'embedding_dim': 128,
        'projection_dim': 64,
        'temperature': 0.07,
        'projection_head': {
            'hidden_dim': 64,
            'output_dim': 64,
            'num_layers': 2,
            'dropout': 0.1
        },
        'prediction_head': {
            'hidden_dim': 64,
            'output_dim': 4,
            'num_layers': 2,
            'dropout': 0.1
        }
    }
    
    # 创建模型
    model = ContrastiveEnhancedPredictor(config)
    
    # 创建测试数据
    batch_size = 32
    input_dim = 3
    x = torch.randn(batch_size, input_dim)
    
    # 测试预测模式
    outputs = model(x, mode='predict')
    print(f"Predictions shape: {outputs['predictions'].shape}")
    
    # 测试对比学习模式
    outputs = model(x, mode='contrastive')
    print(f"Projections shape: {outputs['projections'].shape}")
    
    # 测试两种模式
    outputs = model(x, mode='both')
    print(f"Both mode - Predictions: {outputs['predictions'].shape}, "
          f"Projections: {outputs['projections'].shape}")
    
    # 测试训练模式切换
    model.set_training_mode('pretrain')
    print("Switched to pretrain mode")
    
    model.set_training_mode('finetune')
    print("Switched to finetune mode")
    
    print("Contrastive model test completed successfully!")


if __name__ == "__main__":
    test_contrastive_model()
