#!/usr/bin/env python3
"""
增强版CARNet模型 - 包含多种先进架构组件,结合了Transformer, 残差块, 注意力机制, 门控融合机制, 特征金字塔网络, 多尺度卷积, 位置编码等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [L, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [D/2]
        pe[:, 0::2] = torch.sin(position * div_term) # [L, D/2]
        pe[:, 1::2] = torch.cos(position * div_term) # [L, D/2]
        pe = pe.unsqueeze(0).transpose(0, 1) # [1, L, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :] # [B, L, D]

class MultiScaleConv1d(nn.Module):
    """多尺度卷积模块"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 9]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // len(kernel_sizes), k, padding=k//2)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [B, L, C] -> [B, C, L]
        x = x.transpose(1, 2)
        conv_outs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        # [B, C, L] -> [B, L, C]
        return out.transpose(1, 2)

class TransformerBlock(nn.Module):
    """增强的Transformer块"""
    def __init__(self, hidden_dim, num_heads, dropout, ffn_ratio=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 增强的FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 门控残差连接
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 自注意力
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        gate = self.gate(x)
        x = self.norm2(x + gate * ffn_out)
        
        return x

class ResidualBlock(nn.Module):
    """增强的残差块"""
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.compress = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.compress(out)
        gate = self.gate(x)
        return out + gate * residual

class FeaturePyramid(nn.Module):
    """特征金字塔网络"""
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.pyramid_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 3),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        ])
        
    def forward(self, x):
        features = []
        for layer in self.pyramid_layers:
            x = layer(x)
            features.append(x)
        return torch.stack(features, dim=1)  # [B, 3, D]

class AttentionPooling(nn.Module):
    """注意力池化模块"""
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: [B, L, D]
        # 使用可学习的查询向量
        batch_size = x.size(0)
        query = torch.randn(batch_size, 1, x.size(-1)).to(x.device)
        query = query / query.norm(dim=-1, keepdim=True)
        
        attn_out, _ = self.attention(query, x, x)
        return self.norm(attn_out.squeeze(1))  # [B, D]

class EnhancedCARNet_v2(nn.Module):
    def __init__(self, hidden_dim=2048, num_heads=32, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # 增强的多尺度输入处理
        self.input_proj1 = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.input_proj2 = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 多尺度卷积特征提取
        self.multiscale_conv = MultiScaleConv1d(hidden_dim, hidden_dim)
        
        # 增强的Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout, ffn_ratio=4)
            for _ in range(4)  # 增加层数
        ])
        
        # 多层级联交叉注意力模块
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(4)  # 增加层数
        ])
        
        # 增强的交叉注意力FFN
        self.cross_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 3),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(4)
        ])
        
        # 深度残差处理块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(8)  # 增加层数
        ])
        
        # 特征金字塔网络
        self.feature_pyramid = FeaturePyramid(hidden_dim, dropout)
        
        # 增强的门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 3),  # 5 = 2输入 + 3金字塔特征
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 5),
            nn.Softmax(dim=-1)
        )
        
        # 注意力池化
        self.attention_pool = AttentionPooling(hidden_dim, num_heads, dropout)
        
        # 多尺度输出层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1)
        )
        
        # 辅助输出（用于深度监督）
        '''self.aux_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )'''
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # 输入处理
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        h1 = self.input_proj1(x1)  # [B, D]
        h2 = self.input_proj2(x2)  # [B, D]
        
        # 添加位置编码
        h1 = h1.unsqueeze(1)  # [B, 1, D]
        h2 = h2.unsqueeze(1)  # [B, 1, D]
        h1 = self.pos_encoding(h1)
        h2 = self.pos_encoding(h2)
        
        # 多尺度卷积特征提取
        conv_features = self.multiscale_conv(h1)  # [B, 1, D]
        
        # Transformer自注意力处理
        transformer_out = h1
        for transformer in self.transformer_blocks:
            transformer_out = transformer(transformer_out)
        
        # 多层级联交叉注意力
        attn_outs = []
        query = h1
        key_value = h2
        
        for attn, ffn in zip(self.cross_attn_layers, self.cross_ffn):
            attn_out, _ = attn(query, key_value, key_value)
            attn_out = ffn(attn_out)
            attn_outs.append(attn_out)
            query = attn_out  # 使用上一层的输出作为下一层的查询
        
        # 特征融合
        h1_flat = h1.squeeze(1)  # [B, D]
        h2_flat = h2.squeeze(1)  # [B, D]
        transformer_flat = transformer_out.squeeze(1)  # [B, D]
        conv_flat = conv_features.squeeze(1)  # [B, D]
        
        # 深度残差处理
        residual_out = transformer_flat
        for res_block in self.res_blocks:
            residual_out = res_block(residual_out)
        
        # 特征金字塔处理
        pyramid_features = self.feature_pyramid(residual_out)  # [B, 3, D]
        
        # 门控融合
        combined = torch.cat([h1_flat, h2_flat, transformer_flat, conv_flat, residual_out], dim=-1)
        gate_weights = self.gate(combined)  # [B, 5]
        
        # 加权特征融合
        features = torch.stack([h1_flat, h2_flat, transformer_flat, conv_flat, residual_out], dim=1)  # [B, 5, D]
        fused = torch.sum(features * gate_weights.unsqueeze(-1), dim=1)  # [B, D]
        
        # 注意力池化
        fused_expanded = fused.unsqueeze(1)  # [B, 1, D]
        pooled = self.attention_pool(fused_expanded)  # [B, D]
        
        # 多尺度输出
        main_output = self.output_proj(pooled)
        #aux_output = self.aux_output(pooled)
        
        return main_output #,aux_output

def get_model_info(model):
    """获取模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
        'layers': len(list(model.modules())),
        'modules': [name for name, _ in model.named_modules()]
    }
    return info

if __name__ == '__main__':
    # 测试代码
    print("=== 增强版CARNet模型测试 ===")
    
    # 创建模型
    model = EnhancedCARNet_v2(hidden_dim=128, num_heads=8, dropout=0.1)
    
    # 获取模型信息
    model_info = get_model_info(model)
    print(f"总参数量: {model_info['total_parameters']:,}")
    print(f"可训练参数量: {model_info['trainable_parameters']:,}")
    print(f"模型大小: {model_info['model_size_mb']:.2f} MB")
    print(f"层数: {model_info['layers']}")
    
    # 测试前向传播
    test_input = torch.randn(32, 2)
    main_output, aux_output = model(test_input)
    
    print(f"\n输入形状: {test_input.shape}")
    print(f"主输出形状: {main_output.shape}")
    print(f"辅助输出形状: {aux_output.shape}")
    
    # 测试梯度
    loss = main_output.mean() + aux_output.mean()
    loss.backward()
    
    print("\n梯度测试通过!")
    print("模型创建成功!")
