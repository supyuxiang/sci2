import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCARNet(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.2):
        super().__init__()
        
        # 增强的双流输入处理（多层感知机）
        self.input_proj1 = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.input_proj2 = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 多层级联交叉注意力模块
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(2)
        ])
        self.cross_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim*2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(2)
        ])
        
        # 深度残差处理块（带跳跃连接）
        self.res_blocks = nn.ModuleList()
        for _ in range(4):
            block = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim*2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ),
                nn.Sequential(
                    nn.Linear(hidden_dim*2, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                )
            ])
            self.res_blocks.append(block)
        
        # 特征增强模块
        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim*2),
            nn.Sigmoid(),
            nn.Linear(hidden_dim*2, 3),
            nn.Softmax(dim=-1)
        )
        
        # 多尺度输出层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )

    def forward(self, x):
        # 增强输入投影
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        h1 = self.input_proj1(x1)  # [B, D]
        h2 = self.input_proj2(x2)  # [B, D]
        
        # 多层级联交叉注意力
        attn_outs = []
        query = h1.unsqueeze(1)
        key_value = h2.unsqueeze(1)
        
        for attn, ffn in zip(self.cross_attn_layers, self.cross_ffn):
            attn_out, _ = attn(query, key_value, key_value)
            attn_out = ffn(attn_out)
            attn_outs.append(attn_out)
            query = attn_out  # 使用上一层的输出作为下一层的查询
        
        # 门控特征融合
        h1_exp = h1.unsqueeze(1)
        h2_exp = h2.unsqueeze(1)
        combined = torch.cat([h1_exp, h2_exp, attn_outs[-1]], dim=-1)
        gate_weights = self.gate(combined)  # [B, 1, 3]
        
        # 加权特征融合
        features = torch.stack([h1_exp, h2_exp, attn_outs[-1]], dim=-1)
        fused = torch.sum(features * gate_weights.unsqueeze(-2), dim=-1).squeeze(1)
        
        # 深度残差处理
        for block in self.res_blocks:
            residual = fused
            # 瓶颈结构
            fused = block[0](fused)  # 扩展维度
            fused = block[1](fused)  # 压缩维度
            fused = fused + residual
        
        # 特征增强
        enhanced = self.feature_enhancer(fused) + fused
        
        # 多尺度输出
        return self.output_proj(enhanced)

if __name__ == '__main__':
    # 测试代码
    model = EnhancedCARNet()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    test_input = torch.randn(32, 2)
    output = model(test_input)
    print(f"输入形状: {test_input.shape} -> 输出形状: {output.shape}")