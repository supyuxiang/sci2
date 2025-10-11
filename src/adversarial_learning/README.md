# 对抗学习增强预测器模块

本模块实现了基于对抗学习的增强预测器，通过生成对抗网络（GAN）和对抗训练技术提升模型的鲁棒性和性能。

## 模块概述

对抗学习增强预测器采用三阶段训练策略：
1. **预训练阶段**: 训练预测器学习基础特征表示
2. **对抗训练阶段**: 生成器、判别器和预测器协同训练
3. **微调阶段**: 优化预测器性能

## 核心组件

### 1. 模型架构

- **生成器 (Generator)**: 生成对抗样本或增强数据
- **判别器 (Discriminator)**: 区分真实样本和生成样本
- **预测器 (Predictor)**: 基于学习到的特征进行预测

### 2. 对抗攻击

支持多种对抗攻击方法：
- **FGSM**: 快速梯度符号方法
- **PGD**: 投影梯度下降攻击
- **CW**: Carlini-Wagner攻击

### 3. 损失函数

- **GAN损失**: 生成器和判别器的对抗损失
- **对抗损失**: 提升模型对对抗样本的鲁棒性
- **物理约束损失**: 集成物理信息神经网络约束
- **梯度惩罚损失**: 稳定GAN训练

### 4. 数据增强

- **噪声增强**: 添加高斯噪声或均匀噪声
- **空间变换**: 随机缩放和旋转
- **Dropout增强**: 随机丢弃特征
- **物理约束增强**: 基于物理定律的数据增强
- **对抗增强**: 生成对抗样本进行训练

## 使用方法

### 1. 基本使用

```bash
# 运行对抗学习训练
python main.py --config-path=src/adversarial_learning --config-name=config
```

### 2. 配置参数

主要配置参数说明：

```yaml
# 对抗学习模型配置
AdversarialModel:
  name: "adversarial_enhanced_predictor"
  base_model: "mlp"  # 基础模型类型
  input_dim: 3
  output_dim: 4
  
  # 生成器配置
  generator:
    hidden_dim: 128
    num_layers: 6
    noise_dim: 32
    
  # 判别器配置
  discriminator:
    hidden_dim: 128
    num_layers: 4
    
  # 预测器配置
  predictor:
    hidden_dim: 128
    num_layers: 6

# 对抗学习训练配置
AdversarialTrain:
  pretrain_epochs: 200       # 预训练轮数
  adversarial_epochs: 600    # 对抗训练轮数
  finetune_epochs: 200       # 微调轮数
  
  # GAN训练参数
  gan_training:
    d_steps: 1               # 判别器训练步数
    g_steps: 1               # 生成器训练步数
```

### 3. 对抗攻击配置

```yaml
# 对抗攻击配置
AdversarialAttack:
  enabled: True
  attack_types: ["fgsm", "pgd", "cw"]
  epsilon_range: [0.01, 0.1, 0.2]
  
  # FGSM攻击
  fgsm:
    epsilon: 0.1
    targeted: False
    
  # PGD攻击
  pgd:
    epsilon: 0.1
    alpha: 0.01
    num_iter: 7
```

## 核心功能

### 1. 模型创建

```python
from adversarial_models import create_adversarial_model

# 创建对抗学习模型
config = {
    'name': 'my_adversarial_model',
    'input_dim': 3,
    'output_dim': 4,
    # ... 其他配置
}
model = create_adversarial_model(config)
```

### 2. 对抗样本生成

```python
# 生成对抗样本
x_adv = model.generate_adversarial_samples(x, y, attack_type='fgsm')

# 支持多种攻击类型
x_adv_fgsm = model.generate_adversarial_samples(x, y, attack_type='fgsm')
x_adv_pgd = model.generate_adversarial_samples(x, y, attack_type='pgd')
x_adv_cw = model.generate_adversarial_samples(x, y, attack_type='cw')
```

### 3. 训练模式切换

```python
# 设置训练模式
model.set_training_mode('pretrain')    # 预训练模式
model.set_training_mode('adversarial') # 对抗训练模式
model.set_training_mode('finetune')    # 微调模式
```

### 4. 前向传播

```python
# 不同模式的前向传播
outputs = model(x, mode='predict')      # 预测模式
outputs = model(x, mode='generate')     # 生成模式
outputs = model(x, mode='discriminate') # 判别模式
outputs = model(x, mode='adversarial')  # 对抗模式
```

## 训练流程

### 1. 预训练阶段

```python
# 只训练预测器
model.set_training_mode('pretrain')
trainer.pretrain()
```

### 2. 对抗训练阶段

```python
# 训练所有组件
model.set_training_mode('adversarial')
trainer.adversarial_train()
```

### 3. 微调阶段

```python
# 只训练预测器
model.set_training_mode('finetune')
trainer.finetune()
```

## 评估指标

### 1. 鲁棒性评估

```python
# 评估对抗鲁棒性
robustness_results = evaluate_adversarial_robustness(
    model=model,
    test_loader=test_loader,
    config=config
)
```

### 2. 攻击成功率

```python
# 计算攻击成功率
attack_success_rate = calculate_attack_success_rate(
    model=model,
    test_data=test_data,
    attack_type='fgsm'
)
```

## 测试

运行测试脚本验证模块功能：

```bash
# 测试对抗学习模块
python test_adversarial.py
```

测试包括：
- 模型创建和前向传播
- 对抗攻击生成
- 损失函数计算
- 数据增强功能
- 训练器功能
- 与主项目的集成

## 性能优化

### 1. GPU加速

```yaml
AdversarialTrain:
  device: 'cuda:2'  # 使用GPU训练
```

### 2. 批处理优化

```yaml
Data:
  batch_size: 64  # 调整批处理大小
```

### 3. 内存优化

```yaml
AdversarialModel:
  generator:
    num_layers: 4  # 减少网络层数
  discriminator:
    num_layers: 3
```

## 故障排除

### 1. 常见问题

- **训练不稳定**: 调整学习率和梯度惩罚权重
- **生成质量差**: 增加生成器网络深度
- **判别器过强**: 减少判别器训练步数
- **内存不足**: 减少批处理大小或网络深度

### 2. 调试技巧

```python
# 启用详细日志
config['AdversarialTrain']['log_interval'] = 1

# 保存中间结果
config['AdversarialTrain']['save_freq'] = 10
```

## 扩展功能

### 1. 自定义攻击方法

```python
class CustomAttack(AdversarialAugmentation):
    def generate_adversarial_samples(self, x, y, model):
        # 实现自定义攻击逻辑
        pass
```

### 2. 自定义损失函数

```python
class CustomLoss(nn.Module):
    def forward(self, predictions, targets):
        # 实现自定义损失逻辑
        pass
```

## 参考文献

1. Goodfellow, I., et al. "Generative adversarial nets." NIPS 2014.
2. Madry, A., et al. "Towards deep learning models resistant to adversarial attacks." ICLR 2018.
3. Carlini, N., & Wagner, D. "Towards evaluating the robustness of neural networks." S&P 2017.

## 许可证

本模块遵循项目主许可证。
