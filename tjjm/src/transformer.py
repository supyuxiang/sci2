import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
import warnings
import os
warnings.filterwarnings('ignore')

# 导入数据预处理函数
from data_preprocessing import data_preprocessing_0
# 导入SwanLab配置
from swanlab_config import init_swanlab, log_metrics, log_model_params, finish_run

# 创建结果目录
results_dir = "../results/transformer"
os.makedirs(results_dir, exist_ok=True)

# 获取预处理后的数据
x_scaled, y_scaled, T_array = data_preprocessing_0()

# 设置英文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
color_palette = sns.color_palette("husl", 8)

# 初始化SwanLab
run = init_swanlab("transformer.py")

print("="*60)
print("Transformer Temperature Prediction Analysis")
print("="*60)
print(f"Data shapes: x_scaled={x_scaled.shape}, y_scaled={y_scaled.shape}, T_array={T_array.shape}")

# 数据准备
X = np.column_stack([x_scaled, y_scaled])  # 使用x_scale和y_scale作为特征
y = T_array  # 预测目标为T

print(f"Feature matrix X shape: {X.shape}")
print(f"Target variable y shape: {y.shape}")

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)

# 统一批大小
batch_size = 32

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义Transformer模型
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # 添加序列维度 (batch_size, seq_len, input_dim)
        x = x.unsqueeze(1)  # 添加序列长度维度
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.output_layer(x)
        return x.squeeze(-1)

# 模型训练
print("\nStarting Transformer model training...")
if torch.cuda.is_available():
    device = torch.device("cuda:8")
    try:
        torch.cuda.set_device(8)
    except Exception:
        pass
else:
    device = torch.device("cpu")
input_dim = X_train.shape[1]
model = TransformerRegressor(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 100
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(train_loader))
    
    # 验证
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            test_loss += criterion(outputs, batch_y).item()
        
        test_losses.append(test_loss / len(test_loader))
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

# 预测
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor.to(device)).cpu().numpy()

# 计算评估指标（确保为一维）
mae = mean_absolute_error(np.ravel(y_test), np.ravel(y_pred))
rmse = np.sqrt(mean_squared_error(np.ravel(y_test), np.ravel(y_pred)))
r2 = r2_score(np.ravel(y_test), np.ravel(y_pred))

# 记录模型参数和指标到SwanLab
model_params = {
    'model_type': 'transformer',
    'input_dim': input_dim,
    'hidden_dim': 64,
    'num_heads': 4,
    'num_layers': 2,
    'epochs': num_epochs,
    'learning_rate': 0.001,
    'batch_size': batch_size,
    'n_features': X_train.shape[1],
    'n_samples': X_train.shape[0]
}
log_model_params(run, model_params)

metrics = {
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2
}
log_metrics(run, metrics)

print(f"\nModel Evaluation Results:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# ==================== 可视化部分 ====================

# 1. 预测值与真实值对比图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Transformer Temperature Prediction Analysis Results', fontsize=16, fontweight='bold')

# 1.1 预测对比
axes[0, 0].plot(np.ravel(y_test), color='#1f77b4', lw=2, alpha=0.8, label='True Values')
axes[0, 0].plot(np.ravel(y_pred), '--', color='#ff7f0e', lw=1.5, alpha=0.9, label='Predictions')
axes[0, 0].fill_between(range(len(y_test)), np.ravel(y_test), np.ravel(y_pred), color='gray', alpha=0.1)
axes[0, 0].set_title('Prediction vs True Values Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Temperature')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 1.2 散点图
axes[0, 1].scatter(np.ravel(y_test), np.ravel(y_pred), alpha=0.6, color='#2ca02c')
axes[0, 1].plot([np.ravel(y_test).min(), np.ravel(y_test).max()], [np.ravel(y_test).min(), np.ravel(y_test).max()], 'r--', lw=2)
axes[0, 1].set_title('Predicted vs True Values Scatter Plot', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('True Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].grid(True, alpha=0.3)

# 1.3 残差分析
residuals = np.ravel(y_test) - np.ravel(y_pred)
axes[1, 0].scatter(np.ravel(y_pred), residuals, alpha=0.6, color='#d62728')
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_title('Residual Analysis', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].grid(True, alpha=0.3)

# 1.4 残差分布直方图
axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='#9467bd', edgecolor='black')
axes[1, 1].axvline(x=np.mean(residuals), color='red', linestyle='--', label=f'Mean: {np.mean(residuals):.2f}')
axes[1, 1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'prediction_analysis.png'), dpi=600, bbox_inches='tight')
plt.show()

# 2. 训练过程损失曲线
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_losses, label='Training Loss', color='#1f77b4')
ax.plot(test_losses, label='Validation Loss', color='#ff7f0e')
ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_dir, 'training_loss.png'), dpi=600, bbox_inches='tight')
plt.show()

# 3. 数据分布和相关性分析
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3.1 x_scale分布
axes[0, 0].hist(x_scaled.flatten(), bins=30, alpha=0.7, color='#1f77b4', edgecolor='black')
axes[0, 0].set_title('x_scale Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('x_scale')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# 3.2 y_scale分布
axes[0, 1].hist(y_scaled.flatten(), bins=30, alpha=0.7, color='#ff7f0e', edgecolor='black')
axes[0, 1].set_title('y_scale Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('y_scale')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# 3.3 T分布
axes[1, 0].hist(T_array.flatten(), bins=30, alpha=0.7, color='#2ca02c', edgecolor='black')
axes[1, 0].set_title('Temperature T Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Temperature T')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# 3.4 相关性热力图
data_df = pd.DataFrame({
    'x_scale': x_scaled.flatten(),
    'y_scale': y_scaled.flatten(),
    'T': T_array.flatten()
})
corr_matrix = data_df.corr()

im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
axes[1, 1].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
axes[1, 1].set_xticklabels(corr_matrix.columns)
axes[1, 1].set_yticklabels(corr_matrix.columns)

# 添加相关系数文本
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, ax=axes[1, 1])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'data_distribution_correlation.png'), dpi=600, bbox_inches='tight')
plt.show()

# 5. 输出详细结果
print("\n" + "="*60)
print("Detailed Analysis Results")
print("="*60)

print(f"\nTransformer Model Architecture:")
print(model)

print(f"\nModel Evaluation Results:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

print(f"\nData Statistics:")
print(f"x_scale - Mean: {np.mean(x_scaled.flatten()):.4f}, Std: {np.std(x_scaled.flatten()):.4f}")
print(f"y_scale - Mean: {np.mean(y_scaled.flatten()):.4f}, Std: {np.std(y_scaled.flatten()):.4f}")
print(f"T - Mean: {np.mean(T_array.flatten()):.4f}, Std: {np.std(T_array.flatten()):.4f}")

print(f"\nImages saved to: {results_dir}")
print("\nAnalysis completed!")

# 完成SwanLab运行
finish_run(run)

if __name__ == "__main__":
    pass