import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import os
warnings.filterwarnings('ignore')

# Import data preprocessing function
from data_preprocessing import data_preprocessing_0
# 导入SwanLab配置
from swanlab_config import init_swanlab, log_metrics, log_model_params, finish_run

# Create results directory
results_dir = "../results/neural_network"
os.makedirs(results_dir, exist_ok=True)

# Get preprocessed data
x_scaled, y_scaled, T_array = data_preprocessing_0()

# Set English fonts and style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
color_palette = sns.color_palette("husl", 8)

# 初始化SwanLab
run = init_swanlab("neural_network.py")

print("="*60)
print("Neural Network Regression Temperature Prediction Analysis")
print("="*60)
print(f"Data shapes: x_scaled={x_scaled.shape}, y_scaled={y_scaled.shape}, T_array={T_array.shape}")

# Data preparation
X = np.column_stack([x_scaled, y_scaled])  # Use x_scale and y_scale as features
y = T_array  # Prediction target is T

print(f"Feature matrix X shape: {X.shape}")
print(f"Target variable y shape: {y.shape}")

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Check if GPU is available
device = torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# Define the Residual Block for ResNet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
        
        # Shortcut connection to match the input and output dimensions
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x += residual  # Adding the residual connection
        return x

# Define the ResNet model
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.block1 = ResidualBlock(2, 64)  # 2 input features (x_scale, y_scale)
        self.block2 = ResidualBlock(64, 32)
        self.block3 = ResidualBlock(32, 16)
        self.fc = nn.Linear(16, 1)  # Output layer with a single neuron for regression
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc(x)
        return x

# Initialize the model and move it to the selected device
model = ResNetModel().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 100
batch_size = 256
train_losses = []

# Move data to device (GPU or CPU)
X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

print("\nStarting Neural Network model training...")
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Optimize the weights
        optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Predictions
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)

# Convert predictions to numpy
y_pred = y_pred_tensor.cpu().numpy().flatten()

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 记录模型参数和指标到SwanLab
model_params = {
    'model_type': 'neural_network',
    'architecture': 'ResNet',
    'epochs': epochs,
    'batch_size': batch_size,
    'learning_rate': 0.001,
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

# ==================== Visualization Section ====================

# 1. Prediction vs True Values Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Neural Network Regression Temperature Prediction Analysis Results', fontsize=16, fontweight='bold')

# 1.1 Prediction comparison
axes[0, 0].plot(y_test, color='#1f77b4', lw=2, alpha=0.8, label='True Values')
axes[0, 0].plot(y_pred, '--', color='#ff7f0e', lw=1.5, alpha=0.9, label='Predictions')
axes[0, 0].fill_between(range(len(y_test)), y_test, y_pred, color='gray', alpha=0.1)
axes[0, 0].set_title('Prediction vs True Values Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Temperature')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 1.2 Scatter plot
axes[0, 1].scatter(y_test, y_pred, alpha=0.6, color='#2ca02c')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_title('Predicted vs True Values Scatter Plot', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('True Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].grid(True, alpha=0.3)

# 1.3 Residual analysis
residuals = y_test - y_pred
axes[1, 0].scatter(y_pred, residuals, alpha=0.6, color='#d62728')
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_title('Residual Analysis', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].grid(True, alpha=0.3)

# 1.4 Residual distribution histogram
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

# 2. Neural Network Training Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2.1 Training loss curve
axes[0].plot(train_losses, color='#1f77b4', linewidth=2)
axes[0].set_title('Training Loss Curve', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].grid(True, alpha=0.3)

# 2.2 Model architecture visualization
layer_info = {
    'Input Layer': 2,
    'Hidden Layer 1': 64,
    'Hidden Layer 2': 32,
    'Hidden Layer 3': 16,
    'Output Layer': 1
}
layer_names = list(layer_info.keys())
layer_sizes = list(layer_info.values())

axes[1].bar(layer_names, layer_sizes, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'], alpha=0.8)
axes[1].set_title('Neural Network Architecture', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of Neurons')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'neural_network_analysis.png'), dpi=600, bbox_inches='tight')
plt.show()

# 3. Data distribution and correlation analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3.1 x_scale distribution
axes[0, 0].hist(x_scaled.flatten(), bins=30, alpha=0.7, color='#1f77b4', edgecolor='black')
axes[0, 0].set_title('x_scale Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('x_scale')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# 3.2 y_scale distribution
axes[0, 1].hist(y_scaled.flatten(), bins=30, alpha=0.7, color='#ff7f0e', edgecolor='black')
axes[0, 1].set_title('y_scale Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('y_scale')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# 3.3 T distribution
axes[1, 0].hist(T_array, bins=30, alpha=0.7, color='#2ca02c', edgecolor='black')
axes[1, 0].set_title('Temperature T Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Temperature T')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# 3.4 Correlation heatmap
data_df = pd.DataFrame({
    'x_scale': x_scaled.flatten(),
    'y_scale': y_scaled.flatten(),
    'T': T_array
})
corr_matrix = data_df.corr()

im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
axes[1, 1].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
axes[1, 1].set_xticklabels(corr_matrix.columns)
axes[1, 1].set_yticklabels(corr_matrix.columns)

# Add correlation coefficient text
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, ax=axes[1, 1])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'data_distribution_correlation.png'), dpi=600, bbox_inches='tight')
plt.show()

# 4. Cross-validation analysis
print("\nStarting cross-validation analysis...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Manual cross-validation for neural network
cv_mae = []
cv_rmse = []
cv_r2 = []

for train_idx, val_idx in kfold.split(X):
    X_train_cv, X_val_cv = torch.tensor(X[train_idx], dtype=torch.float32).to(device), torch.tensor(X[val_idx], dtype=torch.float32).to(device)
    y_train_cv, y_val_cv = torch.tensor(y[train_idx], dtype=torch.float32).view(-1, 1).to(device), torch.tensor(y[val_idx], dtype=torch.float32).view(-1, 1).to(device)
    
    model_cv = ResNetModel().to(device)
    optimizer_cv = optim.Adam(model_cv.parameters(), lr=0.001)
    
    # Training for each fold
    model_cv.train()
    for epoch in range(50):  # Reduced epochs for faster CV
        permutation = torch.randperm(X_train_cv.size(0))
        
        for i in range(0, X_train_cv.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_cv[indices], y_train_cv[indices]
            
            optimizer_cv.zero_grad()
            outputs = model_cv(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer_cv.step()
    
    model_cv.eval()
    with torch.no_grad():
        y_val_pred = model_cv(X_val_cv)
    
    y_val_pred_np = y_val_pred.cpu().numpy().flatten()
    y_val_np = y_val_cv.cpu().numpy().flatten()
    
    cv_mae.append(mean_absolute_error(y_val_np, y_val_pred_np))
    cv_rmse.append(np.sqrt(mean_squared_error(y_val_np, y_val_pred_np)))
    cv_r2.append(r2_score(y_val_np, y_val_pred_np))

cv_mae = np.array(cv_mae)
cv_rmse = np.array(cv_rmse)
cv_r2 = np.array(cv_r2)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 4.1 MAE cross-validation results
axes[0].plot(range(1, 6), cv_mae, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
axes[0].axhline(y=cv_mae.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean MAE: {cv_mae.mean():.4f}')
axes[0].set_title('Cross-Validation MAE Results', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('MAE')
axes[0].set_xticks(range(1, 6))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 4.2 RMSE cross-validation results
axes[1].plot(range(1, 6), cv_rmse, marker='s', linestyle='-', color='#ff7f0e', linewidth=2, markersize=8)
axes[1].axhline(y=cv_rmse.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean RMSE: {cv_rmse.mean():.4f}')
axes[1].set_title('Cross-Validation RMSE Results', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('RMSE')
axes[1].set_xticks(range(1, 6))
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 4.3 R² cross-validation results
axes[2].plot(range(1, 6), cv_r2, marker='^', linestyle='-', color='#2ca02c', linewidth=2, markersize=8)
axes[2].axhline(y=cv_r2.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean R²: {cv_r2.mean():.4f}')
axes[2].set_title('Cross-Validation R² Results', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Fold')
axes[2].set_ylabel('R²')
axes[2].set_xticks(range(1, 6))
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'cross_validation_results.png'), dpi=600, bbox_inches='tight')
plt.show()

# 5. Output detailed results
print("\n" + "="*60)
print("Detailed Analysis Results")
print("="*60)

print(f"\nNeural Network Model Parameters:")
print(f"Architecture: ResNet with Residual Blocks")
print(f"Input Features: 2 (x_scale, y_scale)")
print(f"Hidden Layers: [64, 32, 16]")
print(f"Output: 1 (Temperature)")
print(f"Optimizer: Adam")
print(f"Learning Rate: 0.001")
print(f"Loss Function: MSE")
print(f"Training Device: {device}")

print(f"\nCross-Validation Results:")
print(f"MAE - Mean: {cv_mae.mean():.4f}, Std: {cv_mae.std():.4f}")
print(f"RMSE - Mean: {cv_rmse.mean():.4f}, Std: {cv_rmse.std():.4f}")
print(f"R² - Mean: {cv_r2.mean():.4f}, Std: {cv_r2.std():.4f}")

print(f"\nData Statistics:")
print(f"x_scale - Mean: {np.mean(x_scaled.flatten()):.4f}, Std: {np.std(x_scaled.flatten()):.4f}")
print(f"y_scale - Mean: {np.mean(y_scaled.flatten()):.4f}, Std: {np.std(y_scaled.flatten()):.4f}")
print(f"T - Mean: {np.mean(T_array):.4f}, Std: {np.std(T_array):.4f}")

print(f"\nImages saved to: {results_dir}")
print("\nAnalysis completed!")

# 完成SwanLab运行
finish_run(run)

if __name__ == "__main__":
    pass
