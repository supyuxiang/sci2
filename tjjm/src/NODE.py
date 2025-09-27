# node_train.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
import os
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

warnings.filterwarnings('ignore')

# 导入数据预处理函数
from data_preprocessing import data_preprocessing_0
# 导入SwanLab配置
from swanlab_config import init_swanlab, log_metrics, log_model_params, finish_run

# 设置英文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
color_palette = sns.color_palette("husl", 8)

def train_node():
    # 初始化SwanLab
    run = init_swanlab("NODE.py")
    
    print("="*60)
    print("NODE (Neural Oblivious Decision Ensembles) Model Training Analysis")
    print("="*60)
    
    # 创建结果目录
    results_dir = "../results/node"
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取预处理后的数据
    x_scaled, y_scaled, T_array = data_preprocessing_0()
    
    # 数据准备
    X = np.column_stack([x_scaled, y_scaled])
    y = T_array
    
    print(f"Feature matrix X shape: {X.shape}")
    print(f"Target variable y shape: {y.shape}")
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # 标准化数据
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    scaler_y = StandardScaler()
    # 确保目标变量是2D格式，TabNetRegressor需要(n_samples, n_regression)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    # 模型训练
    print("\nStarting NODE model training...")
    node_model = TabNetRegressor(
        n_d=8, n_a=8, n_steps=3,
        gamma=1.3, n_independent=2, n_shared=2,
        cat_idxs=[], cat_dims=[],
        cat_emb_dim=1,
        lambda_sparse=1e-4, momentum=0.3, clip_value=2.0,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=None,
        scheduler_params={"milestones": [150, 250, 300, 350], "gamma": 1.3},
        mask_type="sparsemax",
        seed=42
    )
    
    node_model.fit(
        X_train=X_train_scaled, y_train=y_train_scaled,
        eval_set=[(X_test_scaled, y_test_scaled)],
        eval_name=['test'],
        eval_metric=['rmse'],
        max_epochs=100, patience=20,
        batch_size=32, virtual_batch_size=16,
        num_workers=0,
        drop_last=False,
        loss_fn=torch.nn.functional.mse_loss
    )
    
    # 预测
    y_pred_scaled = node_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Evaluation Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # 记录模型参数和指标到SwanLab
    model_params = {
        'n_d': node_model.n_d,
        'n_a': node_model.n_a,
        'n_steps': node_model.n_steps,
        'gamma': node_model.gamma,
        'n_independent': node_model.n_independent,
        'n_shared': node_model.n_shared,
        'lambda_sparse': node_model.lambda_sparse,
        'momentum': node_model.momentum,
        'clip_value': node_model.clip_value,
        'learning_rate': node_model.optimizer_params['lr'],
        'batch_size': 32,
        'virtual_batch_size': 16,
        'max_epochs': 100,
        'patience': 20,
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
    
    # 可视化特征重要性
    feature_importance = node_model.feature_importances_
    
    plt.figure(figsize=(10, 6))
    feature_names = ['x_scale', 'y_scale']
    plt.bar(feature_names, feature_importance, color=color_palette[:2])
    plt.title('NODE Model Feature Importance', fontsize=16, fontweight='bold')
    plt.ylabel('Feature Importance')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # 可视化预测结果
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('NODE Model Analysis Results', fontsize=16, fontweight='bold')
    
    # 预测对比
    axes[0, 0].plot(y_test, color='#1f77b4', lw=2, alpha=0.8, label='True Values')
    axes[0, 0].plot(y_pred, '--', color='#ff7f0e', lw=1.5, alpha=0.9, label='Predictions')
    axes[0, 0].fill_between(range(len(y_test)), y_test, y_pred, color='gray', alpha=0.1)
    axes[0, 0].set_title('Prediction vs True Values Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Temperature')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 散点图
    axes[0, 1].scatter(y_test, y_pred, alpha=0.6, color='#2ca02c')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_title('Predicted vs True Values Scatter Plot', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('True Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差分析
    residuals = y_test - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6, color='#d62728')
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title('Residual Analysis', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 残差分布
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
    
    # 输出详细结果
    print("\n" + "="*60)
    print("Detailed Analysis Results")
    print("="*60)
    
    print(f"\nNODE Model Parameters:")
    print(f"n_d: {node_model.n_d}")
    print(f"n_a: {node_model.n_a}")
    print(f"n_steps: {node_model.n_steps}")
    print(f"gamma: {node_model.gamma}")
    print(f"learning_rate: {node_model.optimizer_params['lr']}")
    
    print(f"\nFeature Importance:")
    for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
        print(f"{feature}: {importance:.6f}")
    
    print(f"\nImages saved to: {results_dir}")
    print("\nAnalysis completed!")
    
    # 完成SwanLab运行
    finish_run(run)

if __name__ == "__main__":
    train_node()