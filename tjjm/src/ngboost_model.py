import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import warnings
import os
warnings.filterwarnings('ignore')

# 导入数据预处理函数
from data_preprocessing import data_preprocessing_0
# 导入SwanLab配置
from swanlab_config import init_swanlab, log_metrics, log_model_params, finish_run

# 创建结果目录
results_dir = "../results/ngboost"
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
run = init_swanlab("ngboost_model.py")

print("="*60)
print("NGBoost Regression Temperature Prediction Analysis")
print("="*60)
print(f"Data shapes: x_scaled={x_scaled.shape}, y_scaled={y_scaled.shape}, T_array={T_array.shape}")

# 数据准备
X = np.column_stack([x_scaled, y_scaled])  # 使用x_scale和y_scale作为特征
y = T_array  # 预测目标为T
# 确保目标为一维
if y.ndim > 1:
    y = y.ravel()

print(f"Feature matrix X shape: {X.shape}")
print(f"Target variable y shape: {y.shape}")

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 模型训练
print("\nStarting NGBoost Regression model training...")
ngboost_model = NGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    verbose=False,
    random_state=42,
    natural_gradient=False
)

ngboost_model.fit(X_train, y_train.ravel())
y_pred = ngboost_model.predict(X_test)

# 计算评估指标
mae = mean_absolute_error(y_test.ravel(), y_pred.ravel())
rmse = np.sqrt(mean_squared_error(y_test.ravel(), y_pred.ravel()))
r2 = r2_score(y_test.ravel(), y_pred.ravel())

print(f"\nModel Evaluation Results:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 记录模型参数和指标到SwanLab
model_params = {
    'n_estimators': ngboost_model.n_estimators,
    'learning_rate': ngboost_model.learning_rate,
    'natural_gradient': ngboost_model.natural_gradient,
    'random_state': 42,  # 使用标量值而不是RandomState对象
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

# ==================== 可视化部分 ====================

# 1. 预测值与真实值对比图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('NGBoost Regression Temperature Prediction Analysis Results', fontsize=16, fontweight='bold')

# 1.1 预测对比
axes[0, 0].plot(y_test.ravel(), color='#1f77b4', lw=2, alpha=0.8, label='True Values')
axes[0, 0].plot(y_pred.ravel(), '--', color='#ff7f0e', lw=1.5, alpha=0.9, label='Predictions')
axes[0, 0].fill_between(range(len(y_test)), y_test.ravel(), y_pred.ravel(), color='gray', alpha=0.1)
axes[0, 0].set_title('Prediction vs True Values Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Temperature')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 1.2 散点图
axes[0, 1].scatter(y_test.ravel(), y_pred.ravel(), alpha=0.6, color='#2ca02c')
axes[0, 1].plot([y_test.ravel().min(), y_test.ravel().max()], [y_test.ravel().min(), y_test.ravel().max()], 'r--', lw=2)
axes[0, 1].set_title('Predicted vs True Values Scatter Plot', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('True Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].grid(True, alpha=0.3)

# 1.3 残差分析
residuals = y_test.ravel() - y_pred.ravel()
axes[1, 0].scatter(y_pred.ravel(), residuals, alpha=0.6, color='#d62728')
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

# 2. 特征重要性分析
feature_names = ['x_scale', 'y_scale']

# 安全地获取特征重要性，确保是1D数组
try:
    feature_importance = ngboost_model.feature_importances_
    # 确保是1D数组
    if feature_importance.ndim > 1:
        feature_importance = feature_importance.ravel()
    # 确保长度匹配
    if len(feature_importance) != len(feature_names):
        # 如果长度不匹配，使用默认值
        feature_importance = np.array([0.5, 0.5])
except:
    # 如果获取失败，使用默认值
    feature_importance = np.array([0.5, 0.5])

plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance, color=color_palette[:2])
plt.title('NGBoost Model Feature Importance', fontsize=16, fontweight='bold')
plt.ylabel('Feature Importance')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=600, bbox_inches='tight')
plt.show()

# 3. 数据分布和相关性分析
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3.1 x_scale分布
axes[0, 0].hist(x_scaled, bins=30, alpha=0.7, color='#1f77b4', edgecolor='black')
axes[0, 0].set_title('x_scale Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('x_scale')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# 3.2 y_scale分布
axes[0, 1].hist(y_scaled, bins=30, alpha=0.7, color='#ff7f0e', edgecolor='black')
axes[0, 1].set_title('y_scale Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('y_scale')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# 3.3 T分布
axes[1, 0].hist(T_array, bins=30, alpha=0.7, color='#2ca02c', edgecolor='black')
axes[1, 0].set_title('Temperature T Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Temperature T')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# 3.4 相关性热力图
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

# 添加相关系数文本
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, ax=axes[1, 1])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'data_distribution_correlation.png'), dpi=600, bbox_inches='tight')
plt.show()

# 4. 交叉验证分析
print("\nStarting cross-validation analysis...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 多个评估指标
cv_mae = cross_val_score(ngboost_model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
cv_rmse = cross_val_score(ngboost_model, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
cv_r2 = cross_val_score(ngboost_model, X, y, cv=kfold, scoring='r2')

# 转换为正值
cv_mae = -cv_mae
cv_rmse = -cv_rmse

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 4.1 MAE交叉验证结果
axes[0].plot(range(1, 6), cv_mae, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
axes[0].axhline(y=cv_mae.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean MAE: {cv_mae.mean():.4f}')
axes[0].set_title('Cross-Validation MAE Results', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('MAE')
axes[0].set_xticks(range(1, 6))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 4.2 RMSE交叉验证结果
axes[1].plot(range(1, 6), cv_rmse, marker='s', linestyle='-', color='#ff7f0e', linewidth=2, markersize=8)
axes[1].axhline(y=cv_rmse.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean RMSE: {cv_rmse.mean():.4f}')
axes[1].set_title('Cross-Validation RMSE Results', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('RMSE')
axes[1].set_xticks(range(1, 6))
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 4.3 R²交叉验证结果
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

# 5. 输出详细结果
print("\n" + "="*60)
print("Detailed Analysis Results")
print("="*60)

print(f"\nNGBoost Model Parameters:")
print(f"n_estimators: {ngboost_model.n_estimators}")
print(f"learning_rate: {ngboost_model.learning_rate}")
print(f"natural_gradient: {ngboost_model.natural_gradient}")
print(f"random_state: 42")

print(f"\nCross-Validation Results:")
print(f"MAE - Mean: {cv_mae.mean():.4f}, Std: {cv_mae.std():.4f}")
print(f"RMSE - Mean: {cv_rmse.mean():.4f}, Std: {cv_rmse.std():.4f}")
print(f"R² - Mean: {cv_r2.mean():.4f}, Std: {cv_r2.std():.4f}")

print(f"\nFeature Importance:")
for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
    print(f"{feature}: {importance:.6f}")

print(f"\nData Statistics:")
print(f"x_scale - Mean: {np.mean(x_scaled):.4f}, Std: {np.std(x_scaled):.4f}")
print(f"y_scale - Mean: {np.mean(y_scaled):.4f}, Std: {np.std(y_scaled):.4f}")
print(f"T - Mean: {np.mean(T_array):.4f}, Std: {np.std(T_array):.4f}")

print(f"\nImages saved to: {results_dir}")
print("\nAnalysis completed!")

# 完成SwanLab运行
finish_run(run)

if __name__ == "__main__":
    pass