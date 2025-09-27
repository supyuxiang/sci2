import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import warnings
import os
warnings.filterwarnings('ignore')

# Import data preprocessing function
from data_preprocessing import data_preprocessing_0
# 导入SwanLab配置
from swanlab_config import init_swanlab, log_metrics, log_model_params, finish_run

# Create results directory
results_dir = "../results/best_base_model_choose_lgb"
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
run = init_swanlab("best_base_model_choose_lgb.py")

print("="*60)
print("LightGBM Best Base Model Selection Analysis")
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

# ==================== Model training and parameter analysis ====================
print("\nStarting LightGBM parameter optimization...")

n_values = []
mae_list = []

for n in range(100, 10001, 20):
    model = lgb.LGBMRegressor(
        n_estimators=n,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    pre_y = model.predict(X_test)
    mae_list.append(mean_absolute_error(pre_y, y_test))
    n_values.append(n)
    print(f"Processing n_estimators={n}")

# Parameter optimization visualization
plt.figure(figsize=(12, 6))
plt.plot(n_values, mae_list, color='royalblue', linewidth=2)
plt.scatter(n_values[::10], mae_list[::10], color='darkorange', marker='o')
best_idx = np.argmin(mae_list)
plt.scatter(n_values[best_idx], mae_list[best_idx], color='red', s=200, marker='*',
            label=f'Best Parameter: {n_values[best_idx]}\nMAE: {mae_list[best_idx]:.4f}')
plt.xlabel('n_estimators', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.title('LightGBM Parameter Optimization Process (MAE vs n_estimators)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(results_dir, 'parameter_optimization.png'), dpi=600, bbox_inches='tight')
plt.show()

# ==================== Best model analysis ====================
print("\nTraining best LightGBM model...")

best_model = lgb.LGBMRegressor(
    n_estimators=n_values[best_idx],
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Model metrics
mae = mean_absolute_error(y_pred, y_test)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
r2 = r2_score(y_test, y_pred)

print(f"\nBest Model Performance:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 记录模型参数和指标到SwanLab
model_params = {
    'best_n_estimators': n_values[best_idx],
    'learning_rate': best_model.learning_rate,
    'max_depth': best_model.max_depth,
    'num_leaves': best_model.num_leaves,
    'random_state': best_model.random_state,
    'n_features': X_train.shape[1],
    'n_samples': X_train.shape[0],
    'parameter_search_range': f"{n_values[0]}-{n_values[-1]}"
}
log_model_params(run, model_params)

metrics = {
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2,
    'best_n_estimators': n_values[best_idx],
    'min_mae': mae_list[best_idx]
}
log_metrics(run, metrics)

# ==================== Visualization Section ====================

# 1. Model Predictions vs True Values
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('LightGBM Best Base Model Analysis Results', fontsize=16, fontweight='bold')

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

# 1.4 Residual distribution
axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='#9467bd', edgecolor='black')
axes[1, 1].axvline(x=np.mean(residuals), color='red', linestyle='--', label=f'Mean: {np.mean(residuals):.2f}')
axes[1, 1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'best_model_analysis.png'), dpi=600, bbox_inches='tight')
plt.show()

# 2. Feature importance analysis
feature_names = ['x_scale', 'y_scale']
feature_importance = best_model.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance, color=color_palette[:2])
plt.title('LightGBM Best Model Feature Importance', fontsize=16, fontweight='bold')
plt.ylabel('Feature Importance')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=600, bbox_inches='tight')
plt.show()

# 3. Cross-validation analysis
print("\nStarting cross-validation analysis...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_mae = cross_val_score(best_model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
cv_rmse = cross_val_score(best_model, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
cv_r2 = cross_val_score(best_model, X, y, cv=kfold, scoring='r2')

cv_mae = -cv_mae
cv_rmse = -cv_rmse

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 3.1 MAE cross-validation results
axes[0].plot(range(1, 6), cv_mae, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
axes[0].axhline(y=cv_mae.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean MAE: {cv_mae.mean():.4f}')
axes[0].set_title('Cross-Validation MAE Results', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('MAE')
axes[0].set_xticks(range(1, 6))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 3.2 RMSE cross-validation results
axes[1].plot(range(1, 6), cv_rmse, marker='s', linestyle='-', color='#ff7f0e', linewidth=2, markersize=8)
axes[1].axhline(y=cv_rmse.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean RMSE: {cv_rmse.mean():.4f}')
axes[1].set_title('Cross-Validation RMSE Results', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('RMSE')
axes[1].set_xticks(range(1, 6))
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3.3 R² cross-validation results
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

# 4. Output detailed results
print("\n" + "="*60)
print("Detailed Analysis Results")
print("="*60)

print(f"\nLightGBM Best Model Parameters:")
print(f"Best n_estimators: {n_values[best_idx]}")
print(f"Learning Rate: {best_model.learning_rate}")
print(f"Max Depth: {best_model.max_depth}")
print(f"Num Leaves: {best_model.num_leaves}")

print(f"\nCross-Validation Results:")
print(f"MAE - Mean: {cv_mae.mean():.4f}, Std: {cv_mae.std():.4f}")
print(f"RMSE - Mean: {cv_rmse.mean():.4f}, Std: {cv_rmse.std():.4f}")
print(f"R² - Mean: {cv_r2.mean():.4f}, Std: {cv_r2.std():.4f}")

print(f"\nFeature Importance:")
for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
    print(f"{feature}: {importance:.6f}")

print(f"\nImages saved to: {results_dir}")
print("\nAnalysis completed!")

# 完成SwanLab运行
finish_run(run)

if __name__ == "__main__":
    pass

