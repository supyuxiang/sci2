import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import warnings
import os
warnings.filterwarnings('ignore')

# Import data preprocessing function
from data_preprocessing import data_preprocessing_0
# 导入SwanLab配置
from swanlab_config import init_swanlab, log_metrics, log_model_params, finish_run

# Create results directory
results_dir = "../results/svr"
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
run = init_swanlab("SVR.py")

print("="*60)
print("Support Vector Regression Temperature Prediction Analysis")
print("="*60)
print(f"Data shapes: x_scaled={x_scaled.shape}, y_scaled={y_scaled.shape}, T_array={T_array.shape}")

# Data preparation
X = np.column_stack([x_scaled, y_scaled])[0:100000]  # Use x_scale and y_scale as features
y = T_array[0:100000]  # Prediction target is T

print(f"Feature matrix X shape: {X.shape}")
print(f"Target variable y shape: {y.shape}")

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Model training
print("\nStarting Support Vector Regression model training...")
svr_model = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=0.1
)

svr_model.fit(X_train, y_train)
y_pred = svr_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 记录模型参数和指标到SwanLab
model_params = {
    'kernel': svr_model.kernel,
    'C': svr_model.C,
    'gamma': svr_model.gamma,
    'epsilon': svr_model.epsilon,
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
fig.suptitle('Support Vector Regression Temperature Prediction Analysis Results', fontsize=16, fontweight='bold')

# 1.1 Prediction comparison
axes[0, 0].plot(y_test.flatten(), color='#1f77b4', lw=2, alpha=0.8, label='True Values')
axes[0, 0].plot(y_pred.flatten(), '--', color='#ff7f0e', lw=1.5, alpha=0.9, label='Predictions')
axes[0, 0].fill_between(range(len(y_test)), y_test.flatten(), y_pred.flatten(), color='gray', alpha=0.1)
axes[0, 0].set_title('Prediction vs True Values Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Temperature')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 1.2 Scatter plot
axes[0, 1].scatter(y_test.flatten(), y_pred.flatten(), alpha=0.6, color='#2ca02c')
axes[0, 1].plot([y_test.flatten().min(), y_test.flatten().max()], [y_test.flatten().min(), y_test.flatten().max()], 'r--', lw=2)
axes[0, 1].set_title('Predicted vs True Values Scatter Plot', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('True Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].grid(True, alpha=0.3)

# 1.3 Residual analysis
residuals = y_test.flatten() - y_pred.flatten()
axes[1, 0].scatter(y_pred.flatten(), residuals, alpha=0.6, color='#d62728')
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

# 2. SVR Model Analysis (Kernel and Support Vectors)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2.1 Support vectors visualization
n_support_vectors = len(svr_model.support_vectors_)
axes[0].bar(['Support Vectors', 'Non-Support Vectors'], 
            [n_support_vectors, len(X_train) - n_support_vectors], 
            color=['#ff7f0e', '#2ca02c'], alpha=0.8)
axes[0].set_title('Support Vectors Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].grid(True, alpha=0.3)

# 2.2 Model parameters
params = {
    'Kernel': svr_model.kernel,
    'C': svr_model.C,
    'Gamma': svr_model.gamma,
    'Epsilon': svr_model.epsilon
}
param_names = list(params.keys())
param_values = list(params.values())

# Filter numeric parameters for bar plot
numeric_params = {}
for name, value in params.items():
    if isinstance(value, (int, float)):
        numeric_params[name] = value

numeric_names = list(numeric_params.keys())
numeric_values = list(numeric_params.values())

if numeric_values:  # Only plot if we have numeric values
    axes[1].bar(range(len(numeric_names)), numeric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(numeric_names)], alpha=0.8)
    axes[1].set_xticks(range(len(numeric_names)))
    axes[1].set_xticklabels(numeric_names)
    axes[1].set_title('SVR Model Parameters (Numeric)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Parameter Value')
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'No numeric parameters to display', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('SVR Model Parameters', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'svr_model_analysis.png'), dpi=600, bbox_inches='tight')
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
axes[1, 0].hist(T_array.flatten(), bins=30, alpha=0.7, color='#2ca02c', edgecolor='black')
axes[1, 0].set_title('Temperature T Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Temperature T')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# 3.4 Correlation heatmap
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

# Multiple evaluation metrics
cv_mae = cross_val_score(svr_model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
cv_rmse = cross_val_score(svr_model, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
cv_r2 = cross_val_score(svr_model, X, y, cv=kfold, scoring='r2')

# Convert to positive values
cv_mae = -cv_mae
cv_rmse = -cv_rmse

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

print(f"\nSupport Vector Regression Model Parameters:")
print(f"Kernel: {svr_model.kernel}")
print(f"C: {svr_model.C}")
print(f"Gamma: {svr_model.gamma}")
print(f"Epsilon: {svr_model.epsilon}")
print(f"Number of Support Vectors: {len(svr_model.support_vectors_)}")

print(f"\nCross-Validation Results:")
print(f"MAE - Mean: {cv_mae.mean():.4f}, Std: {cv_mae.std():.4f}")
print(f"RMSE - Mean: {cv_rmse.mean():.4f}, Std: {cv_rmse.std():.4f}")
print(f"R² - Mean: {cv_r2.mean():.4f}, Std: {cv_r2.std():.4f}")

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
