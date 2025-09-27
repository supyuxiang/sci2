import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import BaggingRegressor
from sklearn.base import clone
import warnings
import os
warnings.filterwarnings('ignore')

# Import data preprocessing function
from data_preprocessing import data_preprocessing_0

# Create results directory
results_dir = "../results/best_base_bagging_lgb"
os.makedirs(results_dir, exist_ok=True)

# Get preprocessed data
x_scaled, y_scaled, T_array = data_preprocessing_0()

# Set English fonts and style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
color_palette = sns.color_palette("husl", 8)

print("="*60)
print("LightGBM Best Base Model with Bagging Optimization Analysis")
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

# ==================== Base Model Hyperparameter Optimization ====================
print("\nStarting LightGBM base model hyperparameter optimization...")

n_values = []
mae_list = []

# LightGBM hyperparameter optimization
for n in range(100, 10001, 200):  # Adjust step size for faster computation
    model = lgb.LGBMRegressor(
        n_estimators=n,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1  # Suppress warning messages
    )
    model.fit(X_train, y_train)
    pre_y = model.predict(X_test)
    mae_list.append(mean_absolute_error(pre_y, y_test))
    n_values.append(n)
    print(f"Processing n_estimators={n}")

# Visualizing the hyperparameter optimization process
plt.figure(figsize=(12, 6))
plt.plot(n_values, mae_list, color='royalblue', linewidth=2)
plt.scatter(n_values[::2], mae_list[::2], color='darkorange', marker='o')
best_idx = np.argmin(mae_list)
plt.scatter(n_values[best_idx], mae_list[best_idx], color='red', s=200, marker='*',
            label=f'Best Parameter: {n_values[best_idx]}\nMAE: {mae_list[best_idx]:.4f}')
plt.xlabel('n_estimators', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.title('LightGBM Hyperparameter Optimization Process', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(results_dir, 'hyperparameter_optimization.png'), dpi=600, bbox_inches='tight')
plt.show()

# Best base model
best_model = lgb.LGBMRegressor(
    n_estimators=n_values[best_idx], 
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
best_model.fit(X_train, y_train)
pre_y = best_model.predict(X_test)

# Model performance evaluation
mae = mean_absolute_error(pre_y, y_test)
rmse = np.sqrt(mean_squared_error(pre_y, y_test))
r2 = r2_score(y_test, pre_y)

print(f"\nBest Base Model Evaluation Results:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# ==================== Bagging Model Optimization ====================
print("\nStarting Bagging model optimization...")

bagging_mae_list = []
bagging_n_values = []

# Bagging model optimization
for n_bagging in range(1, 21):
    bagging_model = BaggingRegressor(
        estimator=clone(best_model),  # Corrected parameter name
        n_estimators=n_bagging,
        random_state=42,
        max_samples=0.8,
        n_jobs=-1
    )
    bagging_model.fit(X_train, y_train)
    pre_y_bagging = bagging_model.predict(X_test)
    mae_bagging = mean_absolute_error(pre_y_bagging, y_test)
    bagging_mae_list.append(mae_bagging)
    bagging_n_values.append(n_bagging)
    print(f"Bagging model count: {n_bagging}, MAE: {mae_bagging:.4f}")

# Visualizing Bagging results
plt.figure(figsize=(12, 6))
plt.plot(bagging_n_values, bagging_mae_list, color='green', linewidth=2)
plt.scatter(bagging_n_values, bagging_mae_list, color='red', marker='o')
best_bagging_idx = np.argmin(bagging_mae_list)
plt.scatter(
    bagging_n_values[best_bagging_idx],
    bagging_mae_list[best_bagging_idx],
    color='blue', s=200, marker='*',
    label=f'Optimal Bagging Model Count: {bagging_n_values[best_bagging_idx]}\nMAE: {bagging_mae_list[best_bagging_idx]:.4f}'
)
plt.xlabel('Number of Base Models', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.title('LightGBM Bagging Model Optimization (MAE vs Number of Models)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(results_dir, 'bagging_optimization.png'), dpi=600, bbox_inches='tight')
plt.show()

# ==================== Final Model Training and Evaluation ====================
print("\nTraining final optimized bagging model...")

# Final optimized bagging model
final_bagging_model = BaggingRegressor(
    estimator=clone(best_model),
    n_estimators=bagging_n_values[best_bagging_idx],
    random_state=42,
    max_samples=0.8,
    n_jobs=-1
)

final_bagging_model.fit(X_train, y_train)
y_pred_final = final_bagging_model.predict(X_test)

# Calculate evaluation metrics
mae_final = mean_absolute_error(y_test, y_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
r2_final = r2_score(y_test, y_pred_final)

print(f"\nFinal Bagging Model Evaluation Results:")
print(f"MAE: {mae_final:.4f}")
print(f"RMSE: {rmse_final:.4f}")
print(f"R²: {r2_final:.4f}")

# ==================== Visualization Section ====================

# 1. Prediction vs True Values Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('LightGBM Best Base Model with Bagging Optimization Results', fontsize=16, fontweight='bold')

# 1.1 Prediction comparison
axes[0, 0].plot(y_test, color='#1f77b4', lw=2, alpha=0.8, label='True Values')
axes[0, 0].plot(y_pred_final, '--', color='#ff7f0e', lw=1.5, alpha=0.9, label='Predictions')
axes[0, 0].fill_between(range(len(y_test)), y_test, y_pred_final, color='gray', alpha=0.1)
axes[0, 0].set_title('Prediction vs True Values Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Temperature')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 1.2 Scatter plot
axes[0, 1].scatter(y_test, y_pred_final, alpha=0.6, color='#2ca02c')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_title('Predicted vs True Values Scatter Plot', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('True Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].grid(True, alpha=0.3)

# 1.3 Residual analysis
residuals = y_test - y_pred_final
axes[1, 0].scatter(y_pred_final, residuals, alpha=0.6, color='#d62728')
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

# 2. Model Comparison Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2.1 Model performance comparison
models = ['Base Model', 'Bagging Model']
mae_scores = [mae, mae_final]
rmse_scores = [rmse, rmse_final]
r2_scores = [r2, r2_final]

x = np.arange(len(models))
width = 0.25

axes[0].bar(x - width, mae_scores, width, label='MAE', color='#1f77b4', alpha=0.8)
axes[0].bar(x, rmse_scores, width, label='RMSE', color='#ff7f0e', alpha=0.8)
axes[0].bar(x + width, r2_scores, width, label='R²', color='#2ca02c', alpha=0.8)

axes[0].set_xlabel('Models')
axes[0].set_ylabel('Score')
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2.2 Feature importance from base model
importances = best_model.feature_importances_
feature_names = ['x_scaled', 'y_scaled']
axes[1].bar(feature_names, importances, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
axes[1].set_title('Feature Importance (Base Model)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Importance')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_analysis.png'), dpi=600, bbox_inches='tight')
plt.show()

# 3. Cross-validation analysis
print("\nStarting cross-validation analysis...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Multiple evaluation metrics
cv_mae = cross_val_score(final_bagging_model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
cv_rmse = cross_val_score(final_bagging_model, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
cv_r2 = cross_val_score(final_bagging_model, X, y, cv=kfold, scoring='r2')

# Convert to positive values
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

print(f"\nBest Base Model Parameters:")
print(f"n_estimators: {n_values[best_idx]}")
print(f"learning_rate: 0.1")
print(f"max_depth: 6")
print(f"num_leaves: 31")

print(f"\nBagging Model Parameters:")
print(f"n_estimators: {bagging_n_values[best_bagging_idx]}")
print(f"max_samples: 0.8")

print(f"\nModel Performance Comparison:")
print(f"Base Model - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
print(f"Bagging Model - MAE: {mae_final:.4f}, RMSE: {rmse_final:.4f}, R²: {r2_final:.4f}")

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

if __name__ == "__main__":
    pass




