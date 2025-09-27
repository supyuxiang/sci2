import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import warnings
import os
warnings.filterwarnings('ignore')

# Import data preprocessing function
from data_preprocessing import data_preprocessing_0
# 导入SwanLab配置
from swanlab_config import init_swanlab, log_metrics, log_model_params, finish_run

# Create results directory
results_dir = "../results/voting_model_rf_xgb_lgb"
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
run = init_swanlab("Voting_model_rf_xgb_lgb.py")

print("="*60)
print("Voting Model (RF, XGBoost, LightGBM) Temperature Prediction Analysis")
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

# ==================== Base Models Training ====================
print("\nStarting base models training...")

# Base models
base_rf = RandomForestRegressor(n_estimators=1000, random_state=42)
base_xgb = xgb.XGBRegressor(n_estimators=10000, random_state=42, verbosity=0)
base_lgb = lgb.LGBMRegressor(n_estimators=10000, random_state=42, verbose=-1)

# Voting regression model (simple average)
voting_model = VotingRegressor(estimators=[('rf', base_rf), ('xgb', base_xgb), ('lgb', base_lgb)])

# Model fitting
print("Training RandomForest...")
base_rf.fit(X_train, y_train)
print("Training XGBoost...")
base_xgb.fit(X_train, y_train)
print("Training LightGBM...")
base_lgb.fit(X_train, y_train)
print("Training Voting model...")
voting_model.fit(X_train, y_train)

# ==================== Model Predictions ====================
print("\nMaking predictions...")

y_pred_rf = base_rf.predict(X_test)
y_pred_xgb = base_xgb.predict(X_test)
y_pred_lgb = base_lgb.predict(X_test)
y_pred_voting = voting_model.predict(X_test)

# ==================== Model Evaluation ====================
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

rf_metrics = evaluate_model(y_test, y_pred_rf, "RandomForest")
xgb_metrics = evaluate_model(y_test, y_pred_xgb, "XGBoost")
lgb_metrics = evaluate_model(y_test, y_pred_lgb, "LightGBM")
voting_metrics = evaluate_model(y_test, y_pred_voting, "Voting")

# 记录模型参数和指标到SwanLab
model_params = {
    'rf_n_estimators': base_rf.n_estimators,
    'xgb_n_estimators': base_xgb.n_estimators,
    'lgb_n_estimators': base_lgb.n_estimators,
    'n_features': X_train.shape[1],
    'n_samples': X_train.shape[0],
    'voting_method': 'average'
}
log_model_params(run, model_params)

metrics = {
    'RF_MAE': rf_metrics['MAE'],
    'RF_RMSE': rf_metrics['RMSE'],
    'RF_R2': rf_metrics['R2'],
    'XGB_MAE': xgb_metrics['MAE'],
    'XGB_RMSE': xgb_metrics['RMSE'],
    'XGB_R2': xgb_metrics['R2'],
    'LGB_MAE': lgb_metrics['MAE'],
    'LGB_RMSE': lgb_metrics['RMSE'],
    'LGB_R2': lgb_metrics['R2'],
    'Voting_MAE': voting_metrics['MAE'],
    'Voting_RMSE': voting_metrics['RMSE'],
    'Voting_R2': voting_metrics['R2']
}
log_metrics(run, metrics)

# ==================== Visualization Section ====================

# 1. Model Predictions vs True Values Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Voting Model (RF, XGBoost, LightGBM) Analysis Results', fontsize=16, fontweight='bold')

# 1.1 All models prediction comparison
axes[0, 0].plot(y_test, color='#1f77b4', lw=2, alpha=0.8, label='True Values')
axes[0, 0].plot(y_pred_rf, '--', color='#ff7f0e', lw=1.5, alpha=0.9, label='RandomForest')
axes[0, 0].plot(y_pred_xgb, '--', color='#2ca02c', lw=1.5, alpha=0.9, label='XGBoost')
axes[0, 0].plot(y_pred_lgb, '--', color='#d62728', lw=1.5, alpha=0.9, label='LightGBM')
axes[0, 0].plot(y_pred_voting, '--', color='#9467bd', lw=2, alpha=0.9, label='Voting')
axes[0, 0].set_title('All Models Prediction Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Temperature')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 1.2 Voting model detailed comparison
axes[0, 1].plot(y_test, color='#1f77b4', lw=2, alpha=0.8, label='True Values')
axes[0, 1].plot(y_pred_voting, '--', color='#ff7f0e', lw=1.5, alpha=0.9, label='Voting Predictions')
axes[0, 1].fill_between(range(len(y_test)), y_test, y_pred_voting, color='gray', alpha=0.1)
axes[0, 1].set_title('Voting Model Prediction vs True Values', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Sample Index')
axes[0, 1].set_ylabel('Temperature')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 1.3 Scatter plot for voting model
axes[1, 0].scatter(y_test, y_pred_voting, alpha=0.6, color='#2ca02c')
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_title('Voting Model: Predicted vs True Values', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('True Values')
axes[1, 0].set_ylabel('Predicted Values')
axes[1, 0].grid(True, alpha=0.3)

# 1.4 Model performance comparison
models = ['RandomForest', 'XGBoost', 'LightGBM', 'Voting']
mae_scores = [rf_metrics['MAE'], xgb_metrics['MAE'], lgb_metrics['MAE'], voting_metrics['MAE']]
r2_scores = [rf_metrics['R2'], xgb_metrics['R2'], lgb_metrics['R2'], voting_metrics['R2']]

x = np.arange(len(models))
width = 0.35

axes[1, 1].bar(x - width/2, mae_scores, width, label='MAE', color='#1f77b4', alpha=0.8)
axes[1, 1].bar(x + width/2, r2_scores, width, label='R²', color='#ff7f0e', alpha=0.8)
axes[1, 1].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Models')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(models, rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'voting_model_analysis.png'), dpi=600, bbox_inches='tight')
plt.show()

# 2. Feature importance comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 2.1 RandomForest feature importance
rf_importance = base_rf.feature_importances_
feature_names = ['x_scale', 'y_scale']
axes[0].bar(feature_names, rf_importance, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
axes[0].set_title('RandomForest Feature Importance', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Importance')
axes[0].grid(True, alpha=0.3)

# 2.2 XGBoost feature importance
xgb_importance = base_xgb.feature_importances_
axes[1].bar(feature_names, xgb_importance, color=['#2ca02c', '#d62728'], alpha=0.8)
axes[1].set_title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Importance')
axes[1].grid(True, alpha=0.3)

# 2.3 LightGBM feature importance
lgb_importance = base_lgb.feature_importances_
axes[2].bar(feature_names, lgb_importance, color=['#9467bd', '#8c564b'], alpha=0.8)
axes[2].set_title('LightGBM Feature Importance', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Importance')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'feature_importance_comparison.png'), dpi=600, bbox_inches='tight')
plt.show()

# 3. Residual analysis for voting model
residuals = y_test - y_pred_voting

plt.figure(figsize=(12, 6))
plt.scatter(y_pred_voting, residuals, alpha=0.6, color='#d62728')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Voting Model Residual Analysis', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_dir, 'voting_residual_analysis.png'), dpi=600, bbox_inches='tight')
plt.show()

# 4. Output detailed results
print("\n" + "="*60)
print("Detailed Analysis Results")
print("="*60)

print(f"\nVoting Model Configuration:")
print(f"RandomForest n_estimators: {base_rf.n_estimators}")
print(f"XGBoost n_estimators: {base_xgb.n_estimators}")
print(f"LightGBM n_estimators: {base_lgb.n_estimators}")

print(f"\nModel Performance Summary:")
print(f"RandomForest - MAE: {rf_metrics['MAE']:.4f}, RMSE: {rf_metrics['RMSE']:.4f}, R²: {rf_metrics['R2']:.4f}")
print(f"XGBoost - MAE: {xgb_metrics['MAE']:.4f}, RMSE: {xgb_metrics['RMSE']:.4f}, R²: {xgb_metrics['R2']:.4f}")
print(f"LightGBM - MAE: {lgb_metrics['MAE']:.4f}, RMSE: {lgb_metrics['RMSE']:.4f}, R²: {lgb_metrics['R2']:.4f}")
print(f"Voting - MAE: {voting_metrics['MAE']:.4f}, RMSE: {voting_metrics['RMSE']:.4f}, R²: {voting_metrics['R2']:.4f}")

print(f"\nImages saved to: {results_dir}")
print("\nAnalysis completed!")

# 完成SwanLab运行
finish_run(run)

if __name__ == "__main__":
    pass
















