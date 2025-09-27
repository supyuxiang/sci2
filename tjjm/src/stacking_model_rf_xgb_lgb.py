import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
import os
import torch
import torch.nn as nn
import torch.optim as optim
warnings.filterwarnings('ignore')

# Import data preprocessing function
from data_preprocessing import data_preprocessing_0
# 导入SwanLab配置
from swanlab_config import init_swanlab, log_metrics, log_model_params, finish_run

# Create results directory
results_dir = "../results/stacking_model_rf_xgb_lgb"
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
run = init_swanlab("stacking_model_rf_xgb_lgb.py")

print("="*60)
print("Stacking Model (RF, XGBoost, LightGBM) Temperature Prediction Analysis")
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

# ==================== MLP Model for Stacking ====================
class MLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_size=100, learning_rate=0.001, epochs=100, batch_size=32):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        # 标准化输入
        X_scaled = self.scaler.fit_transform(X)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        
        # 创建模型
        input_size = X.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        # 训练模型
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        
        # 标准化输入
        X_scaled = self.scaler.transform(X)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_scaled)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.numpy().flatten()
    
    def get_params(self, deep=True):
        return {
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# ==================== Base Models Training ====================
print("\nStarting base models training...")

# Base models
base_rf = RandomForestRegressor(n_estimators=1000, random_state=42)
base_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
base_lgb = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)

# Model fitting
print("Training RandomForest...")
base_rf.fit(X_train, y_train)
print("Training XGBoost...")
base_xgb.fit(X_train, y_train)
print("Training LightGBM...")
base_lgb.fit(X_train, y_train)
print('Base models training completed!')

# Stacking model with LinearRegression as meta-learner (for compatibility)
stacking_model = StackingRegressor(
    estimators=[('rf', base_rf), ('xgb', base_xgb), ('lgb', base_lgb)],
    final_estimator=LinearRegression()
)

print("Training Stacking model...")
stacking_model.fit(X_train, y_train)
print('Stacking model training completed!')

# 单独训练MLP作为比较
print("Training MLP model...")
mlp_model = MLPRegressor(hidden_size=100, learning_rate=0.001, epochs=100, batch_size=32)

# 获取base models的预测作为MLP的输入特征
y_pred_rf_train = base_rf.predict(X_train)
y_pred_xgb_train = base_xgb.predict(X_train)
y_pred_lgb_train = base_lgb.predict(X_train)

# 组合base models的预测作为MLP的输入
X_meta_train = np.column_stack([y_pred_rf_train, y_pred_xgb_train, y_pred_lgb_train])
mlp_model.fit(X_meta_train, y_train)
print('MLP model training completed!')

# ==================== Model Predictions ====================
print("\nMaking predictions...")

y_pred_rf = base_rf.predict(X_test)
y_pred_xgb = base_xgb.predict(X_test)
y_pred_lgb = base_lgb.predict(X_test)
y_pred_stacking = stacking_model.predict(X_test)

# MLP预测
y_pred_rf_test = base_rf.predict(X_test)
y_pred_xgb_test = base_xgb.predict(X_test)
y_pred_lgb_test = base_lgb.predict(X_test)
X_meta_test = np.column_stack([y_pred_rf_test, y_pred_xgb_test, y_pred_lgb_test])
y_pred_mlp = mlp_model.predict(X_meta_test)

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
stacking_metrics = evaluate_model(y_test, y_pred_stacking, "Stacking")
mlp_metrics = evaluate_model(y_test, y_pred_mlp, "MLP")

# 记录模型参数和指标到SwanLab
model_params = {
    'model_type': 'stacking_comparison',
    'base_models': ['RandomForest', 'XGBoost', 'LightGBM'],
    'stacking_final_estimator': 'LinearRegression',
    'mlp_hidden_size': 100,
    'mlp_learning_rate': 0.001,
    'mlp_epochs': 100,
    'n_features': X_train.shape[1],
    'n_samples': X_train.shape[0]
}
log_model_params(run, model_params)

# 记录所有模型的指标
all_metrics = {
    'RF_MAE': rf_metrics['MAE'],
    'RF_RMSE': rf_metrics['RMSE'],
    'RF_R2': rf_metrics['R2'],
    'XGB_MAE': xgb_metrics['MAE'],
    'XGB_RMSE': xgb_metrics['RMSE'],
    'XGB_R2': xgb_metrics['R2'],
    'LGB_MAE': lgb_metrics['MAE'],
    'LGB_RMSE': lgb_metrics['RMSE'],
    'LGB_R2': lgb_metrics['R2'],
    'Stacking_MAE': stacking_metrics['MAE'],
    'Stacking_RMSE': stacking_metrics['RMSE'],
    'Stacking_R2': stacking_metrics['R2'],
    'MLP_MAE': mlp_metrics['MAE'],
    'MLP_RMSE': mlp_metrics['RMSE'],
    'MLP_R2': mlp_metrics['R2']
}
log_metrics(run, all_metrics)

# ==================== Visualization Section ====================

# 1. Model Predictions vs True Values Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Stacking Model (RF, XGBoost, LightGBM) Analysis Results', fontsize=16, fontweight='bold')

# 1.1 All models prediction comparison
colors = ['red', 'blue', 'green', 'purple']
models = ['RandomForest', 'XGBoost', 'LightGBM', 'Stacking']
y_preds = [y_pred_rf, y_pred_xgb, y_pred_lgb, y_pred_stacking]

for i, (model, color) in enumerate(zip(models, colors)):
    axes[0, 0].scatter(y_test, y_preds[i], color=color, label=model, alpha=0.7, s=20)
axes[0, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label='Ideal')
axes[0, 0].set_title('All Models Predictions vs True Values', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('True Values')
axes[0, 0].set_ylabel('Predicted Values')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 1.2 Stacking model detailed analysis
axes[0, 1].plot(y_test, color='#1f77b4', lw=2, alpha=0.8, label='True Values')
axes[0, 1].plot(y_pred_stacking, '--', color='#ff7f0e', lw=1.5, alpha=0.9, label='Stacking Predictions')
axes[0, 1].fill_between(range(len(y_test)), y_test, y_pred_stacking, color='gray', alpha=0.1)
axes[0, 1].set_title('Stacking Model Prediction vs True Values', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Sample Index')
axes[0, 1].set_ylabel('Temperature')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 1.3 Residual analysis for stacking model
residuals = y_test - y_pred_stacking
axes[1, 0].scatter(y_pred_stacking, residuals, alpha=0.6, color='#d62728')
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_title('Stacking Model Residual Analysis', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].grid(True, alpha=0.3)

# 1.4 Residual distribution histogram
axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='#9467bd', edgecolor='black')
axes[1, 1].axvline(x=np.mean(residuals), color='red', linestyle='--', label=f'Mean: {np.mean(residuals):.2f}')
axes[1, 1].set_title('Stacking Model Residual Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'prediction_analysis.png'), dpi=600, bbox_inches='tight')
plt.show()

# 2. Model Performance Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2.1 Performance metrics comparison
metrics_names = ['MAE', 'RMSE', 'R²']
rf_scores = [rf_metrics['MAE'], rf_metrics['RMSE'], rf_metrics['R2']]
xgb_scores = [xgb_metrics['MAE'], xgb_metrics['RMSE'], xgb_metrics['R2']]
lgb_scores = [lgb_metrics['MAE'], lgb_metrics['RMSE'], lgb_metrics['R2']]
stacking_scores = [stacking_metrics['MAE'], stacking_metrics['RMSE'], stacking_metrics['R2']]

x = np.arange(len(metrics_names))
width = 0.2

axes[0].bar(x - 1.5*width, rf_scores, width, label='RandomForest', color='red', alpha=0.8)
axes[0].bar(x - 0.5*width, xgb_scores, width, label='XGBoost', color='blue', alpha=0.8)
axes[0].bar(x + 0.5*width, lgb_scores, width, label='LightGBM', color='green', alpha=0.8)
axes[0].bar(x + 1.5*width, stacking_scores, width, label='Stacking', color='purple', alpha=0.8)

axes[0].set_xlabel('Metrics')
axes[0].set_ylabel('Score')
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_names)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2.2 Feature importance comparison
feature_names = ['x_scaled', 'y_scaled']
rf_importance = base_rf.feature_importances_
xgb_importance = base_xgb.feature_importances_
lgb_importance = base_lgb.feature_importances_

x = np.arange(len(feature_names))
width = 0.25

axes[1].bar(x - width, rf_importance, width, label='RandomForest', color='red', alpha=0.8)
axes[1].bar(x, xgb_importance, width, label='XGBoost', color='blue', alpha=0.8)
axes[1].bar(x + width, lgb_importance, width, label='LightGBM', color='green', alpha=0.8)

axes[1].set_xlabel('Features')
axes[1].set_ylabel('Importance')
axes[1].set_title('Feature Importance Comparison', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(feature_names)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=600, bbox_inches='tight')
plt.show()

# 3. Cross-validation analysis for stacking model
print("\nStarting cross-validation analysis for stacking model...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Multiple evaluation metrics
cv_mae = cross_val_score(stacking_model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
cv_rmse = cross_val_score(stacking_model, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
cv_r2 = cross_val_score(stacking_model, X, y, cv=kfold, scoring='r2')

# Convert to positive values
cv_mae = -cv_mae
cv_rmse = -cv_rmse

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 3.1 MAE cross-validation results
axes[0].plot(range(1, 6), cv_mae, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
axes[0].axhline(y=cv_mae.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean MAE: {cv_mae.mean():.4f}')
axes[0].set_title('Stacking Model Cross-Validation MAE Results', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('MAE')
axes[0].set_xticks(range(1, 6))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 3.2 RMSE cross-validation results
axes[1].plot(range(1, 6), cv_rmse, marker='s', linestyle='-', color='#ff7f0e', linewidth=2, markersize=8)
axes[1].axhline(y=cv_rmse.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean RMSE: {cv_rmse.mean():.4f}')
axes[1].set_title('Stacking Model Cross-Validation RMSE Results', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('RMSE')
axes[1].set_xticks(range(1, 6))
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3.3 R² cross-validation results
axes[2].plot(range(1, 6), cv_r2, marker='^', linestyle='-', color='#2ca02c', linewidth=2, markersize=8)
axes[2].axhline(y=cv_r2.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean R²: {cv_r2.mean():.4f}')
axes[2].set_title('Stacking Model Cross-Validation R² Results', fontsize=14, fontweight='bold')
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

print(f"\nBase Models Parameters:")
print(f"RandomForest - n_estimators: 1000")
print(f"XGBoost - n_estimators: 100")
print(f"LightGBM - n_estimators: 100")
print(f"Stacking - Meta-learner: MLP (hidden_size=100, lr=0.001, epochs=100)")

print(f"\nModel Performance Comparison:")
print(f"RandomForest - MAE: {rf_metrics['MAE']:.4f}, RMSE: {rf_metrics['RMSE']:.4f}, R²: {rf_metrics['R2']:.4f}")
print(f"XGBoost - MAE: {xgb_metrics['MAE']:.4f}, RMSE: {xgb_metrics['RMSE']:.4f}, R²: {xgb_metrics['R2']:.4f}")
print(f"LightGBM - MAE: {lgb_metrics['MAE']:.4f}, RMSE: {lgb_metrics['RMSE']:.4f}, R²: {lgb_metrics['R2']:.4f}")
print(f"Stacking - MAE: {stacking_metrics['MAE']:.4f}, RMSE: {stacking_metrics['RMSE']:.4f}, R²: {stacking_metrics['R2']:.4f}")

print(f"\nStacking Model Cross-Validation Results:")
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
