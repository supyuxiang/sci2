###XGB
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import BaggingRegressor
from sklearn.base import clone
import xgboost as xgb

# Visualization settings
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams['font.sans-serif'] = ['Arial']  # Use Arial font for English text
plt.rcParams['axes.unicode_minus'] = False

# Data loading & preprocessing
data = pd.read_csv("仿真数据1.csv", header=None)
data.columns = ['X', 'Y', 'Z', 'Temperature']
data = data.apply(pd.to_numeric, errors='coerce').dropna()

# Parameter optimization (Base model)
X = data[['X', 'Y', 'Z']].values
Y = data['Temperature'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

n_values = []
mae_list = []

# Finding the optimal n_estimators for XGBoost
for n in range(100, 10001, 200):
    model = xgb.XGBRegressor(n_estimators=n, random_state=42)
    model.fit(X_train, Y_train)
    pre_y = model.predict(X_test)
    mae_list.append(mean_absolute_error(pre_y, Y_test))
    n_values.append(n)
    print(n)

# Visualizing the parameter optimization process
plt.figure(figsize=(12, 6))
plt.plot(n_values, mae_list, color='royalblue', linewidth=2)
plt.scatter(n_values[::10], mae_list[::10], color='darkorange', marker='o')
best_idx = np.argmin(mae_list)
plt.scatter(n_values[best_idx], mae_list[best_idx], color='red', s=200, marker='*',
            label=f'Best Parameter: {n_values[best_idx]}\nMAE: {mae_list[best_idx]:.4f}')
plt.xlabel('n_estimators', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.title('XGB :Parameter Optimization Process (MAE vs n_estimators)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# Optimized base model
best_model = xgb.XGBRegressor(n_estimators=n_values[best_idx], random_state=42)
best_model.fit(X_train, Y_train)
pre_y = best_model.predict(X_test)

# Performance metrics
metrics = {
    'MAE': mean_absolute_error(pre_y, Y_test),
}

# Bagging model training and selection
bagging_mae_list = []
bagging_n_values = []

# Key modification: Changed base_estimator to estimator
for n_bagging in range(1, 21):  # Testing from 10 to 100 base models
    bagging_model = BaggingRegressor(
        estimator=clone(best_model),  # Corrected parameter name
        n_estimators=n_bagging,
        random_state=42,
        max_samples=0.8,  # 80% Bootstrap samples
        n_jobs=-1         # Using all CPU cores
    )
    bagging_model.fit(X_train, Y_train)
    pre_y_bagging = bagging_model.predict(X_test)
    mae_bagging = mean_absolute_error(pre_y_bagging, Y_test)
    bagging_mae_list.append(mae_bagging)
    bagging_n_values.append(n_bagging)
    print(f"Bagging model count {n_bagging}, MAE: {mae_bagging:.4f}")

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
plt.title('XGB Bagging Model Optimization (MAE vs Number of Models)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# Optimized Bagging model
optimal_bagging_model = BaggingRegressor(
    estimator=clone(best_model),  # Corrected parameter name
    n_estimators=bagging_n_values[best_bagging_idx],
    random_state=42,
    max_samples=0.8,
    n_jobs=-1
)
optimal_bagging_model.fit(X_train, Y_train)
pre_y_optimal_bagging = optimal_bagging_model.predict(X_test)

# Bagging performance metrics
bagging_metrics = {
    'MAE': mean_absolute_error(pre_y_optimal_bagging, Y_test),
}

# Final results output
print("Optimal Bagging Model Performance Metrics:")
for k, v in bagging_metrics.items():
    print(f"{k}: {v:.4f}")


# Enhanced Bagging visualization
def plot_bagging_analysis(bagging_model, X_test, Y_test, base_pred, bagging_pred):
    """Enhanced visualization for Bagging process"""
    fig = plt.figure(figsize=(20, 15))

    # Subplot 1: Prediction Distribution
    ax1 = fig.add_subplot(2, 3, 1)
    sample_predictions = np.array([estimator.predict(X_test) for estimator in bagging_model.estimators_[:50]])
    sns.violinplot(data=sample_predictions.T, ax=ax1, palette="Set3")
    ax1.plot(base_pred, 'r*', markersize=10, label='Base Model')
    ax1.set_xticks([])
    ax1.set_title('1. Prediction Distribution of Base Learners(XGB)', fontsize=12)
    ax1.set_ylabel('Temperature Prediction')
    ax1.legend()

    # Subplot 2: Error Comparison
    ax2 = fig.add_subplot(2, 3, 2)
    base_errors = base_pred - Y_test
    bagging_errors = bagging_pred - Y_test
    sns.histplot(base_errors, kde=True, color='blue', label='Base Model', ax=ax2, alpha=0.5)
    sns.histplot(bagging_errors, kde=True, color='red', label='Bagging Model', ax=ax2, alpha=0.5)
    ax2.set_title('2. Error Distribution Comparison(XGB)', fontsize=12)
    ax2.legend()

    # Subplot 3: Feature Importance
    ax3 = fig.add_subplot(2, 3, 3)
    base_importance = bagging_model.estimators_[0].feature_importances_
    bagging_importance = np.mean([est.feature_importances_ for est in bagging_model.estimators_], axis=0)
    sns.barplot(x=['X', 'Y', 'Z'], y=base_importance, color='blue', label='Base Model', ax=ax3, alpha=0.7)
    sns.barplot(x=['X', 'Y', 'Z'], y=bagging_importance, color='red', label='Bagging Model', ax=ax3, alpha=0.5)
    ax3.set_title('3. Feature Importance Comparison(XGB)', fontsize=12)
    ax3.legend()

    # Subplot 4: Spatial Error
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    abs_errors = np.abs(bagging_errors)
    sc = ax4.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=abs_errors,
                     cmap='viridis', s=20, alpha=0.7)
    ax4.set_title('4. 3D Spatial Error Distribution(XGB)', fontsize=12)
    fig.colorbar(sc, ax=ax4, label='Absolute Error')

    # Subplot 5: Stability Analysis
    ax5 = fig.add_subplot(2, 3, 5)
    mae_history = [mean_absolute_error(est.predict(X_test), Y_test)
                   for est in bagging_model.estimators_]
    window_size = 10
    rolling_mae = np.convolve(mae_history, np.ones(window_size) / window_size, mode='valid')
    ax5.plot(mae_history, alpha=0.3, label='Single Model MAE')
    ax5.plot(np.arange(window_size - 1, len(mae_history)), rolling_mae,
             color='red', label=f'{window_size}-Model Rolling MAE')
    ax5.set_title('5. Model Stability Analysis(XGB)', fontsize=12)
    ax5.legend()

    # Subplot 6: Prediction vs Actual
    ax6 = fig.add_subplot(2, 3, 6)
    sns.scatterplot(x=Y_test, y=bagging_pred, ax=ax6, alpha=0.6, edgecolor='k')
    lims = [min(Y_test.min(), bagging_pred.min()), max(Y_test.max(), bagging_pred.max())]
    ax6.plot(lims, lims, 'r--', lw=2)
    ax6.set_title('6. Prediction vs Actual Values(XGB)', fontsize=12)
    ax6.set_xlabel('Actual Temperature')
    ax6.set_ylabel('Predicted Temperature')

    plt.tight_layout()
    plt.show()


# Execute enhanced visualization
base_pred = best_model.predict(X_test)
bagging_pred = optimal_bagging_model.predict(X_test)
plot_bagging_analysis(optimal_bagging_model, X_test, Y_test, base_pred, bagging_pred)









###LGB
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import BaggingRegressor
from sklearn.base import clone
import lightgbm as lgb

# Visualization settings
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams['font.sans-serif'] = ['Arial']  # Use Arial font for English text
plt.rcParams['axes.unicode_minus'] = False

# Data loading and preprocessing
data = pd.read_csv(r"仿真数据1.csv", header=None)
data.columns = ['X', 'Y', 'Z', 'Temperature']

# Data cleaning
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna().reset_index(drop=True)

# Base model training and parameter analysis
X = data[['X', 'Y', 'Z']].values
Y = data['Temperature'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

n_values = []
mae_list = []

# LightGBM hyperparameter optimization
for n in range(100, 10001, 200):  # Adjust step size for faster computation
    model = lgb.LGBMRegressor(
        n_estimators=n,
        random_state=42,
        verbosity=-1  # Suppress warning messages
    )
    model.fit(X_train, Y_train)
    pre_y = model.predict(X_test)
    mae_list.append(mean_absolute_error(pre_y, Y_test))
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
plt.show()

# Best base model
best_model = lgb.LGBMRegressor(n_estimators=n_values[best_idx], random_state=42)
best_model.fit(X_train, Y_train)
pre_y = best_model.predict(X_test)

# Model performance evaluation
metrics = {
    'MAE': mean_absolute_error(pre_y, Y_test),
    'MSE': mean_squared_error(pre_y, Y_test),
    'RMSE': np.sqrt(mean_squared_error(pre_y, Y_test)),
    'R²': r2_score(Y_test, pre_y)
}

# Bagging model optimization
bagging_mae_list = []
bagging_n_values = []

# Key modification: Changed base_estimator to estimator
for n_bagging in range(1, 21):
    bagging_model = BaggingRegressor(
        estimator=clone(best_model),  # Corrected parameter name
        n_estimators=n_bagging,
        random_state=42,
        max_samples=0.8,
        n_jobs=-1
    )
    bagging_model.fit(X_train, Y_train)
    pre_y_bagging = bagging_model.predict(X_test)
    mae_bagging = mean_absolute_error(pre_y_bagging, Y_test)
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
    label=f'Optimal Model Count: {bagging_n_values[best_bagging_idx]}\nMAE: {bagging_mae_list[best_bagging_idx]:.4f}'
)
plt.xlabel('Number of Base Models', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.title('Bagging Model Optimization Results', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# Optimal Bagging model
optimal_bagging_model = BaggingRegressor(
    estimator=clone(best_model),  # Corrected parameter name
    n_estimators=bagging_n_values[best_bagging_idx],
    random_state=42,
    max_samples=0.8,
    n_jobs=-1
)
optimal_bagging_model.fit(X_train, Y_train)
pre_y_optimal_bagging = optimal_bagging_model.predict(X_test)

# Bagging model performance evaluation
bagging_metrics = {
    'MAE': mean_absolute_error(pre_y_optimal_bagging, Y_test),
    'MSE': mean_squared_error(pre_y_optimal_bagging, Y_test),
    'RMSE': np.sqrt(mean_squared_error(pre_y_optimal_bagging, Y_test)),
    'R²': r2_score(Y_test, pre_y_optimal_bagging)
}

# Model performance visualization
fig = plt.figure(figsize=(18, 12))
residuals = pre_y - Y_test  # Base model residuals
abs_errors = np.abs(residuals)

# Feature importance
ax1 = fig.add_subplot(2, 3, 1)
sns.barplot(x=best_model.feature_importances_, y=['X', 'Y', 'Z'], ax=ax1, palette='viridis')
ax1.set_title('Feature Importance Analysis(LGB)', fontsize=12)

# Prediction comparison
ax2 = fig.add_subplot(2, 3, 2)
sns.scatterplot(x=Y_test, y=pre_y, ax=ax2, alpha=0.6, color='teal')
lims = [np.min([Y_test.min(), pre_y.min()]), np.max([Y_test.max(), pre_y.max()])]
ax2.plot(lims, lims, 'r--', lw=2)
ax2.set(xlabel='Actual Temperature', ylabel='Predicted Temperature', title='Prediction Comparison Scatter Plot(LGB)')

# Residual distribution
ax3 = fig.add_subplot(2, 3, 3)
sns.histplot(residuals, kde=True, ax=ax3, bins=30, color='purple')
ax3.axvline(0, color='red', linestyle='--')
ax3.set_title('Residual Distribution Histogram(LGB)', fontsize=12)

# Error violin plot
ax4 = fig.add_subplot(2, 3, 4)
sns.violinplot(x=residuals, ax=ax4, inner="quartile", color='orange')
ax4.set_title('Error Distribution Violin Plot(LGB)', fontsize=12)

# Performance radar chart
ax5 = fig.add_subplot(2, 3, 5, polar=True)
categories = list(metrics.keys())
values = list(metrics.values())
values += values[:1]
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
ax5.plot(angles, values, color='darkcyan', linewidth=2)
ax5.fill(angles, values, color='darkcyan', alpha=0.25)
ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories)
ax5.set_title('Performance Metrics Radar Chart(LGB)', fontsize=12, pad=20)

# 3D error distribution
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
sc = ax6.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=abs_errors,
                cmap='plasma', s=20, alpha=0.7)
ax6.set_title('3D Spatial Error Distribution(LGB)', fontsize=12)
fig.colorbar(sc, ax=ax6, label='Absolute Error')

plt.tight_layout()
plt.show()

# Output final results
print("【Base Model Performance(LGB)】")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("\n【Optimal Bagging Model Performance(LGB)】")
for k, v in bagging_metrics.items():
    print(f"{k}: {v:.4f}")






