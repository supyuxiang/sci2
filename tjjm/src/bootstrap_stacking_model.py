import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import learning_curve

# 设置可视化风格
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用Arial字体显示英文
plt.rcParams['axes.unicode_minus'] = False

# 数据加载与预处理
data = pd.read_csv("仿真数据1.csv", header=None)
data.columns = ['X', 'Y', 'Z', 'Temperature']
data = data.apply(pd.to_numeric, errors='coerce').dropna()

# 参数优化（基础模型）
X = data[['X', 'Y', 'Z']].values
Y = data['Temperature'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 基础模型
base_rf = RandomForestRegressor(n_estimators=1000, random_state=42)
base_xgb = xgb.XGBRegressor(n_estimators=10000, random_state=42, n_jobs=-1)
base_lgb = lgb.LGBMRegressor(n_estimators=10000, random_state=42, n_jobs=-1)

# 使用StackingRegressor，元学习器采用线性回归模型
stacking_model = StackingRegressor(
    estimators=[('rf', base_rf), ('xgb', base_xgb), ('lgb', base_lgb)],
    final_estimator=LinearRegression()
)

# 模型拟合
base_rf.fit(X_train, Y_train)
base_xgb.fit(X_train, Y_train)
base_lgb.fit(X_train, Y_train)
stacking_model.fit(X_train, Y_train)

# 计算评估指标
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# 4. 为三个基模型添加bootstrap敏感性分析
def bootstrap_analysis(model, X, Y, n_iterations=1000):
    bootstrap_metrics = {
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'R2': []
    }
    for i in range(n_iterations):
        # Bootstrap抽样
        X_resample, Y_resample = resample(X, Y, random_state=i)
        y_pred_bootstrap = model.predict(X_resample)
        # 预测并评估
        metrics = evaluate_model(Y_resample, y_pred_bootstrap, f"Bootstrap {i + 1}")
        # 存储结果
        bootstrap_metrics['MAE'].append(metrics['MAE'])
        bootstrap_metrics['MSE'].append(metrics['MSE'])
        bootstrap_metrics['RMSE'].append(metrics['RMSE'])
        bootstrap_metrics['R2'].append(metrics['R2'])
    return bootstrap_metrics

# 进行Bootstrap分析
stacking_model_bootstrap_metrics = bootstrap_analysis(stacking_model,X,Y)
rf_bootstrap_metrics = bootstrap_analysis(base_rf, X, Y)
xgb_bootstrap_metrics = bootstrap_analysis(base_xgb, X, Y)
lgb_bootstrap_metrics = bootstrap_analysis(base_lgb, X, Y)

print('stacking_model_bootstrap_metrics:',stacking_model_bootstrap_metrics)

# 5. 可视化bootstrap结果（基模型）
# RandomForest MAE Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=rf_bootstrap_metrics['MAE'])
plt.title('RandomForest Bootstrap MAE Distribution')
plt.ylabel('Mean Absolute Error')
plt.show()

# RandomForest MSE Violinplot
plt.figure(figsize=(8, 6))
sns.violinplot(data=rf_bootstrap_metrics['MSE'])
plt.title('RandomForest Bootstrap MSE Distribution')
plt.ylabel('Mean Squared Error')
plt.show()

# XGBoost MAE Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=xgb_bootstrap_metrics['MAE'])
plt.title('XGBoost Bootstrap MAE Distribution')
plt.ylabel('Mean Absolute Error')
plt.show()

# XGBoost MSE Violinplot
plt.figure(figsize=(8, 6))
sns.violinplot(data=xgb_bootstrap_metrics['MSE'])
plt.title('XGBoost Bootstrap MSE Distribution')
plt.ylabel('Mean Squared Error')
plt.show()

# LightGBM MAE Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=lgb_bootstrap_metrics['MAE'])
plt.title('LightGBM Bootstrap MAE Distribution')
plt.ylabel('Mean Absolute Error')
plt.show()

# LightGBM MSE Violinplot
plt.figure(figsize=(8, 6))
sns.violinplot(data=lgb_bootstrap_metrics['MSE'])
plt.title('LightGBM Bootstrap MSE Distribution')
plt.ylabel('Mean Squared Error')
plt.show()



# 1. 整合所有模型的bootstrap分析结果
def plot_bootstrap_comparison(bootstrap_metrics_dict, metric_name, title):
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=[bootstrap_metrics_dict[model][metric_name] for model in bootstrap_metrics_dict],
                labels=[model for model in bootstrap_metrics_dict])
    plt.title(title)
    plt.ylabel(metric_name)
    plt.show()

# 2. 绘制所有模型的MAE分布对比
plot_bootstrap_comparison(
    {'RandomForest': rf_bootstrap_metrics, 'XGBoost': xgb_bootstrap_metrics, 'LightGBM': lgb_bootstrap_metrics, 'Stacking': stacking_model_bootstrap_metrics},
    'MAE',
    'Bootstrap MAE Distribution for All Models'
)

# 3. 绘制所有模型的MSE分布对比
plot_bootstrap_comparison(
    {'RandomForest': rf_bootstrap_metrics, 'XGBoost': xgb_bootstrap_metrics, 'LightGBM': lgb_bootstrap_metrics, 'Stacking': stacking_model_bootstrap_metrics},
    'MSE',
    'Bootstrap MSE Distribution for All Models'
)

# 4. 绘制所有模型的RMSE分布对比
plot_bootstrap_comparison(
    {'RandomForest': rf_bootstrap_metrics, 'XGBoost': xgb_bootstrap_metrics, 'LightGBM': lgb_bootstrap_metrics, 'Stacking': stacking_model_bootstrap_metrics},
    'RMSE',
    'Bootstrap RMSE Distribution for All Models'
)

# 5. 绘制所有模型的R²分布对比
plot_bootstrap_comparison(
    {'RandomForest': rf_bootstrap_metrics, 'XGBoost': xgb_bootstrap_metrics, 'LightGBM': lgb_bootstrap_metrics, 'Stacking': stacking_model_bootstrap_metrics},
    'R2',
    'Bootstrap R² Distribution for All Models'
)





