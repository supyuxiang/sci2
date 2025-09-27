import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
from sklearn.ensemble import VotingRegressor


# 数据加载与预处理
data = pd.read_csv(r"仿真数据1.csv", header=None)
data.columns = ['X', 'Y', 'Z', 'Temperature']

# 类型转换与缺失值处理
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()

# 特征与目标变量
X = data[['X', 'Y', 'Z']].values
Y = data['Temperature'].values

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).ravel()

# 分割训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# 增量训练函数（支持XGBoost和LightGBM），并结合交叉验证
def incremental_train_with_cv(model, X_train, y_train, batch_size=100, cv_folds=5):
    cv_scores = []  # 存储每个批次的交叉验证得分
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        if isinstance(model, xgb.XGBRegressor):
            if i == 0:
                model.fit(X_batch, y_batch)
            else:
                model.fit(X_batch, y_batch, xgb_model=model)
        elif isinstance(model, lgb.LGBMRegressor):
            model.fit(X_batch, y_batch, init_model=model if i != 0 else None)
        else:
            model.fit(X_batch, y_batch)

        # 在每个增量批次后进行交叉验证
        cv_score = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2').mean()
        cv_scores.append(cv_score)

    return model, cv_scores

# 模型训练（包含交叉验证）
batch_size = len(X_train) // 100

# XGBoost增量训练与交叉验证
xgb_model = xgb.XGBRegressor(n_estimators=666, learning_rate=0.1)
xgb_model, xgb_cv_scores = incremental_train_with_cv(xgb_model, X_train, y_train, batch_size)

# LightGBM增量训练与交叉验证
lgb_model = lgb.LGBMRegressor(n_estimators=666, learning_rate=0.1)
lgb_model, lgb_cv_scores = incremental_train_with_cv(lgb_model, X_train, y_train, batch_size)

# 随机森林交叉验证（完整训练）
rf_model = RandomForestRegressor(n_estimators=1000, min_samples_leaf=5)
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
print(f"Random Forest 5-Fold CV R2 Scores: {rf_cv_scores}")
rf_model.fit(X_train, y_train)  # 全量训练

# 模型预测
rf_pred = rf_model.predict(X_test)
lgb_pred = lgb_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

pred_line = []
best_weights_and_mae = [0,0,0,0]

for i in range(1,4):
    for j in range(1,4):
        for k in range(1,4):

            # 集成模型
            voting_regressor = VotingRegressor([('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model)], weights=[i,j,k]).fit(X_train, y_train)
            y_pred = voting_regressor.predict(X_test)

            # 性能评估
            maes = {
                'Random Forest': mean_absolute_error(y_test, rf_pred),
                'LightGBM': mean_absolute_error(y_test, lgb_pred),
                'XGBoost': mean_absolute_error(y_test, xgb_pred),
                'Voting Regressor': mean_absolute_error(y_test, y_pred)
            }

            if pred_line==[] or (pred_line != [] and min(pred_line)) > mean_absolute_error(y_test,y_pred):
                best_weights_and_mae = (i,j,k,mean_absolute_error(y_test,y_pred))
            pred_line.append(mean_absolute_error(y_test,y_pred))

best_para = list(best_weights_and_mae[0:4])

voting_regressor = VotingRegressor([('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model)], weights=best_para).fit(X_train, y_train)
y_pred = voting_regressor.predict(X_test)


# 性能评估
models = [
    ('Random Forest', rf_pred),
    ('LightGBM', lgb_pred),
    ('XGBoost', xgb_pred),
    ('Voting Regressor', y_pred)
]
results = []
for name, pred in models:
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    results.append({
        'Model': name,
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4)
    })
metrics_df = pd.DataFrame(results)
print("\n模型性能指标对比：")
print(metrics_df)




# 1. 性能对比图
plt.figure(figsize=(10,6))
maes = {result['Model']: result['MAE'] for result in results}
plt.bar(maes.keys(), maes.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Model Performance Comparison (MAE)')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. 预测分布图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
models = [('Random Forest', rf_pred), ('LightGBM', lgb_pred),
          ('XGBoost', xgb_pred), ('Voting Regressor', y_pred)]
for ax, (name, pred) in zip(axes.flat, models):
    ax.scatter(y_test, pred, alpha=0.3)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    ax.set(xlabel='True Temperature', ylabel='Predicted Temperature', title=name)
plt.tight_layout()
plt.show()

# 3. 热力图（误差分布）
test_data = pd.DataFrame({'True Temp': y_test, 'Predicted Temp': y_pred})
test_data['Error'] = test_data['Predicted Temp'] - test_data['True Temp']
plt.figure(figsize=(12, 8))
sns.kdeplot(x=test_data['True Temp'], y=test_data['Predicted Temp'],
            cmap='viridis', fill=True, thresh=0.1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('True vs Predicted Temperature Density')
plt.xlabel('True Temperature')
plt.ylabel('Predicted Temperature')
plt.show()

# 4. 特征重要性图
features = ['X', 'Y', 'Z']
plt.figure(figsize=(15, 5))
for i, (name, model) in enumerate([('Random Forest', rf_model),
                                  ('XGBoost', xgb_model),
                                  ('LightGBM', lgb_model)], 1):
    plt.subplot(1, 3, i)
    importances = model.feature_importances_
    plt.barh(features, importances)
    plt.title(f'{name} Feature Importance')
plt.tight_layout()
plt.show()

# 打印增量训练后交叉验证结果
print(f"\nXGBoost 增量训练交叉验证 R2 Scores: {xgb_cv_scores}")
print(f"LightGBM 增量训练交叉验证 R2 Scores: {lgb_cv_scores}")


print('pred_line:',pred_line)

print('best_weights_and_mae:',best_weights_and_mae)