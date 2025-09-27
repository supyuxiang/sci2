# rulefit_train.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
import os
from rulefit import RuleFit

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

def train_rulefit():
    # 初始化SwanLab
    run = init_swanlab("RuleFit.py")
    
    print("="*60)
    print("RuleFit Model Training Analysis")
    print("="*60)
    
    # 创建结果目录
    results_dir = "../results/rulefit"
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
    
    # 模型训练
    print("\nStarting RuleFit model training...")
    rulefit = RuleFit(
        rfmode='regress',  # 回归模式
        max_rules=20,      # 最大规则数
        random_state=42
    )
    
    rulefit.fit(X_train, y_train, feature_names=['x_scale', 'y_scale'])
    
    # 预测
    y_pred = rulefit.predict(X_test)
    
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
        'rfmode': rulefit.rfmode,
        'max_rules': rulefit.max_rules,
        'random_state': rulefit.random_state,
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
    
    # 获取规则重要性
    rules = rulefit.get_rules()
    rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
    
    # 可视化最重要的规则
    plt.figure(figsize=(12, 8))
    top_rules = rules.head(10)
    plt.barh(range(len(top_rules)), top_rules['importance'].values, color=color_palette)
    plt.yticks(range(len(top_rules)), top_rules['rule'].values, fontsize=10)
    plt.title('Top 10 Rule Importance', fontsize=16, fontweight='bold')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'rule_importance.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # 保存规则
    rules.to_csv(os.path.join(results_dir, 'rules.csv'), index=False)
    
    print(f"\nResults saved to: {results_dir}")
    print("RuleFit model training completed!")
    
    # 完成SwanLab运行
    finish_run(run)

if __name__ == "__main__":
    train_rulefit()