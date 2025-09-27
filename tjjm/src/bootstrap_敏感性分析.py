import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
import os
from tqdm import tqdm

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

# 创建结果目录
results_dir = "../results/bootstrap_sensitivity"
os.makedirs(results_dir, exist_ok=True)

# 定义评估模型的函数
def evaluate_model(y_true, y_pred, model_name="Model"):
    """评估模型性能"""
    # 确保输入为一维数组
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# Bootstrap分析函数
def bootstrap_analysis(model, X, y, n_iterations=100, test_size=0.2):
    """执行Bootstrap分析"""
    bootstrap_metrics = {'MAE': [], 'MSE': [], 'RMSE': [], 'R2': []}
    
    print(f"Starting Bootstrap analysis with {n_iterations} iterations...")
    
    for i in tqdm(range(n_iterations), desc="Bootstrap Progress"):
        try:
            # 重采样数据
            X_resample, y_resample = resample(X, y, random_state=i, n_samples=int(len(X) * 0.8))
            
            # 分割训练测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_resample, y_resample, test_size=test_size, random_state=i
            )
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 评估
            metrics = evaluate_model(y_test, y_pred, f"Bootstrap {i + 1}")
            
            # 存储结果
            for metric in bootstrap_metrics:
                bootstrap_metrics[metric].append(metrics[metric])
                
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            continue
    
    return bootstrap_metrics

def main():
    # 初始化SwanLab
    run = init_swanlab("bootstrap_敏感性分析.py")
    
    print("="*60)
    print("Bootstrap Sensitivity Analysis")
    print("="*60)
    
    # 获取预处理后的数据
    x_scaled, y_scaled, T_array = data_preprocessing_0()
    
    # 数据准备
    X = np.column_stack([x_scaled, y_scaled])
    y = T_array
    
    print(f"Feature matrix X shape: {X.shape}")
    print(f"Target variable y shape: {y.shape}")
    
    # 初始化模型
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }
    
    # 执行Bootstrap分析
    bootstrap_results = {}
    for name, model in models.items():
        print(f"\nAnalyzing {name}...")
        bootstrap_results[name] = bootstrap_analysis(model, X, y, n_iterations=100)
    
    # 可视化Bootstrap结果
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bootstrap Sensitivity Analysis Results', fontsize=16, fontweight='bold')
    
    # MAE分布
    mae_data = [bootstrap_results[name]['MAE'] for name in models.keys()]
    axes[0, 0].boxplot(mae_data, labels=list(models.keys()))
    axes[0, 0].set_title('MAE Distribution Across Models', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE分布
    rmse_data = [bootstrap_results[name]['RMSE'] for name in models.keys()]
    axes[0, 1].boxplot(rmse_data, labels=list(models.keys()))
    axes[0, 1].set_title('RMSE Distribution Across Models', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Root Mean Squared Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # R²分布
    r2_data = [bootstrap_results[name]['R2'] for name in models.keys()]
    axes[1, 0].boxplot(r2_data, labels=list(models.keys()))
    axes[1, 0].set_title('R² Distribution Across Models', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 小提琴图比较
    violin_data = []
    violin_labels = []
    for name in models.keys():
        violin_data.extend(bootstrap_results[name]['R2'])
        violin_labels.extend([name] * len(bootstrap_results[name]['R2']))
    
    violin_df = pd.DataFrame({'Model': violin_labels, 'R2': violin_data})
    sns.violinplot(data=violin_df, x='Model', y='R2', ax=axes[1, 1])
    axes[1, 1].set_title('R² Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'bootstrap_sensitivity_analysis.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # 统计摘要
    print("\n" + "="*60)
    print("Bootstrap Analysis Summary")
    print("="*60)
    
    for name in models.keys():
        print(f"\n{name} Results:")
        for metric in ['MAE', 'RMSE', 'R2']:
            values = bootstrap_results[name][metric]
            print(f"  {metric}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}, "
                  f"95% CI=[{np.percentile(values, 2.5):.4f}, {np.percentile(values, 97.5):.4f}]")
    
    # 学习曲线分析
    print("\n" + "="*60)
    print("Learning Curve Analysis")
    print("="*60)
    
    # 使用Stacking模型进行学习曲线分析
    stacking_model = StackingRegressor(
        estimators=[(name, model) for name, model in models.items()],
        final_estimator=LinearRegression(),
        n_jobs=-1
    )
    
    # 分割数据用于学习曲线
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 计算学习曲线
    train_sizes, train_scores, test_scores = learning_curve(
        stacking_model, X_train, y_train, cv=5,
        scoring='neg_mean_squared_error', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    # 绘制学习曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # MSE学习曲线
    ax1.plot(train_sizes, -train_scores.mean(axis=1), 'o-', color='blue', label='Training Error')
    ax1.fill_between(train_sizes, -train_scores.mean(axis=1) - train_scores.std(axis=1),
                     -train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='blue')
    ax1.plot(train_sizes, -test_scores.mean(axis=1), 'o-', color='red', label='Validation Error')
    ax1.fill_between(train_sizes, -test_scores.mean(axis=1) - test_scores.std(axis=1),
                     -test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.1, color='red')
    ax1.set_title('Learning Curve (MSE)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Training Set Size')
    ax1.set_ylabel('Mean Squared Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 训练时间分析
    train_times = []
    for size in train_sizes:
        start_time = time.time()
        stacking_model.fit(X_train[:int(size*len(X_train))], y_train[:int(size*len(y_train))])
        train_times.append(time.time() - start_time)
    
    ax2.plot(train_sizes, train_times, 'o-', color='green')
    ax2.set_title('Training Time vs Dataset Size', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Training Set Size')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'learning_curves.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # 保存结果
    results_summary = {}
    for name in models.keys():
        results_summary[name] = {
            'MAE_mean': np.mean(bootstrap_results[name]['MAE']),
            'MAE_std': np.std(bootstrap_results[name]['MAE']),
            'RMSE_mean': np.mean(bootstrap_results[name]['RMSE']),
            'RMSE_std': np.std(bootstrap_results[name]['RMSE']),
            'R2_mean': np.mean(bootstrap_results[name]['R2']),
            'R2_std': np.std(bootstrap_results[name]['R2'])
        }
    
    results_df = pd.DataFrame(results_summary).T
    results_df.to_csv(os.path.join(results_dir, 'bootstrap_results_summary.csv'))
    
    print(f"\nResults saved to: {results_dir}")
    print("Bootstrap sensitivity analysis completed!")
    
    # 完成SwanLab运行
    finish_run(run)

if __name__ == "__main__":
    import time
    main()








