# cox_train.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
import os
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

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

def train_cox():
    # 初始化SwanLab
    run = init_swanlab("cox_model.py")
    
    print("="*60)
    print("Cox Proportional Hazards Model Training Analysis")
    print("="*60)
    
    # 创建结果目录
    results_dir = "../results/cox"
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取预处理后的数据
    x_scaled, y_scaled, T_array = data_preprocessing_0()

    # lifelines 需要 DataFrame，准备时间与事件列（示例中全部事件=1）
    time = np.asarray(T_array).ravel()
    event = np.ones_like(time, dtype=int)

    df = pd.DataFrame({
        'time': time,
        'event': event,
        'x_scale': np.asarray(x_scaled).ravel(),
        'y_scale': np.asarray(y_scaled).ravel()
    })

    # 数据分割
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    # 模型训练（lifelines）
    print("\nStarting Cox model training (lifelines)...")
    cph = CoxPHFitter()
    cph.fit(train_df, duration_col='time', event_col='event')

    # 评估：基于测试集的 C-index
    partial_hazards = cph.predict_partial_hazard(test_df)
    c_index = concordance_index(test_df['time'], -partial_hazards.values.ravel(), test_df['event'])
    print(f"\nConcordance Index: {c_index:.4f}")

    # 记录模型参数和指标到SwanLab
    model_params = {
        'model_type': 'cox_proportional_hazards',
        'n_features': 2,
        'n_samples': len(train_df),
        'test_samples': len(test_df),
        'duration_col': 'time',
        'event_col': 'event'
    }
    log_model_params(run, model_params)
    
    metrics = {
        'Concordance_Index': c_index
    }
    log_metrics(run, metrics)

    # 可视化特征重要性
    feature_names = ['x_scale', 'y_scale']
    coefficients = cph.params_[feature_names].values
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, coefficients, color=color_palette[:2])
    plt.title('Cox Model Feature Coefficients (lifelines)', fontsize=16, fontweight='bold')
    plt.ylabel('Coefficient Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'feature_coefficients.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # 保存模型结果
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    coef_df.to_csv(os.path.join(results_dir, 'cox_coefficients.csv'), index=False)
    
    print(f"\nResults saved to: {results_dir}")
    print("Cox model training completed (lifelines)!")
    
    # 完成SwanLab运行
    finish_run(run)

if __name__ == "__main__":
    train_cox()