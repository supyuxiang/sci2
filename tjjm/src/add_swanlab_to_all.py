#!/usr/bin/env python3
"""
为tjjm/src中的所有Python脚本添加SwanLab支持
"""

import os
import re
import glob

def add_swanlab_to_file(file_path):
    """为单个文件添加SwanLab支持"""
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 跳过已经包含swanlab的文件
    if 'from swanlab_config import' in content:
        print(f"跳过 {file_path} - 已包含SwanLab")
        return
    
    # 跳过数据预处理文件
    if 'data_preprocessing.py' in file_path:
        print(f"跳过 {file_path} - 数据预处理文件")
        return
    
    # 跳过配置文件本身
    if 'swanlab_config.py' in file_path:
        print(f"跳过 {file_path} - SwanLab配置文件")
        return
    
    # 跳过这个脚本本身
    if 'add_swanlab_to_all.py' in file_path:
        print(f"跳过 {file_path} - 批量处理脚本")
        return
    
    print(f"处理 {file_path}...")
    
    # 1. 添加SwanLab导入
    import_pattern = r'(from data_preprocessing import data_preprocessing_0)'
    swanlab_import = r'\1\n# 导入SwanLab配置\nfrom swanlab_config import init_swanlab, log_metrics, log_model_params, finish_run'
    
    if 'from data_preprocessing import data_preprocessing_0' in content:
        content = re.sub(import_pattern, swanlab_import, content)
    else:
        # 如果没有data_preprocessing导入，在第一个import后添加
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                if i + 1 < len(lines) and not lines[i + 1].strip().startswith('import ') and not lines[i + 1].strip().startswith('from '):
                    lines.insert(i + 1, '# 导入SwanLab配置')
                    lines.insert(i + 2, 'from swanlab_config import init_swanlab, log_metrics, log_model_params, finish_run')
                    break
        content = '\n'.join(lines)
    
    # 2. 添加SwanLab初始化
    script_name = os.path.basename(file_path)
    
    # 查找合适的位置添加初始化代码
    init_patterns = [
        r'(print\("="\*60\))',
        r'(print\("="\*60\)\s*\nprint\("[^"]*")',
        r'(# 获取预处理后的数据)',
        r'(x_scaled, y_scaled, T_array = data_preprocessing_0\(\))'
    ]
    
    swanlab_init = f'\n# 初始化SwanLab\nrun = init_swanlab("{script_name}")'
    
    init_added = False
    for pattern in init_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, r'\1' + swanlab_init, content)
            init_added = True
            break
    
    if not init_added:
        # 在文件开头添加
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'data_preprocessing_0()' in line:
                lines.insert(i + 1, swanlab_init)
                break
        content = '\n'.join(lines)
    
    # 3. 添加指标记录
    # 查找评估指标计算部分
    metrics_patterns = [
        r'(mae = mean_absolute_error\([^)]+\))',
        r'(rmse = np\.sqrt\(mean_squared_error\([^)]+\))',
        r'(r2 = r2_score\([^)]+\))'
    ]
    
    # 在指标计算后添加SwanLab记录
    metrics_log = '''
# 记录指标到SwanLab
metrics = {
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2
}
log_metrics(run, metrics)'''
    
    # 查找是否已经有指标记录
    if 'log_metrics(run, metrics)' not in content:
        # 在print语句前添加
        print_pattern = r'(print\(f"\\nModel Evaluation Results:")'
        if re.search(print_pattern, content):
            content = re.sub(print_pattern, metrics_log + r'\n\n\1', content)
    
    # 4. 添加模型参数记录
    # 查找模型训练部分
    model_patterns = [
        r'(linear_model\.fit\([^)]+\))',
        r'(rf_model\.fit\([^)]+\))',
        r'(xgb_model\.fit\([^)]+\))',
        r'(lgb_model\.fit\([^)]+\))',
        r'(svr_model\.fit\([^)]+\))',
        r'(model\.fit\([^)]+\))'
    ]
    
    # 在模型训练后添加参数记录
    params_log = '''
# 记录模型参数到SwanLab
try:
    model_params = {
        'model_type': 'linear_regression',
        'n_features': X_train.shape[1],
        'n_samples': X_train.shape[0]
    }
    log_model_params(run, model_params)
except Exception as e:
    print(f"Warning: Could not log model parameters: {e}")'''
    
    # 查找是否已经有参数记录
    if 'log_model_params(run, model_params)' not in content:
        # 在模型训练后添加
        for pattern in model_patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, r'\1' + params_log, content)
                break
    
    # 5. 添加SwanLab完成
    # 在文件末尾添加
    finish_patterns = [
        r'(print\("\\nAnalysis completed!"\))',
        r'(print\("Analysis completed!"\))',
        r'(if __name__ == "__main__":)'
    ]
    
    swanlab_finish = '\n# 完成SwanLab运行\nfinish_run(run)'
    
    finish_added = False
    for pattern in finish_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, r'\1' + swanlab_finish, content)
            finish_added = True
            break
    
    if not finish_added:
        # 在文件末尾添加
        content += '\n\n# 完成SwanLab运行\nfinish_run(run)'
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 完成 {file_path}")

def main():
    """主函数"""
    # 获取所有Python文件
    python_files = glob.glob("*.py")
    
    print(f"找到 {len(python_files)} 个Python文件")
    
    for file_path in python_files:
        add_swanlab_to_file(file_path)
    
    print("\n所有文件处理完成！")

if __name__ == "__main__":
    main()
