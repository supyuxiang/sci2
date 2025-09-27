import swanlab
import os
from datetime import datetime

def init_swanlab(script_name):
    """
    初始化SwanLab配置
    
    Args:
        script_name (str): 脚本名称（不包含.py扩展名）
    """
    # 项目名称固定为 'sci'
    project_name = "sci"
    
    # 实验名称是脚本名去掉.py
    experiment_name = script_name.replace('.py', '')
    
    # 创建swanlab运行实例
    run = swanlab.init(
        project=project_name,
        name=experiment_name,
        description=f"Experiment: {experiment_name}",
        config={
            "script_name": script_name,
            "timestamp": datetime.now().isoformat(),
            "python_version": os.sys.version,
        }
    )
    
    return run

def log_metrics(run, metrics_dict, step=None):
    """
    记录指标到SwanLab
    
    Args:
        run: SwanLab运行实例
        metrics_dict (dict): 要记录的指标字典
        step (int, optional): 步骤数
    """
    if step is not None:
        swanlab.log(metrics_dict, step=step)
    else:
        swanlab.log(metrics_dict)

def log_model_params(run, model_params):
    """
    记录模型参数到SwanLab
    
    Args:
        run: SwanLab运行实例
        model_params (dict): 模型参数字典
    """
    swanlab.log({"model_params": model_params})

def log_training_info(run, training_info):
    """
    记录训练信息到SwanLab
    
    Args:
        run: SwanLab运行实例
        training_info (dict): 训练信息字典
    """
    swanlab.log({"training_info": training_info})

def finish_run(run):
    """
    完成SwanLab运行
    
    Args:
        run: SwanLab运行实例
    """
    swanlab.finish()
