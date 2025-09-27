"""
SwanLab配置文件
用于管理实验参数和记录配置
"""

import torch

# SwanLab实验配置
SWANLAB_CONFIGS = {
    "train_model_main_0_adaptive": {
        "name": "EnhancedCARNet_SingleTask_Training",
        "project": "sci_enhanced_carnet_single",
        "config": {
            "model": "EnhancedCARNet",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "epochs": 200,
            "loss_function": "MSELoss",
            "scheduler": "ReduceLROnPlateau",
            "early_stopping_patience": 10,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "task": "T_prediction",
            "description": "单任务温度预测训练"
        }
    },
    
    "train_model_main_1_base_adaptive": {
        "name": "EnhancedCARNet_MultiTask_Training",
        "project": "sci_enhanced_carnet_multitask",
        "config": {
            "model": "EnhancedCARNet",
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "batch_size": 128,
            "epochs": 200,
            "loss_function": "MSELoss",
            "scheduler": "ReduceLROnPlateau",
            "early_stopping_patience": 15,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "tasks": ["T", "atm", "u", "v"],
            "loss_weights": {
                "T": 1.0,
                "atm": 1.0,
                "u": 1.0,
                "v": 1.0
            },
            "description": "多任务预测训练"
        }
    },
    
    "novel_architectures": {
        "multi_scale_st": {
            "name": "MultiScaleST_Attention_Network",
            "project": "sci_novel_architectures",
            "config": {
                "model": "MultiScaleSTAttentionNet",
                "optimizer": "AdamW",
                "learning_rate": 0.0001,
                "batch_size": 128,
                "epochs": 200,
                "loss_function": "MSELoss",
                "scheduler": "ReduceLROnPlateau",
                "early_stopping_patience": 15,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "tasks": ["T", "atm", "u", "v"],
                "description": "多尺度时空注意力网络"
            }
        },
        
        "physics_aware_gnn": {
            "name": "Physics_Aware_GNN",
            "project": "sci_novel_architectures",
            "config": {
                "model": "PhysicsAwareGNN",
                "optimizer": "AdamW",
                "learning_rate": 0.0001,
                "batch_size": 128,
                "epochs": 200,
                "loss_function": "MSELoss",
                "scheduler": "ReduceLROnPlateau",
                "early_stopping_patience": 15,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "tasks": ["T", "atm", "u", "v"],
                "description": "物理感知图神经网络"
            }
        },
        
        "mixture_of_experts": {
            "name": "Mixture_of_Experts",
            "project": "sci_novel_architectures",
            "config": {
                "model": "HierarchicalMoE",
                "optimizer": "AdamW",
                "learning_rate": 0.0001,
                "batch_size": 128,
                "epochs": 200,
                "loss_function": "MSELoss",
                "scheduler": "ReduceLROnPlateau",
                "early_stopping_patience": 15,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "tasks": ["T", "atm", "u", "v"],
                "description": "混合专家系统"
            }
        }
    }
}

# SwanLab记录配置
SWANLAB_LOGGING_CONFIG = {
    "metrics_to_log": [
        "epoch",
        "loss",
        "learning_rate",
        "best_loss",
        "patience_counter",
        "is_best_model",
        "r2_score",
        "mse",
        "mae",
        "final_loss",
        "final_r2_score",
        "final_mse",
        "final_mae",
        "total_epochs",
        "training_completed"
    ],
    
    "multitask_metrics": [
        "T_r2", "T_mse", "T_mae",
        "atm_r2", "atm_mse", "atm_mae",
        "u_r2", "u_mse", "u_mae",
        "v_r2", "v_mse", "v_mae"
    ],
    
    "log_frequency": {
        "epoch": 1,  # 每个epoch都记录
        "batch": 10,  # 每10个batch记录一次
        "visualization": 10  # 每10个epoch可视化一次
    }
}

def get_swanlab_config(experiment_name):
    """获取指定实验的SwanLab配置"""
    if experiment_name in SWANLAB_CONFIGS:
        return SWANLAB_CONFIGS[experiment_name]
    else:
        # 默认配置
        return {
            "name": f"{experiment_name}_experiment",
            "project": "sci_experiments",
            "config": {
                "model": "Unknown",
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "batch_size": 128,
                "epochs": 200,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        }

def log_training_metrics(swanlab_run, metrics_dict, epoch):
    """记录训练指标到SwanLab"""
    try:
        # 添加epoch信息
        metrics_dict["epoch"] = epoch
        
        # 记录到SwanLab
        swanlab_run.log(metrics_dict)
        
    except Exception as e:
        print(f"Warning: Failed to log metrics to SwanLab: {e}")

def log_final_results(swanlab_run, final_metrics):
    """记录最终结果到SwanLab"""
    try:
        # 添加完成标记
        final_metrics["training_completed"] = True
        
        # 记录最终结果
        swanlab_run.log(final_metrics)
        
        # 完成运行
        swanlab_run.finish()
        
    except Exception as e:
        print(f"Warning: Failed to log final results to SwanLab: {e}")
