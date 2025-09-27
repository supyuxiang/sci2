import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_CARNet_enhance_v2 import EnhancedCARNet_v2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import seaborn as sns
import yaml
from data_preprocessing_0 import data_preprocessing_0
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch.optim as optim
from datetime import datetime
import warnings
import swanlab
import argparse
from utils import Optimizer
from utils import Scheduler
from utils import LossFunction

# 加载YAML配置
def load_config(config_path='config.yaml'):
    with open(config_path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)
# 加载配置和数据
config = load_config()
x_scaled_0, y_scaled_0, T_array_0 = data_preprocessing_0()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() and config['device']['use_cuda'] else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs_0', type=int, default=config['training_0']['epochs'])
parser.add_argument('--batch_size_0', type=int, default=config['training_0']['batch_size'])
parser.add_argument('--learning_rate_0', type=float, default=config['training_0']['learning_rate'])
parser.add_argument('--weight_decay_0', type=float, default=config['training_0']['weight_decay'])
parser.add_argument('--patience_0', type=int, default=config['training_0']['patience'])
parser.add_argument('--device_0', type=str, default=str(device))
parser.add_argument('--use_data_parallel_0', type=bool, default=config['device']['use_data_parallel'])
parser.add_argument('--gpu_device_ids_0', type=str, default=str(config['device']['gpu_device_ids']), 
                   help='GPU device IDs as string, e.g., "8" or "0,1,2"')
parser.add_argument('--early_stopping_patience_0', type=int, default=config['training_0']['early_stopping_patience'])
parser.add_argument('--loss_function_0', type=str, default=config['training_0']['loss_function'])
parser.add_argument('--scheduler_0', type=str, default=config['training_0']['scheduler'])
parser.add_argument('--hidden_size_0', type=int, default=config['model']['hidden_size'])
parser.add_argument('--num_heads_0', type=int, default=config['model']['num_heads'])
parser.add_argument('--dropout_0', type=float, default=config['model']['dropout'])
parser.add_argument('--optimizer_0', type=str, default=config['training_0']['optimizer'])
parser.add_argument('--x_scaled_0', type=np.ndarray, default=x_scaled_0)
parser.add_argument('--y_scaled_0', type=np.ndarray, default=y_scaled_0)
parser.add_argument('--T_array_0', type=np.ndarray, default=T_array_0)
parser.add_argument('--model_0', type=str, default=config['model']['name_0'])

# 调度器参数
parser.add_argument('--T_max_0', type=int, default=100)
parser.add_argument('--T_0_0', type=int, default=10)
parser.add_argument('--T_mult_0', type=int, default=2)
parser.add_argument('--gamma_0', type=float, default=0.95)
parser.add_argument('--step_size_0', type=int, default=30)
parser.add_argument('--milestones_0', type=list, default=[100, 200, 300])
parser.add_argument('--max_lr_0', type=float, default=0.001)
parser.add_argument('--steps_per_epoch_0', type=int, default=100)
parser.add_argument('--pct_start_0', type=float, default=0.3)
parser.add_argument('--anneal_strategy_0', type=str, default='cos')
parser.add_argument('--base_lr_0', type=float, default=0.0001)
parser.add_argument('--start_factor_0', type=float, default=0.1)
parser.add_argument('--end_factor_0', type=float, default=1.0)
parser.add_argument('--total_iters_0', type=int, default=40000)

args = parser.parse_args()

# 处理GPU设备ID参数
def parse_gpu_device_ids(device_ids_str):
    """解析GPU设备ID字符串为整数列表"""
    try:
        # 移除方括号和空格，按逗号分割
        device_ids_str = device_ids_str.strip('[]').replace(' ', '')
        if not device_ids_str:
            return []
        return [int(x) for x in device_ids_str.split(',')]
    except Exception as e:
        print(f"Error parsing GPU device IDs '{device_ids_str}': {e}")
        return []

# 转换GPU设备ID
args.gpu_device_ids_0 = parse_gpu_device_ids(args.gpu_device_ids_0)

# 初始化SwanLab运行
run_0 = swanlab.init(
    name=f'{args.model_0}_Training_0',
    project='sci',
    config={
        "model": "EnhancedCARNet",
        "optimizer": args.optimizer_0,
        "learning_rate": args.learning_rate_0,
        "batch_size": args.batch_size_0,
        "epochs": args.epochs_0,
        "loss_function": args.loss_function_0,
        "scheduler": args.scheduler_0,
        "early_stopping_patience": args.early_stopping_patience_0,
        "device": args.device_0,
        "use_data_parallel": args.use_data_parallel_0,
        "gpu_device_ids": args.gpu_device_ids_0,
        "hidden_size": args.hidden_size_0,
        "num_heads": args.num_heads_0,
        "dropout": args.dropout_0
    }
)

warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 创建结果目录
def plot_training_metrics(history_loss, history_metrics, history_lr, save_path='../results/visualizations/0'):
    """Plot training metrics with learning rate tracking"""
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(15, 12))
    
    # Loss Curve
    plt.subplot(3, 3, 1)
    plt.plot(history_loss, 'r-', linewidth=2, label='Training Loss')
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # R2 Score
    plt.subplot(3, 3, 2)
    plt.plot(history_metrics['r2'], 'b-', linewidth=2, label='R² Score')
    plt.title('R² Score Changes', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Learning Rate Changes
    plt.subplot(3, 3, 3)
    plt.plot(history_lr, 'purple', linewidth=2, label='Learning Rate')
    plt.title('Learning Rate Changes', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Loss vs Learning Rate
    plt.subplot(3, 3, 4)
    plt.scatter(history_lr, history_loss, alpha=0.6, color='red', s=20)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Loss vs Learning Rate', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.subplot(3,3,5)
    plt.plot(history_metrics['mae'],'g-',linewidth=2,label='MAE')
    plt.title('MAE Changes',fontsize=14,fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.grid(True,alpha=0.3)
    plt.legend()

    plt.subplot(3,3,6)
    plt.plot(history_metrics['mse'],'b-',linewidth=2,label='MSE')
    plt.title('MSE Changes',fontsize=14,fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True,alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_metrics_0_{args.optimizer_0}_{args.scheduler_0}_{args.loss_function_0}.png', dpi=1200, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_prediction_analysis(all_predictions, all_targets, epoch, save_path='../results/visualizations/0'):
    """Plot prediction analysis for single variable T"""
    os.makedirs(save_path, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 12))
    
    pred = all_predictions
    true = all_targets
    
    # 散点图
    ax.scatter(true, pred, alpha=0.6, color='red', s=20)
    
    # Ideal line
    min_val = min(min(true), min(pred))
    max_val = max(max(true), max(pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Ideal Line')
    
    # 计算指标
    r2 = r2_score(true, pred)
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    
    ax.set_xlabel('True Values (T)')
    ax.set_ylabel('Predicted Values (T)')
    ax.set_title(f'Temperature T Predictions vs True Values\nR²={r2:.4f}, MSE={mse:.6f}, MAE={mae:.6f}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/prediction_analysis_epoch_{epoch}_0_{args.optimizer_0}_{args.scheduler_0}_{args.loss_function_0}.png', dpi=1200, bbox_inches='tight')
    plt.show()
    plt.close()

def train():
    x_scaled, y_scaled, T = args.x_scaled_0, args.y_scaled_0, args.T_array_0
    if args.model_0 == "EnhancedCARNet":
        from model_CARNet_enhance import EnhancedCARNet
        T_model = EnhancedCARNet(hidden_dim=args.hidden_size_0, num_heads=args.num_heads_0, dropout=args.dropout_0)
        print(f"Using EnhancedCARNet model")
    elif args.model_0 == 'EnhancedCARNet_v2':
        from model_CARNet_enhance_v2 import EnhancedCARNet_v2
        T_model = EnhancedCARNet_v2(hidden_dim=args.hidden_size_0, num_heads=args.num_heads_0, dropout=args.dropout_0)
        print(f"Using EnhancedCARNet_v2 model")
    else:
        raise ValueError(f"Unsupported model: {args.model_0}")
    
    # 多GPU支持：修正设备检查条件
    if (args.use_data_parallel_0 and torch.cuda.is_available() 
        and torch.cuda.device_count() >= len(args.gpu_device_ids_0) 
        and len(args.gpu_device_ids_0) >= 2):
        device_ids = args.gpu_device_ids_0
        # 对于DataParallel，模型应该在第一个设备上
        primary_device = f"cuda:{device_ids[0]}"
        T_model = T_model.to(primary_device)
        print(f"Enabling DataParallel on GPUs: {device_ids}")
        print(f"Model moved to primary device: {primary_device}")
        T_model = nn.DataParallel(T_model, device_ids=device_ids)
        print(f"Model is now wrapped in DataParallel")
    else:
        print(f"Using single GPU or CPU. Device: {args.device_0}")
        T_model = T_model.to(args.device_0)

    optimizer = Optimizer(args.optimizer_0, args.learning_rate_0, args.weight_decay_0)
    optimizer = optimizer.get_optimizer(T_model)
    scheduler = Scheduler(args.scheduler_0, args.learning_rate_0, args.weight_decay_0, args.patience_0, args.T_max_0, args.T_0_0, args.T_mult_0, args.gamma_0, args.step_size_0, args.milestones_0, args.max_lr_0, args.epochs_0, args.steps_per_epoch_0, args.pct_start_0, args.anneal_strategy_0, args.base_lr_0, args.start_factor_0, args.end_factor_0, args.total_iters_0)
    scheduler = scheduler.get_scheduler(optimizer)
    loss_function = LossFunction(args.loss_function_0)
    loss_function = loss_function.get_loss_function()

    # 损失函数：适用于分类任务，而T是连续值
    criterion = loss_function

    train_dataset = TensorDataset(torch.FloatTensor(np.concatenate((args.x_scaled_0, args.y_scaled_0), axis=1)), torch.FloatTensor(args.T_array_0))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_0, shuffle=True)
    
    print("Starting training with adaptive optimizer...")
    print(f"Device: {args.device_0}")
    print(f"Use DataParallel: {args.use_data_parallel_0}")
    print(f"GPU Device IDs: {args.gpu_device_ids_0}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Batch size: {args.batch_size_0}")
    print(f"Epochs: {args.epochs_0}")
    print(f"Initial learning rate: {args.learning_rate_0}")
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Scheduler: {type(scheduler).__name__}")
    print(f"Loss function: {type(loss_function).__name__}")
    print("-" * 50)

    # Initialize history records
    history_loss = []
    history_metrics = {
        'r2': [],
        'mse': [],
        'mae': []
    }
    history_lr = []
    
    best_loss = float('inf')
    patience = args.early_stopping_patience_0 
    patience_counter = 0

    for epoch in range(args.epochs_0):
        if isinstance(T_model, nn.DataParallel):
            T_model.module.train()
        else:
            T_model.train()
        total_loss = 0
        epoch_predictions = []
        epoch_targets = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 确定目标设备
            if (args.use_data_parallel_0 and torch.cuda.is_available() 
                and len(args.gpu_device_ids_0) >= 2):
                target_device = f"cuda:{args.gpu_device_ids_0[0]}"
            else:
                target_device = args.device_0
            
            inputs = inputs.to(target_device)
            targets = targets.to(target_device)

            optimizer.zero_grad()

            outputs = T_model(inputs)
            
            # 处理双输出模型（主输出和辅助输出）
            if isinstance(outputs, tuple):
                main_output , aux_output = outputs
                # 使用主输出计算损失，辅助输出用于深度监督
                main_loss = criterion(main_output, targets)
                aux_loss = criterion(aux_output, targets)
                loss = main_loss + 0.1 * aux_loss  # 辅助损失权重为0.1
                outputs = main_output  # 用于后续处理
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # 对于需要每个batch调用的调度器
            if args.scheduler_0 in ["OneCycleLR", "CyclicLR"]:
                scheduler.step()

            total_loss += loss.item()
            
            # Collect predictions and targets
            epoch_predictions.extend(outputs.squeeze().cpu().detach().numpy())
            epoch_targets.extend(targets.squeeze().cpu().detach().numpy())

            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{args.epochs_0}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, LR: {current_lr:.2e}")

        avg_loss = total_loss / len(train_loader)
        
        # 学习率调度器步进：根据类型调用
        if args.scheduler_0 == "ReduceLROnPlateau":
            scheduler.step(avg_loss)
        elif args.scheduler_0 == "OneCycleLR":
            # OneCycleLR需要在每个batch后调用
            pass  # 在batch循环中调用
        elif args.scheduler_0 == "CyclicLR":
            # CyclicLR需要在每个batch后调用
            pass  # 在batch循环中调用
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate metrics
        avg_r2 = r2_score(epoch_targets, epoch_predictions)
        avg_mse = mean_squared_error(epoch_targets, epoch_predictions)
        avg_mae = mean_absolute_error(epoch_targets, epoch_predictions)

        history_loss.append(avg_loss)
        history_metrics['r2'].append(avg_r2)
        history_metrics['mse'].append(avg_mse)
        history_metrics['mae'].append(avg_mae)
        history_lr.append(current_lr)

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model：确保目录存在
            model_dir = f'../results/best_model_0_{args.optimizer_0}_{args.scheduler_0}_{args.loss_function_0}'
            os.makedirs(model_dir, exist_ok=True)
            if isinstance(T_model, nn.DataParallel):
                state_dict = T_model.module.state_dict()
            else:
                state_dict = T_model.state_dict()
            torch.save({
                'model': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch_': epoch,
                'epoch_loss': best_loss,
                'epoch_learning_rate': current_lr
            }, os.path.join(model_dir, 'best_model.pth'))
        else:
            patience_counter += 1

        # 记录到SwanLab
        swanlab.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "r2_score": avg_r2,
            "mse": avg_mse,
            "mae": avg_mae,
            "learning_rate": current_lr,
            "best_loss": best_loss,
            "patience_counter": patience_counter,
            "is_best_model": avg_loss < best_loss
        })
        
        print(f"Epoch {epoch+1}/{args.epochs_0}, Avg Loss: {avg_loss:.6f}, Avg R2: {avg_r2:.6f}, Avg MSE: {avg_mse:.6f}, Avg MAE: {avg_mae:.6f}")
        print(f"Learning Rate: {current_lr:.2e}, Best Loss: {best_loss:.6f}, Patience: {patience_counter}/{patience}")

        # Regular visualization：修正路径中的参数名
        if (epoch + 1) % 10 == 0:
            save_dir = f'../results/visualizations/0_{args.optimizer_0}_{args.scheduler_0}_{args.loss_function_0}/epoch_{epoch+1}'
            plot_prediction_analysis(epoch_predictions, epoch_targets, epoch + 1, save_dir)
            plot_training_metrics(history_loss, history_metrics, history_lr, save_dir)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered! No improvement for {patience} epochs")
            break

    # Final visualization
    print("Plotting final training results...")
    final_save_dir = f'../results/visualizations/0_{args.optimizer_0}_{args.scheduler_0}_{args.loss_function_0}/epoch_{epoch+1}'
    plot_training_metrics(history_loss, history_metrics, history_lr, final_save_dir)
    
    # Save training history：确保目录存在
    history_dir = f'../results/training_history_0_{args.optimizer_0}_{args.scheduler_0}_{args.loss_function_0}'
    os.makedirs(history_dir, exist_ok=True)
    training_history = {
        'loss': history_loss,
        'metrics': history_metrics,
        'learning_rate': history_lr,
        'best_loss': best_loss,
        'final_epoch': epoch + 1,
        'optimizer_type': type(optimizer).__name__,
        'scheduler_type': type(scheduler).__name__
    }
    np.save(os.path.join(history_dir, 'training_history.npy'), training_history)
    print(f"Training history saved to {history_dir}/training_history.npy")

    # 记录最终结果到SwanLab
    swanlab.log({
        "final_loss": best_loss,
        "final_r2_score": history_metrics['r2'][-1] if history_metrics['r2'] else 0.0,
        "final_mse": history_metrics['mse'][-1] if history_metrics['mse'] else float('inf'),
        "final_mae": history_metrics['mae'][-1] if history_metrics['mae'] else float('inf'),
        "final_learning_rate": current_lr,
        "total_epochs": epoch + 1,
        "training_completed": True
    })
    
    # 完成SwanLab运行
    swanlab.finish()
    
    print("Training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Final epoch: {epoch + 1}")
    print(f"Final learning rate: {current_lr:.2e}")
    
    return history_loss, history_metrics, history_lr

if __name__ == '__main__':
    start_time = datetime.now()
    print(f"Training start time: {start_time}")
    
    history_loss, history_metrics, history_lr = train()
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"Training end time: {end_time}")
    print(f"Total training duration: {training_duration}")
    
    print(f'Final loss: {history_loss[-1]:.6f}')
    print(f'Final learning rate: {history_lr[-1]:.2e}')
    print(f"Training completed! All visualization results saved to ../results/visualizations/0_{args.optimizer_0}_{args.scheduler_0}_{args.loss_function_0}/")
    print(f"Training history saved to ../results/training_history_0_{args.optimizer_0}_{args.scheduler_0}_{args.loss_function_0}/")