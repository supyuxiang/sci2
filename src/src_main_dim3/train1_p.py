import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import seaborn as sns
from config import config_train
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch.optim as optim
from datetime import datetime
import warnings
import swanlab
import argparse
from tqdm import tqdm

# 添加argparse参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--epochs_1', type=int, default=config_train.epochs_1)
parser.add_argument('--batch_size_1', type=int, default=config_train.batch_size_1)
parser.add_argument('--learning_rate_1', type=float, default=config_train.learning_rate_1)
parser.add_argument('--weight_decay_1', type=float, default=config_train.weight_decay_1)
parser.add_argument('--patience_1', type=int, default=config_train.patience_1)
parser.add_argument('--device_1', type=str, default=config_train.device)
parser.add_argument('--use_data_parallel_1', type=bool, default=config_train.use_data_parallel)
parser.add_argument('--gpu_device_ids_1', type=str, default=str(config_train.gpu_device_ids), 
                   help='GPU device IDs as string, e.g., "8" or "0,1,2"')
parser.add_argument('--early_stopping_patience_1', type=int, default=config_train.early_stopping_patience_1)
parser.add_argument('--loss_function_1', type=str, default=config_train.loss_function_1)
parser.add_argument('--scheduler_1', type=str, default=config_train.scheduler_1)
parser.add_argument('--hidden_size_1', type=int, default=config_train.hidden_size)
parser.add_argument('--num_heads_1', type=int, default=config_train.num_heads)
parser.add_argument('--dropout_1', type=float, default=config_train.dropout)
parser.add_argument('--optimizer_1', type=str, default=config_train.optimizer_1)
parser.add_argument('--x_scaled_1', type=np.ndarray, default=config_train.x_scaled_1)
parser.add_argument('--y_scaled_1', type=np.ndarray, default=config_train.y_scaled_1)
parser.add_argument('--T_1', type=np.ndarray, default=config_train.T_1)
parser.add_argument('--p_1', type=np.ndarray, default=config_train.p_1)
parser.add_argument('--u_1', type=np.ndarray, default=config_train.u_1)
parser.add_argument('--v_1', type=np.ndarray, default=config_train.v_1)
parser.add_argument('--loss_T_weight_1', type=float, default=config_train.loss_T_weight)
parser.add_argument('--loss_atm_weight_1', type=float, default=config_train.loss_atm_weight)
parser.add_argument('--loss_u_weight_1', type=float, default=config_train.loss_u_weight)
parser.add_argument('--loss_v_weight_1', type=float, default=config_train.loss_v_weight)
parser.add_argument('--model_1', type=str, default=config_train.model_1)
# 添加散度损失权重参数
parser.add_argument('--divergence_loss_weight', type=float, default=config_train.divergence_loss_weight, help='Weight for divergence loss term')

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
args.gpu_device_ids_1 = parse_gpu_device_ids(args.gpu_device_ids_1)

# 初始化SwanLab运行
run_1 = swanlab.init(
    name=f'{args.model_1}_MultiTask_Adaptive_Training_1_with_Divergence',
    project='sci',
    config={
        "model": args.model_1,
        "optimizer": args.optimizer_1,
        "learning_rate": args.learning_rate_1,
        "batch_size": args.batch_size_1,
        "epochs": args.epochs_1,
        "loss_function": args.loss_function_1,
        "scheduler": args.scheduler_1,
        "early_stopping_patience": args.early_stopping_patience_1,
        "device": args.device_1,
        "tasks": ["T", "p", "u", "v"],
        "loss_weights": {
            "T": args.loss_T_weight_1,
            "p": args.loss_atm_weight_1,
            "u": args.loss_u_weight_1,
            "v": args.loss_v_weight_1
        },
        "use_data_parallel": args.use_data_parallel_1,
        "gpu_device_ids": args.gpu_device_ids_1,
        "hidden_size": args.hidden_size_1,
        "num_heads": args.num_heads_1,
        "dropout": args.dropout_1,
        "divergence_loss_weight": args.divergence_loss_weight  # 添加散度损失权重
    }
)

# 导入自适应优化器
try:
    from torch.optim import AdamW
except ImportError:
    print("AdamW not available, using Adam instead")

try:
    from adabelief_pytorch import AdaBelief
except ImportError:
    print("AdaBelief not available, install with: pip install adabelief-pytorch")

try:
    from torch_optimizer import RAdam, AdaBound, AdamP
except ImportError:
    print("torch_optimizer not available, install with: pip install torch-optimizer")

warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 创建结果目录
os.makedirs('../results', exist_ok=True)
os.makedirs('../results/visualizations', exist_ok=True)

# OOP wrapper aligned with Trainer1 style
class Trainer1P:
    def __init__(self):
        # 使用已存在的解析结果，避免大范围改动
        self.args = args
        self.run_1 = run_1

    def train(self):
        return train()

# 可视化函数
def plot_training_metrics(history_loss, history_metrics, history_lr, save_path='../results/visualizations/1'):
    """Plot training metrics with learning rate tracking"""
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Loss Curve
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 3, 1)
    plt.plot(history_loss, 'r-', linewidth=2, label='Training Loss')
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. R2 Score
    plt.subplot(3, 3, 2)
    for i, (name, values) in enumerate(history_metrics['r2'].items()):
        plt.plot(values, label=name, linewidth=2)
    plt.title('R² Score Changes', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. MSE
    plt.subplot(3, 3, 3)
    for i, (name, values) in enumerate(history_metrics['mse'].items()):
        plt.plot(values, label=name, linewidth=2)
    plt.title('MSE Changes', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 4. MAE
    plt.subplot(3, 3, 4)
    for i, (name, values) in enumerate(history_metrics['mae'].items()):
        plt.plot(values, label=name, linewidth=2)
    plt.title('MAE Changes', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 5. Loss Analysis (Log Scale)
    plt.subplot(3, 3, 5)
    plt.plot(history_loss, 'r-', linewidth=2, label='Total Loss')
    plt.title('Loss Trend Analysis (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 6. Learning Rate Tracking
    plt.subplot(3, 3, 6)
    plt.plot(history_lr, 'b-', linewidth=2, label='Learning Rate')
    plt.title('Learning Rate Changes', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 7. Loss vs Learning Rate
    plt.subplot(3, 3, 7)
    plt.scatter(history_lr, history_loss, alpha=0.6, c=range(len(history_loss)), cmap='viridis')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Loss vs Learning Rate', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Epoch')
    
    # 8. Learning Rate Adjustment
    plt.subplot(3, 3, 8)
    epochs = range(len(history_loss))
    plt.plot(epochs, history_loss, 'b-', linewidth=2, label='Current Loss')
    plt.title('Learning Rate Adjustment', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 9. Convergence Analysis
    plt.subplot(3, 3, 9)
    if len(history_loss) > 1:
        loss_diff = np.diff(history_loss)
        plt.plot(loss_diff, 'g-', linewidth=2, label='Loss Difference')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Loss Convergence Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_metrics_1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_prediction_analysis(all_predictions, all_targets, epoch, save_path='../results/visualizations/1'):
    """Plot prediction analysis"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Epoch {epoch} - Prediction Analysis', fontsize=16, fontweight='bold')
    
    variables = ['T', 'p', 'u', 'v']
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (var, color) in enumerate(zip(variables, colors)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        pred = all_predictions[var]
        true = all_targets[var]
        
        # 散点图
        ax.scatter(true, pred, alpha=0.6, color=color, s=20)
        
        # Ideal line
        min_val = min(min(true), min(pred))
        max_val = max(max(true), max(pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Ideal Line')
        
        # 计算指标
        r2 = r2_score(true, pred)
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        
        ax.set_xlabel(f'True Values ({var})')
        ax.set_ylabel(f'Predicted Values ({var})')
        ax.set_title(f'{var} Predictions vs True Values\nR²={r2:.4f}, MSE={mse:.6f}, MAE={mae:.6f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/prediction_analysis_epoch_{epoch}_1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_data_distribution(x_scaled, y_scaled, T, p, u, v, save_path='../results/visualizations/1'):
    """Plot data distribution"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Input feature distributions
    axes[0, 0].hist(x_scaled.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('X Coordinate Distribution')
    axes[0, 0].set_xlabel('X Coordinate')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(y_scaled.flatten(), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Y Coordinate Distribution')
    axes[0, 1].set_xlabel('Y Coordinate')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Target variable distributions
    axes[0, 2].hist(T.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 2].set_title('Temperature T Distribution')
    axes[0, 2].set_xlabel('Temperature')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].hist(p.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Pressure p Distribution')
    axes[1, 0].set_xlabel('Pressure p')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(u.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Velocity u Distribution')
    axes[1, 1].set_xlabel('Velocity u')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(v.flatten(), bins=50, alpha=0.7, color='brown', edgecolor='black')
    axes[1, 2].set_title('Velocity v Distribution')
    axes[1, 2].set_xlabel('Velocity v')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/data_distribution_1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_correlation_matrix(x_scaled, y_scaled, T, p, u, v, save_path='../results/visualizations/1'):
    """Plot correlation matrix"""
    os.makedirs(save_path, exist_ok=True)
    
    # Create data frame
    data = pd.DataFrame({
        'x': x_scaled.flatten(),
        'y': y_scaled.flatten(),
        'T': T.flatten(),
        'p': p.flatten(),
        'u': u.flatten(),
        'v': v.flatten()
    })
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Variable Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}/correlation_matrix_1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_model_architecture(save_path='../results/visualizations/1'):
    """Plot model architecture"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define component positions
    components = {
        'Input': (1, 8),
        'T3_model': (3, 6),
        'atm_model': (3, 4),
        'v_model': (3, 2),
        'u_model': (3, 0),
        'Loss Calculation': (6, 3),
        'Adaptive Optimizer': (9, 3)
    }
    
    # Draw components
    for name, pos in components.items():
        if name == 'Input':
            ax.add_patch(plt.Rectangle((pos[0]-0.5, pos[1]-0.3), 1, 0.6, 
                                     facecolor='lightblue', edgecolor='black', linewidth=2))
        elif 'model' in name:
            ax.add_patch(plt.Rectangle((pos[0]-0.5, pos[1]-0.3), 1, 0.6, 
                                     facecolor='lightgreen', edgecolor='black', linewidth=2))
        elif 'Optimizer' in name:
            ax.add_patch(plt.Rectangle((pos[0]-0.5, pos[1]-0.3), 1, 0.6, 
                                     facecolor='lightyellow', edgecolor='black', linewidth=2))
        else:
            ax.add_patch(plt.Rectangle((pos[0]-0.5, pos[1]-0.3), 1, 0.6, 
                                     facecolor='lightcoral', edgecolor='black', linewidth=2))
        
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontweight='bold')
    
    # Draw connections
    connections = [
        ((1, 8), (3, 6)),  # Input to T3_model
        ((1, 8), (3, 4)),  # Input to atm_model
        ((1, 8), (3, 2)),  # Input to v_model
        ((1, 8), (3, 0)),  # Input to u_model
        ((3, 6), (6, 3)),  # T3_model to Loss
        ((3, 4), (6, 3)),  # atm_model to Loss
        ((3, 2), (6, 3)),  # v_model to Loss
        ((3, 0), (6, 3)),  # u_model to Loss
        ((6, 3), (9, 3)),  # Loss to Optimizer
    ]
    
    for start, end in connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 9)
    ax.set_title('CARNet Multi-Model with Adaptive Optimizer', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/model_architecture_1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_training_progress(epoch, total_epochs, loss, lr, save_path='../results/visualizations/1'):
    """Plot training progress bar with learning rate"""
    os.makedirs(save_path, exist_ok=True)
    
    progress = (epoch + 1) / total_epochs
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
    
    # Progress bar
    ax1.barh(0, progress, color='green', alpha=0.7, height=0.5)
    ax1.barh(0, 1-progress, left=progress, color='lightgray', alpha=0.3, height=0.5)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Training Progress')
    ax1.set_title(f'Training Progress: {progress*100:.1f}% (Epoch {epoch+1}/{total_epochs})')
    ax1.set_yticks([])
    
    # Add percentage text
    ax1.text(progress/2, 0, f'{progress*100:.1f}%', ha='center', va='center', 
            fontweight='bold', color='white')
    
    # Learning rate display
    ax2.text(0.5, 0.5, f'Current LR: {lr:.2e}\nCurrent Loss: {loss:.6f}', 
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_progress_1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_adaptive_optimizer(params, optimizer_type='adamw', **kwargs):
    """Create adaptive optimizer based on type"""
    
    if optimizer_type.lower() == 'adamw':
        return optim.AdamW(params, **kwargs)
    
    elif optimizer_type.lower() == 'adabelief':
        try:
            return AdaBelief(params, **kwargs)
        except NameError:
            print("AdaBelief not available, falling back to AdamW")
            return optim.AdamW(params, **kwargs)
    
    elif optimizer_type.lower() == 'radam':
        try:
            return RAdam(params, **kwargs)
        except NameError:
            print("RAdam not available, falling back to AdamW")
            return optim.AdamW(params, **kwargs)
    
    elif optimizer_type.lower() == 'adabound':
        try:
            return AdaBound(params, **kwargs)
        except NameError:
            print("AdaBound not available, falling back to AdamW")
            return optim.AdamW(params, **kwargs)
    
    elif optimizer_type.lower() == 'adamp':
        try:
            return AdamP(params, **kwargs)
        except NameError:
            print("AdamP not available, falling back to AdamW")
            return optim.AdamW(params, **kwargs)
    
    elif optimizer_type.lower() == 'adam':
        return optim.Adam(params, **kwargs)
    
    else:
        print(f"Unknown optimizer type: {optimizer_type}, using AdamW")
        return optim.AdamW(params, **kwargs)

# 损失函数选择
def get_loss_function(loss_function_name):
    """Get loss function based on name"""
    if loss_function_name == "MSELoss":
        return nn.MSELoss()
    elif loss_function_name == "L1Loss":
        return nn.L1Loss()
    elif loss_function_name == "HuberLoss":
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_function_name}")

def compute_divergence_loss(batch_in, outputs_u, outputs_v, u_model, v_model):
    """
    计算速度场(u, v)的散度损失
    ∇·V = ∂u/∂x + ∂v/∂y ≈ 0 (对于不可压缩流体)
    
    通过重新计算模型输出来建立与输入坐标的计算关系
    """
    # 确保输入需要梯度
    batch_in.requires_grad_(True)
    
    # 重新计算模型输出，建立计算图
    outputs_u_recomputed = u_model(batch_in)
    outputs_v_recomputed = v_model(batch_in)
    
    # 如果模型返回元组，取第一个元素
    if isinstance(outputs_u_recomputed, tuple):
        outputs_u_recomputed = outputs_u_recomputed[0]
    if isinstance(outputs_v_recomputed, tuple):
        outputs_v_recomputed = outputs_v_recomputed[0]
    
    # 计算u对输入的梯度
    u_grad = torch.autograd.grad(
        outputs=outputs_u_recomputed,
        inputs=batch_in,
        grad_outputs=torch.ones_like(outputs_u_recomputed),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    # 计算v对输入的梯度
    v_grad = torch.autograd.grad(
        outputs=outputs_v_recomputed,
        inputs=batch_in,
        grad_outputs=torch.ones_like(outputs_v_recomputed),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]

    # 提取u和v对x和y的偏导数
    # batch_in的格式: [batch_size, 2] 其中[:, 0]是x坐标，[:, 1]是y坐标
    du_dx = u_grad[:, 0:1]  # ∂u/∂x (u对x的偏导数)
    du_dy = u_grad[:, 1:2]  # ∂u/∂y (u对y的偏导数)
    dv_dx = v_grad[:, 0:1]  # ∂v/∂x (v对x的偏导数)
    dv_dy = v_grad[:, 1:2]  # ∂v/∂y (v对y的偏导数)
    
    # 计算散度 (∂u/∂x + ∂v/∂y)
    divergence = du_dx +  du_dy + dv_dx +  dv_dy

    # 散度损失 (希望散度接近0)
    divergence_loss = torch.mean(divergence**2)
    
    return divergence_loss

# Training model
def train():
    x_scaled, y_scaled, T_reshaped, p_reshaped, u_reshaped, v_reshaped = args.x_scaled_1, args.y_scaled_1, args.T_1, args.p_1, args.u_1, args.v_1

    # 训练模型的选择
    if args.model_1 == "EnhancedCARNet":
        from model_CARNet_enhance import EnhancedCARNet
        T3_model = EnhancedCARNet(hidden_dim=args.hidden_size_1, num_heads=args.num_heads_1, dropout=args.dropout_1)
        p_model = EnhancedCARNet(hidden_dim=args.hidden_size_1, num_heads=args.num_heads_1, dropout=args.dropout_1)
        v_model = EnhancedCARNet(hidden_dim=args.hidden_size_1, num_heads=args.num_heads_1, dropout=args.dropout_1)
        u_model = EnhancedCARNet(hidden_dim=args.hidden_size_1, num_heads=args.num_heads_1, dropout=args.dropout_1)
        print(f"Using EnhancedCARNet model")
    elif args.model_1 == "EnhancedCARNet_v2":
        from model_CARNet_enhance_v2 import EnhancedCARNet_v2
        T3_model = EnhancedCARNet_v2(hidden_dim=args.hidden_size_1, num_heads=args.num_heads_1, dropout=args.dropout_1)
        p_model = EnhancedCARNet_v2(hidden_dim=args.hidden_size_1, num_heads=args.num_heads_1, dropout=args.dropout_1)
        v_model = EnhancedCARNet_v2(hidden_dim=args.hidden_size_1, num_heads=args.num_heads_1, dropout=args.dropout_1)
        u_model = EnhancedCARNet_v2(hidden_dim=args.hidden_size_1, num_heads=args.num_heads_1, dropout=args.dropout_1)
        print(f"Using EnhancedCARNet_v2 model")
    else:
        raise ValueError(f"Unsupported model: {args.model_1}")

    # 将模型移动到设备
    p_model = p_model.to(args.device_1)
    v_model = v_model.to(args.device_1)
    u_model = u_model.to(args.device_1)
    T3_model = T3_model.to(args.device_1)

    # 多GPU支持
    if (args.use_data_parallel_1 and torch.cuda.is_available() 
        and torch.cuda.device_count() >= len(args.gpu_device_ids_1) 
        and len(args.gpu_device_ids_1) >= 2):
        device_ids = args.gpu_device_ids_1
        print(f"Enabling DataParallel on GPUs: {device_ids}")
        p_model = nn.DataParallel(p_model, device_ids=device_ids)
        v_model = nn.DataParallel(v_model, device_ids=device_ids)
        u_model = nn.DataParallel(u_model, device_ids=device_ids)
        T3_model = nn.DataParallel(T3_model, device_ids=device_ids)
        print(f"All models are now wrapped in DataParallel")
    else:
        print(f"Using single GPU or CPU. Device: {args.device_1}")
        p_model = p_model.to(args.device_1)
        v_model = v_model.to(args.device_1)
        u_model = u_model.to(args.device_1)
        T3_model = T3_model.to(args.device_1)

    # Set training mode for all models
    if isinstance(p_model, nn.DataParallel):
        p_model.module.train()
    else:
        p_model.train()
    
    if isinstance(v_model, nn.DataParallel):
        v_model.module.train()
    else:
        v_model.train()
    
    if isinstance(u_model, nn.DataParallel):
        u_model.module.train()
    else:
        u_model.train()
    
    if isinstance(T3_model, nn.DataParallel):
        T3_model.module.train()
    else:
        T3_model.train()

    # 定义损失函数
    criterion = get_loss_function(args.loss_function_1)

    # 收集所有模型的参数
    all_params = list(T3_model.parameters()) + \
                 list(p_model.parameters()) + \
                 list(v_model.parameters()) + \
                 list(u_model.parameters())

    # 创建自适应优化器
    optimizer = create_adaptive_optimizer(
        all_params, 
        optimizer_type=args.optimizer_1,
        lr=args.learning_rate_1, 
        weight_decay=args.weight_decay_1,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 学习率调度器：多种选择
    if args.scheduler_1 == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=args.patience_1,
            verbose=True,
            min_lr=1e-8
        )
    elif args.scheduler_1 == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_1, eta_min=1e-8)
    elif args.scheduler_1 == "CosineAnnealingWarmRestarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # 第一次重启的周期
            T_mult=2,  # 每次重启后周期翻倍
            eta_min=1e-8
        )
    elif args.scheduler_1 == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif args.scheduler_1 == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif args.scheduler_1 == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.5)
    elif args.scheduler_1 == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=args.learning_rate_1 * 10,  # 最大学习率
            epochs=args.epochs_1,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # 预热阶段占总训练的比例
            anneal_strategy='cos'
        )
    elif args.scheduler_1 == "CyclicLR":
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-8,
            max_lr=args.learning_rate_1,
            step_size_up=2000,
            mode='triangular'
        )
    elif args.scheduler_1 == "LinearLR":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=args.epochs_1
        )
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler_1}")

    # 创建数据集
    # 确保数据维度正确
    if T_reshaped.ndim > 2:
        T_reshaped = T_reshaped.reshape(T_reshaped.shape[0], -1)
    if p_reshaped.ndim > 2:
        p_reshaped = p_reshaped.reshape(p_reshaped.shape[0], -1)
    if u_reshaped.ndim > 2:
        u_reshaped = u_reshaped.reshape(u_reshaped.shape[0], -1)
    if v_reshaped.ndim > 2:
        v_reshaped = v_reshaped.reshape(v_reshaped.shape[0], -1)
    
    train_dataset = TensorDataset(torch.FloatTensor(np.concatenate((x_scaled, y_scaled), axis=1)), 
                                 torch.FloatTensor(np.concatenate((T_reshaped, p_reshaped, u_reshaped, v_reshaped), axis=1)))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_1, shuffle=True)

    print("Starting training with adaptive optimizer...")
    print(f"Device: {args.device_1}")
    print(f"Use DataParallel: {args.use_data_parallel_1}")
    print(f"GPU Device IDs: {args.gpu_device_ids_1}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Batch size: {args.batch_size_1}")
    print(f"Epochs: {args.epochs_1}")
    print(f"Initial learning rate: {args.learning_rate_1}")
    print(f"Optimizer: {args.optimizer_1}")
    print(f"Scheduler: {args.scheduler_1}")
    print(f"Loss function: {args.loss_function_1}")
    print(f"Loss weights - T: {args.loss_T_weight_1}, p: {args.loss_atm_weight_1}, u: {args.loss_u_weight_1}, v: {args.loss_v_weight_1}")
    print(f"Divergence loss weight: {args.divergence_loss_weight}")
    print("-" * 50)
    
    # Initialize history records
    history_loss = []
    history_metrics = {
        'r2': {'T': [], 'p': [], 'u': [], 'v': []},
        'mse': {'T': [], 'p': [], 'u': [], 'v': []},
        'mae': {'T': [], 'p': [], 'u': [], 'v': []}
    }
    history_lr = []
    history_divergence_loss = []  # 记录散度损失
    
    # Plot initial data distribution
    print("Plotting data distribution...")
    plot_data_distribution(x_scaled, y_scaled, T_reshaped, p_reshaped, u_reshaped, v_reshaped, f'../results/visualizations/1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}')
    plot_correlation_matrix(x_scaled, y_scaled, T_reshaped, p_reshaped, u_reshaped, v_reshaped, f'../results/visualizations/1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}')
    plot_model_architecture(f'../results/visualizations/1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}')
    
    best_loss = float('inf')
    patience = args.early_stopping_patience_1
    patience_counter = 0
    
    for epoch in range(args.epochs_1):
        epoch_loss = 0
        epoch_divergence_loss = 0  # 每个epoch的散度损失
        epoch_predictions = {'T': [], 'p': [], 'u': [], 'v': []}
        epoch_targets = {'T': [], 'p': [], 'u': [], 'v': []}
        
        for batch_idx, (batch_in, batch_out) in enumerate(train_loader):
            batch_in = batch_in.to(args.device_1)
            batch_out = batch_out.to(args.device_1)
            optimizer.zero_grad()
            
            # Forward pass
            outputs_T = T3_model(batch_in)
            outputs_p = p_model(batch_in)
            outputs_v = v_model(batch_in)
            outputs_u = u_model(batch_in)


            
            # 处理双输出模型（主输出和辅助输出）
            if isinstance(outputs_T, tuple):
                outputs_T, _ = outputs_T
            if isinstance(outputs_p, tuple):
                outputs_p, _ = outputs_p
            if isinstance(outputs_v, tuple):
                outputs_v, _ = outputs_v
            if isinstance(outputs_u, tuple):
                outputs_u, _ = outputs_u
            
            # 添加偏移量
            outputs_p = outputs_p + 101325  # 凸出小变化

            loss_T = criterion(outputs_T, batch_out[:,0].unsqueeze(1))
            loss_p = criterion(outputs_p, batch_out[:,1].unsqueeze(1))
            loss_v = criterion(outputs_v, batch_out[:,2].unsqueeze(1))
            loss_u = criterion(outputs_u, batch_out[:,3].unsqueeze(1))

            # 计算散度损失
            divergence_loss = compute_divergence_loss(batch_in, outputs_u, outputs_v, u_model, v_model)
            
            # 总损失 = 数据损失 + 散度损失
            loss_sum = (args.loss_T_weight_1 * loss_T + 
                       args.loss_atm_weight_1 * loss_p + 
                       args.loss_v_weight_1 * loss_v + 
                       args.loss_u_weight_1 * loss_u +
                       args.divergence_loss_weight * divergence_loss)
            
            # Backward pass
            loss_sum.backward()
            optimizer.step()
            
            # 对于需要每个batch调用的调度器
            if args.scheduler_1 in ["OneCycleLR", "CyclicLR"]:
                scheduler.step()

            # Accumulate loss
            epoch_loss += loss_sum.item()
            epoch_divergence_loss += divergence_loss.item()
            
            # Collect predictions and targets
            epoch_predictions['T'].extend(outputs_T.squeeze().cpu().detach().numpy())
            epoch_predictions['p'].extend(outputs_p.squeeze().cpu().detach().numpy())
            epoch_predictions['u'].extend(outputs_u.squeeze().cpu().detach().numpy())
            epoch_predictions['v'].extend(outputs_v.squeeze().cpu().detach().numpy())
            
            epoch_targets['T'].extend(batch_out[:,0].cpu().numpy())
            epoch_targets['p'].extend(batch_out[:,1].cpu().numpy())
            epoch_targets['u'].extend(batch_out[:,2].cpu().numpy())
            epoch_targets['v'].extend(batch_out[:,3].cpu().numpy())
            
            # Print loss (reduced frequency)
            if batch_idx % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{args.epochs_1}, Batch {batch_idx}, Loss: {loss_sum.item():.6f}, "
                      f"Divergence Loss: {divergence_loss.item():.6f}, LR: {current_lr:.2e}")
            
            # Clean GPU memory
            del outputs_T, outputs_p, outputs_v, outputs_u
            torch.cuda.empty_cache()
        
        # Calculate average loss and current learning rate
        avg_loss = epoch_loss / len(train_loader)
        avg_divergence_loss = epoch_divergence_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 学习率调度器步进：根据类型调用
        if args.scheduler_1 == "ReduceLROnPlateau":
            scheduler.step(avg_loss)
        elif args.scheduler_1 == "OneCycleLR":
            # OneCycleLR需要在每个batch后调用
            pass  # 在batch循环中调用
        elif args.scheduler_1 == "CyclicLR":
            # CyclicLR需要在每个batch后调用
            pass  # 在batch循环中调用
        else:
            scheduler.step()
        
        history_loss.append(avg_loss)
        history_divergence_loss.append(avg_divergence_loss)
        history_lr.append(current_lr)
        
        # 记录到SwanLab
        swanlab.log({
            "epoch": epoch + 1,
            "total_loss": avg_loss,
            "divergence_loss": avg_divergence_loss,
            "learning_rate": current_lr,
            "best_loss": best_loss,
            "patience_counter": patience_counter,
            "is_best_model": avg_loss < best_loss
        })
        
        # Calculate evaluation metrics
        metrics_log = {}
        for var in ['T', 'p', 'u', 'v']:
            pred = np.array(epoch_predictions[var])
            true = np.array(epoch_targets[var])
            
            r2 = r2_score(true, pred)
            mse = mean_squared_error(true, pred)
            mae = mean_absolute_error(true, pred)
            
            history_metrics['r2'][var].append(r2)
            history_metrics['mse'][var].append(mse)
            history_metrics['mae'][var].append(mae)
            
            # 记录每个变量的指标到SwanLab
            metrics_log[f"{var}_r2"] = r2
            metrics_log[f"{var}_mse"] = mse
            metrics_log[f"{var}_mae"] = mae
        
        # 更新SwanLab记录，包含详细的指标
        swanlab.log(metrics_log)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            def _sd(m):
                return m.module.state_dict() if isinstance(m, nn.DataParallel) else m.state_dict()
            
            model_dir = f'../results/best_model_1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}'
            os.makedirs(model_dir, exist_ok=True)
            
            torch.save({
                'T3_model': _sd(T3_model),
                'p_model': _sd(p_model),
                'v_model': _sd(v_model),
                'u_model': _sd(u_model),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'optimizer_type': args.optimizer_1
            }, f'{model_dir}/best_model.pth')
        else:
            patience_counter += 1
        
        # Plot training progress
        plot_training_progress(epoch, args.epochs_1, avg_loss, current_lr, f'../results/visualizations/1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}')
        
        # Regular visualization
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs_1}, Avg Loss: {avg_loss:.6f}, "
                  f"Avg Divergence Loss: {avg_divergence_loss:.6f}, LR: {current_lr:.2e}")
            print(f"Best Loss: {best_loss:.6f}")
            print(f"Patience: {patience_counter}/{patience}")
            
            # Plot prediction analysis
            plot_prediction_analysis(epoch_predictions, epoch_targets, epoch + 1, f'../results/visualizations/1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}')
            
            # Plot training metrics
            plot_training_metrics(history_loss, history_metrics, history_lr, f'../results/visualizations/1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered! No improvement for {patience} epochs")
            break
    
    # Final visualization
    print("Plotting final training results...")
    plot_training_metrics(history_loss, history_metrics, history_lr, f'../results/visualizations/1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}')
    
    # Save training history
    history_dir = f'../results/training_history_1_{args.optimizer_1}_{args.scheduler_1}_{args.loss_function_1}'
    os.makedirs(history_dir, exist_ok=True)
    
    training_history = {
        'loss': history_loss,
        'divergence_loss': history_divergence_loss,
        'metrics': history_metrics,
        'learning_rates': history_lr,
        'best_loss': best_loss,
        'final_epoch': epoch + 1,
        'optimizer_type': args.optimizer_1
    }
    
    np.save(f'{history_dir}/training_history.npy', training_history)
    
    # 记录最终结果到SwanLab
    final_metrics = {}
    for var in ['T', 'atm', 'u', 'v']:
        final_metrics[f"final_{var}_r2"] = history_metrics['r2'][var][-1] if history_metrics['r2'][var] else 0.0
        final_metrics[f"final_{var}_mse"] = history_metrics['mse'][var][-1] if history_metrics['mse'][var] else float('inf')
        final_metrics[f"final_{var}_mae"] = history_metrics['mae'][var][-1] if history_metrics['mae'][var] else float('inf')
    
    swanlab.log({
        "final_total_loss": best_loss,
        "final_divergence_loss": history_divergence_loss[-1] if history_divergence_loss else 0.0,
        "final_learning_rate": current_lr,
        "total_epochs": epoch + 1,
        "training_completed": True,
        **final_metrics
    })
    
    # 完成SwanLab运行
    swanlab.finish()
    
    print("Training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Final epoch: {epoch + 1}")
    print(f"Final learning rate: {current_lr:.2e}")
    print(f"Final divergence loss: {history_divergence_loss[-1] if history_divergence_loss else 0.0:.6f}")
    
    return history_loss, history_metrics, history_lr, history_divergence_loss

if __name__ == '__main__':
    start_time = datetime.now()
    print(f"Training start time: {start_time}")

    trainer = Trainer1P()
    history_loss, history_metrics, history_lr, history_divergence_loss = trainer.train()

    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"Training end time: {end_time}")
    print(f"Total training duration: {training_duration}")

    print(f'Final loss: {history_loss[-1]:.6f}')
    print(f'Final learning rate: {history_lr[-1]:.2e}')
    print(f'Final divergence loss: {history_divergence_loss[-1] if history_divergence_loss else 0.0:.6f}')
    print("Training completed! All visualization results saved to ../results/visualizations/")
    print("Training history saved to ../results/")