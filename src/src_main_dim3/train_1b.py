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

# OOP wrapper aligned with Trainer0 style
class Trainer1:
    def __init__(self):
        self.set_parser()
        self.args.gpu_device_ids_1 = self.parse_gpu_device_ids(self.args.gpu_device_ids_1)
        self.init_swanlab()

    def set_parser(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--epochs_1', type=int, default=config_train.epochs_1)
        self.parser.add_argument('--batch_size_1', type=int, default=config_train.batch_size_1)
        self.parser.add_argument('--learning_rate_1', type=float, default=config_train.learning_rate_1)
        self.parser.add_argument('--weight_decay_1', type=float, default=config_train.weight_decay_1)
        self.parser.add_argument('--patience_1', type=int, default=config_train.patience_1)
        self.parser.add_argument('--device_1', type=str, default=config_train.device)
        self.parser.add_argument('--use_data_parallel_1', type=bool, default=config_train.use_data_parallel)
        self.parser.add_argument('--gpu_device_ids_1', type=str, default=str(config_train.gpu_device_ids),
                   help='GPU device IDs as string, e.g., "8" or "0,1,2"')
        self.parser.add_argument('--early_stopping_patience_1', type=int, default=config_train.early_stopping_patience_1)
        self.parser.add_argument('--loss_function_1', type=str, default=config_train.loss_function_1)
        self.parser.add_argument('--scheduler_1', type=str, default=config_train.scheduler_1)
        self.parser.add_argument('--hidden_size_1', type=int, default=config_train.hidden_size)
        self.parser.add_argument('--num_heads_1', type=int, default=config_train.num_heads)
        self.parser.add_argument('--method_1', type=str, default=config_train.method_1)
        self.parser.add_argument('--dropout_1', type=float, default=config_train.dropout)
        self.parser.add_argument('--optimizer_1', type=str, default=config_train.optimizer_1)
        self.parser.add_argument('--x_scaled_1', type=np.ndarray, default=config_train.x_scaled_1)
        self.parser.add_argument('--y_scaled_1', type=np.ndarray, default=config_train.y_scaled_1)
        self.parser.add_argument('--T_1', type=np.ndarray, default=config_train.T_1)
        self.parser.add_argument('--p_1', type=np.ndarray, default=config_train.p_1)
        self.parser.add_argument('--u_1', type=np.ndarray, default=config_train.u_1)
        self.parser.add_argument('--v_1', type=np.ndarray, default=config_train.v_1)
        self.parser.add_argument('--loss_T_weight_1', type=float, default=config_train.loss_T_weight)
        self.parser.add_argument('--loss_atm_weight_1', type=float, default=config_train.loss_atm_weight)
        self.parser.add_argument('--loss_u_weight_1', type=float, default=config_train.loss_u_weight)
        self.parser.add_argument('--loss_v_weight_1', type=float, default=config_train.loss_v_weight)
        self.parser.add_argument('--model_1', type=str, default=config_train.model_1)
        self.args = self.parser.parse_args()
        return self.args

    # 处理GPU设备ID参数
    def parse_gpu_device_ids(self, device_ids_str):
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

    # 初始化SwanLab运行
    def init_swanlab(self):
        """Initialize SwanLab run and return the run handle."""
        self.run_1 = swanlab.init(
            name=f"{self.args.model_1}_MultiTask_Training_1",
            project="sci",
    config={
                "model": self.args.model_1,
                "optimizer": self.args.optimizer_1,
                "learning_rate": self.args.learning_rate_1,
                "batch_size": self.args.batch_size_1,
                "epochs": self.args.epochs_1,
                "loss_function": self.args.loss_function_1,
                "scheduler": self.args.scheduler_1,
                "early_stopping_patience": self.args.early_stopping_patience_1,
                "device": str(self.args.device_1),
        "tasks": ["T", "p", "u", "v"],
        "loss_weights": {
                    "T": self.args.loss_T_weight_1,
                    "p": self.args.loss_atm_weight_1,
                    "u": self.args.loss_u_weight_1,
                    "v": self.args.loss_v_weight_1
                },
                "use_data_parallel": self.args.use_data_parallel_1,
                "gpu_device_ids": self.args.gpu_device_ids_1,
                "hidden_size": self.args.hidden_size_1,
                "num_heads": self.args.num_heads_1,
                "dropout": self.args.dropout_1,
            },
        )
        return self.run_1

    def train(self):
        warnings.filterwarnings('ignore')
        x_scaled, y_scaled, T_reshaped, p_reshaped, u_reshaped, v_reshaped = self.args.x_scaled_1, self.args.y_scaled_1, self.args.T_1, self.args.p_1, self.args.u_1, self.args.v_1

        # 训练模型的选择
        if self.args.model_1 == "EnhancedCARNet":
            from model_CARNet_enhance import EnhancedCARNet
            T3_model = EnhancedCARNet(hidden_dim=self.args.hidden_size_1, num_heads=self.args.num_heads_1, dropout=self.args.dropout_1)
            p_model = EnhancedCARNet(hidden_dim=self.args.hidden_size_1, num_heads=self.args.num_heads_1, dropout=self.args.dropout_1)
            v_model = EnhancedCARNet(hidden_dim=self.args.hidden_size_1, num_heads=self.args.num_heads_1, dropout=self.args.dropout_1)
            u_model = EnhancedCARNet(hidden_dim=self.args.hidden_size_1, num_heads=self.args.num_heads_1, dropout=self.args.dropout_1)
            print(f"Using EnhancedCARNet model")
        elif self.args.model_1 == 'EnhancedCARNet_v2':
            from model_CARNet_enhance_v2 import EnhancedCARNet_v2
            T3_model = EnhancedCARNet_v2(hidden_dim=self.args.hidden_size_1, num_heads=self.args.num_heads_1, dropout=self.args.dropout_1)
            p_model = EnhancedCARNet_v2(hidden_dim=self.args.hidden_size_1, num_heads=self.args.num_heads_1, dropout=self.args.dropout_1)
            v_model = EnhancedCARNet_v2(hidden_dim=self.args.hidden_size_1, num_heads=self.args.num_heads_1, dropout=self.args.dropout_1)
            u_model = EnhancedCARNet_v2(hidden_dim=self.args.hidden_size_1, num_heads=self.args.num_heads_1, dropout=self.args.dropout_1)
            print(f"Using EnhancedCARNet_v2 model")
        else:
            raise ValueError(f"Unsupported model: {self.args.model_1}")

        # 将模型移动到设备
        p_model = p_model.to(self.args.device_1)
        v_model = v_model.to(self.args.device_1)
        u_model = u_model.to(self.args.device_1)
        T3_model = T3_model.to(self.args.device_1)

        # 多GPU支持
        if (self.args.use_data_parallel_1 and torch.cuda.is_available()
            and torch.cuda.device_count() >= len(self.args.gpu_device_ids_1)
            and len(self.args.gpu_device_ids_1) >= 2):
            device_ids = self.args.gpu_device_ids_1
            print(f"Enabling DataParallel on GPUs: {device_ids}")
            p_model = nn.DataParallel(p_model, device_ids=device_ids)
            v_model = nn.DataParallel(v_model, device_ids=device_ids)
            u_model = nn.DataParallel(u_model, device_ids=device_ids)
            T3_model = nn.DataParallel(T3_model, device_ids=device_ids)
            print(f"All models are now wrapped in DataParallel")
        else:
            print(f"Using single GPU or CPU. Device: {self.args.device_1}")
            p_model = p_model.to(self.args.device_1)
            v_model = v_model.to(self.args.device_1)
            u_model = u_model.to(self.args.device_1)
            T3_model = T3_model.to(self.args.device_1)

        # 定义损失函数
        if self.args.loss_function_1 == "MSELoss":
            criterion = nn.MSELoss()
        elif self.args.loss_function_1 == "L1Loss":
            criterion = nn.L1Loss()
        elif self.args.loss_function_1 == "HuberLoss":
            criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.args.loss_function_1}")

        # 收集所有模型的参数
        all_params = list(T3_model.parameters()) + \
                     list(p_model.parameters()) + \
                     list(v_model.parameters()) + \
                     list(u_model.parameters())

        # 优化器
        if self.args.optimizer_1 == "AdamW":
            optimizer = optim.AdamW(all_params,
                                    lr=self.args.learning_rate_1,
                                    weight_decay=self.args.weight_decay_1,
                                    betas=(0.9, 0.999),
                                    eps=1e-8)
        elif self.args.optimizer_1 == "Adam":
            optimizer = optim.Adam(all_params,
                                   lr=self.args.learning_rate_1,
                                   weight_decay=self.args.weight_decay_1,
                                   betas=(0.9, 0.999),
                                   eps=1e-8)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer_1}")

        # 创建数据集
        train_dataset = TensorDataset(torch.FloatTensor(np.concatenate((x_scaled, y_scaled), axis=1)),
                                     torch.FloatTensor(np.concatenate((T_reshaped, p_reshaped, u_reshaped, v_reshaped), axis=1)))
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size_1, shuffle=True)

        # 学习率调度器：多种选择
        if self.args.scheduler_1 == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=self.args.patience_1,
                verbose=True,
                min_lr=1e-8
            )
        elif self.args.scheduler_1 == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_1, eta_min=1e-8)
        elif self.args.scheduler_1 == "CosineAnnealingWarmRestarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,  # 第一次重启的周期
                T_mult=2,  # 每次重启后周期翻倍
                eta_min=1e-8
            )
        elif self.args.scheduler_1 == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif self.args.scheduler_1 == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        elif self.args.scheduler_1 == "MultiStepLR":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.5)
        elif self.args.scheduler_1 == "OneCycleLR":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.args.learning_rate_1 * 10,  # 最大学习率
                epochs=self.args.epochs_1,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,  # 预热阶段占总训练的比例
                anneal_strategy='cos'
            )
        elif self.args.scheduler_1 == "CyclicLR":
            scheduler = optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=1e-8,
                max_lr=self.args.learning_rate_1,
                step_size_up=2000,
                mode='triangular'
            )
        elif self.args.scheduler_1 == "LinearLR":
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.args.epochs_1
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.args.scheduler_1}")

        print("Starting training with adaptive optimizer...")
        print(f"Device: {self.args.device_1}")
        print(f"Use DataParallel: {self.args.use_data_parallel_1}")
        print(f"GPU Device IDs: {self.args.gpu_device_ids_1}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"Batch size: {self.args.batch_size_1}")
        print(f"Epochs: {self.args.epochs_1}")
        print(f"Initial learning rate: {self.args.learning_rate_1}")
        print(f"Optimizer: {type(optimizer).__name__}")
        print(f"Scheduler: {type(scheduler).__name__}")
        print(f"Loss function: {self.args.loss_function_1}")
        print(f"Loss weights - T: {self.args.loss_T_weight_1}, p: {self.args.loss_atm_weight_1}, u: {self.args.loss_u_weight_1}, v: {self.args.loss_v_weight_1}")
        print("-" * 50)

        # Initialize history records
        history_loss = []
        history_metrics = {
            'r2': {'T': [], 'p': [], 'u': [], 'v': []},
            'mse': {'T': [], 'p': [], 'u': [], 'v': []},
            'mae': {'T': [], 'p': [], 'u': [], 'v': []}
        }
        history_lr = []

        best_loss = float('inf')
        patience = self.args.early_stopping_patience_1
        patience_counter = 0

        for epoch in tqdm(range(self.args.epochs_1)):
            epoch_loss = 0
            epoch_predictions = {'T': [], 'p': [], 'u': [], 'v': []}
            epoch_targets = {'T': [], 'p': [], 'u': [], 'v': []}

            for batch_idx, (batch_in, batch_out) in enumerate(train_loader):
                batch_in = batch_in.to(self.args.device_1)
                batch_out = batch_out.to(self.args.device_1)
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

                loss_T = criterion(outputs_T, batch_out[:,0].unsqueeze(1))
                loss_p = criterion(outputs_p, batch_out[:,1].unsqueeze(1))
                loss_v = criterion(outputs_v, batch_out[:,2].unsqueeze(1))
                loss_u = criterion(outputs_u, batch_out[:,3].unsqueeze(1))

                loss_sum = (self.args.loss_T_weight_1 * loss_T +
                           self.args.loss_atm_weight_1 * loss_p +
                           self.args.loss_v_weight_1 * loss_v +
                           self.args.loss_u_weight_1 * loss_u)

                # Backward pass
                loss_sum.backward()
                optimizer.step()

                # 对于需要每个batch调用的调度器
                if self.args.scheduler_1 in ["OneCycleLR", "CyclicLR"]:
                    scheduler.step()

                # Accumulate loss
                epoch_loss += loss_sum.item()

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
                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{self.args.epochs_1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss_sum.item():.6f}, LR: {current_lr:.2e}")

            # Calculate average loss and current learning rate
            avg_loss = epoch_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']

            # 学习率调度器步进：根据类型调用
            if self.args.scheduler_1 == "ReduceLROnPlateau":
                scheduler.step(avg_loss)
            elif self.args.scheduler_1 == "OneCycleLR":
                # OneCycleLR需要在每个batch后调用
                pass  # 在batch循环中调用
            elif self.args.scheduler_1 == "CyclicLR":
                # CyclicLR需要在每个batch后调用
                pass  # 在batch循环中调用
            else:
                scheduler.step()

            history_loss.append(avg_loss)
            history_lr.append(current_lr)

            # 记录到SwanLab
            swanlab.log({
                "epoch": epoch + 1,
                "total_loss": avg_loss,
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

                model_dir = f'../results/best_model_1_{self.args.optimizer_1}_{self.args.scheduler_1}_{self.args.loss_function_1}'
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
                    'optimizer_type': self.args.optimizer_1
                }, f'{model_dir}/best_model.pth')
            else:
                patience_counter += 1

            # Regular visualization
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.args.epochs_1}, Avg Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
                print(f"Best Loss: {best_loss:.6f}")
                print(f"Patience: {patience_counter}/{patience}")

                # Plot prediction analysis
                plot_prediction_analysis(epoch_predictions, epoch_targets, epoch + 1, f'../results/visualizations/1_{self.args.optimizer_1}_{self.args.scheduler_1}_{self.args.loss_function_1}')

                # Plot training metrics
                plot_training_metrics(history_loss, history_metrics, history_lr, f'../results/visualizations/1_{self.args.optimizer_1}_{self.args.scheduler_1}_{self.args.loss_function_1}')

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs")
                break

        # Final visualization
        print("Plotting final training results...")
        plot_training_metrics(history_loss, history_metrics, history_lr, f'../results/visualizations/1_{self.args.optimizer_1}_{self.args.scheduler_1}_{self.args.loss_function_1}')

        # Save training history
        history_dir = f'../results/training_history_1_{self.args.optimizer_1}_{self.args.scheduler_1}_{self.args.loss_function_1}'
        os.makedirs(history_dir, exist_ok=True)

        training_history = {
            'loss': history_loss,
            'metrics': history_metrics,
            'learning_rates': history_lr,
            'best_loss': best_loss,
            'final_epoch': epoch + 1,
            'optimizer_type': self.args.optimizer_1
        }

        np.save(f'{history_dir}/training_history.npy', training_history)

        # 记录最终结果到SwanLab
        final_metrics = {}
        for var in ['T', 'p', 'u', 'v']:
            final_metrics[f"final_{var}_r2"] = history_metrics['r2'][var][-1] if history_metrics['r2'][var] else 0.0
            final_metrics[f"final_{var}_mse"] = history_metrics['mse'][var][-1] if history_metrics['mse'][var] else float('inf')
            final_metrics[f"final_{var}_mae"] = history_metrics['mae'][var][-1] if history_metrics['mae'][var] else float('inf')

        swanlab.log({
            "final_total_loss": best_loss,
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

        return history_loss, history_metrics, history_lr


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
    plt.savefig(f'{save_path}/training_metrics.png', dpi=300, bbox_inches='tight')
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
    plt.savefig(f'{save_path}/prediction_analysis_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
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
    plt.savefig(f'{save_path}/data_distribution.png', dpi=300, bbox_inches='tight')
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
    plt.savefig(f'{save_path}/correlation_matrix.png', dpi=300, bbox_inches='tight')
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
        'p_model': (3, 4),
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
        ((1, 8), (3, 4)),  # Input to p_model
        ((1, 8), (3, 2)),  # Input to v_model
        ((1, 8), (3, 0)),  # Input to u_model
        ((3, 6), (6, 3)),  # T3_model to Loss
        ((3, 4), (6, 3)),  # p_model to Loss
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
    plt.savefig(f'{save_path}/model_architecture.png', dpi=300, bbox_inches='tight')
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
    plt.savefig(f'{save_path}/training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    start_time = datetime.now()
    print(f"Training start time: {start_time}")
    
    trainer1 = Trainer1()
    history_loss, history_metrics, history_lr = trainer1.train()
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"Training end time: {end_time}")
    print(f"Total training duration: {training_duration}")
    
    print(f'Final loss: {history_loss[-1]:.6f}')
    print(f'Final learning rate: {history_lr[-1]:.2e}')
    print("Training completed! All visualization results saved to ../results/visualizations/")
    print("Training history saved to ../results/")