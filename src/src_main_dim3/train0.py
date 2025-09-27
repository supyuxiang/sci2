import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_CARNet_enhance_v2 import EnhancedCARNet_v2
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



class Trainer0:
    def __init__(self):
        self.set_parser()
        self.args.gpu_device_ids_0 = self.parse_gpu_device_ids(self.args.gpu_device_ids_0)
        self.init_swanlab()

    def set_parser(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--epochs_0', type=int, default=config_train.epochs_0)
        self.parser.add_argument('--batch_size_0', type=int, default=config_train.batch_size_0)
        self.parser.add_argument('--learning_rate_0', type=float, default=config_train.learning_rate_0)
        self.parser.add_argument('--weight_decay_0', type=float, default=config_train.weight_decay_0)
        self.parser.add_argument('--patience_0', type=int, default=config_train.patience_0)
        self.parser.add_argument('--device_0', type=str, default=config_train.device)
        self.parser.add_argument('--use_data_parallel_0', type=bool, default=config_train.use_data_parallel)
        self.parser.add_argument('--gpu_device_ids_0', type=str, default=str(config_train.gpu_device_ids), 
                        help='GPU device IDs as string, e.g., "8" or "0,1,2"')
        self.parser.add_argument('--early_stopping_patience_0', type=int, default=config_train.early_stopping_patience_0)
        self.parser.add_argument('--loss_function_0', type=str, default=config_train.loss_function)
        self.parser.add_argument('--scheduler_0', type=str, default=config_train.scheduler_0)
        self.parser.add_argument('--hidden_size_0', type=int, default=config_train.hidden_size)
        self.parser.add_argument('--num_heads_0', type=int, default=config_train.num_heads)
        self.parser.add_argument('--method_0', type=str, default=config_train.method_0)
        self.parser.add_argument('--dropout_0', type=float, default=config_train.dropout)
        self.parser.add_argument('--optimizer_0', type=str, default=config_train.optimizer_0)
        self.parser.add_argument('--x_scaled_0', type=np.ndarray, default=config_train.x_scaled_0)
        self.parser.add_argument('--y_scaled_0', type=np.ndarray, default=config_train.y_scaled_0)
        self.parser.add_argument('--T_array_0', type=np.ndarray, default=config_train.T_array_0)
        self.parser.add_argument('--model_0', type=str, default=config_train.model_0)
        self.parser.add_argument('--use_mixed_precision_0', type=bool, default=config_train.use_mixed_precision)
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
        self.run_0 = swanlab.init(
            name=f"{self.args.model_0}_Training_0",
            project="sci",
            config={
                "model": self.args.model_0,
                "optimizer": self.args.optimizer_0,
                "learning_rate": self.args.learning_rate_0,
                "batch_size": self.args.batch_size_0,
                "epochs": self.args.epochs_0,
                "loss_function": self.args.loss_function_0,
                "scheduler": self.args.scheduler_0,
                "early_stopping_patience": self.args.early_stopping_patience_0,
                "device": str(self.args.device_0),
                "use_data_parallel": self.args.use_data_parallel_0,
                "gpu_device_ids": self.args.gpu_device_ids_0,
                "hidden_size": self.args.hidden_size_0,
                "num_heads": self.args.num_heads_0,
                "dropout": self.args.dropout_0,
            },
        )
        return self.run_0



    # 创建结果目录
    def plot_training_metrics(self, history_loss, history_metrics, history_lr, save_path='../results/visualizations/0'):
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
    
        plt.tight_layout()
        plt.savefig(f'{save_path}/training_metrics.png', dpi=1200, bbox_inches='tight')
        plt.show()
    plt.close()

    def plot_prediction_analysis(self, all_predictions, all_targets, epoch, save_path='../results/visualizations/0'):
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
        plt.savefig(f'{save_path}/prediction_analysis_epoch_{epoch}.png', dpi=1200, bbox_inches='tight')
        plt.show()
        plt.close()

    def train(self):
        warnings.filterwarnings('ignore')
        x_scaled, y_scaled, T = self.args.x_scaled_0, self.args.y_scaled_0, self.args.T_array_0
        if self.args.model_0 == "EnhancedCARNet":
            from model_CARNet_enhance import EnhancedCARNet
            T_model = EnhancedCARNet(hidden_dim=self.args.hidden_size_0, num_heads=self.args.num_heads_0, dropout=self.args.dropout_0)
            print(f"Using EnhancedCARNet model")
        elif self.args.model_0 == 'EnhancedCARNet_v2':
            from model_CARNet_enhance_v2 import EnhancedCARNet_v2
            T_model = EnhancedCARNet_v2(hidden_dim=self.args.hidden_size_0, num_heads=self.args.num_heads_0, dropout=self.args.dropout_0)
            print(f"Using EnhancedCARNet_v2 model")
        else:
            raise ValueError(f"Unsupported model: {self.args.model_0}")
    
        T_model = T_model.to(self.args.device_0)

        # 多GPU支持：修正设备检查条件
        if (self.args.use_data_parallel_0 and torch.cuda.is_available() 
            and torch.cuda.device_count() >= len(self.args.gpu_device_ids_0) 
            and len(self.args.gpu_device_ids_0) >= 2):
            device_ids = self.args.gpu_device_ids_0
            print(f"Enabling DataParallel on GPUs: {device_ids}")
            T_model = nn.DataParallel(T_model, device_ids=device_ids)
            print(f"Model is now wrapped in DataParallel")
        else:
            print(f"Using single GPU or CPU. Device: {self.args.device_0}")
            T_model = T_model.to(self.args.device_0)
    
        if self.args.optimizer_0 == "AdamW":
            optimizer = optim.AdamW(T_model.parameters(), 
                            lr=self.args.learning_rate_0, 
                            weight_decay=self.args.weight_decay_0,
                            betas=(0.9, 0.999),
                            eps=1e-8)
        elif self.args.optimizer_0 == "Adam":
            optimizer = optim.Adam(T_model.parameters(), 
                            lr=self.args.learning_rate_0, 
                            weight_decay=self.args.weight_decay_0,
                            betas=(0.9, 0.999),
                            eps=1e-8)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer_0}")
    
        # 学习率调度器：多种选择
        if self.args.scheduler_0 == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=self.args.patience_0,
                verbose=True,
                min_lr=1e-8
            )
        elif self.args.scheduler_0 == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_0, eta_min=1e-8)
        elif self.args.scheduler_0 == "CosineAnnealingWarmRestarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=10,  # 第一次重启的周期
                T_mult=2,  # 每次重启后周期翻倍
                eta_min=1e-8
            )
        elif self.args.scheduler_0 == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif self.args.scheduler_0 == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        elif self.args.scheduler_0 == "MultiStepLR":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.5)
        elif self.args.scheduler_0 == "OneCycleLR":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.args.learning_rate_0 * 10,  # 最大学习率
                epochs=self.args.epochs_0,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,  # 预热阶段占总训练的比例
                anneal_strategy='cos'
            )
        
        elif self.args.scheduler_0 == "CyclicLR":
            scheduler = optim.lr_scheduler.CyclicLR(
                optimizer,
            base_lr=1e-8,
                max_lr=self.args.learning_rate_0,
                step_size_up=2000,
                mode='triangular'
            )
        
        elif self.args.scheduler_0 == "LinearLR":
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.args.epochs_0
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.args.scheduler_0}")

        # 损失函数：适用于分类任务，而T是连续值
        if self.args.loss_function_0 == "MSELoss":
            criterion = nn.MSELoss()
        elif self.args.loss_function_0 == "L1Loss":
            criterion = nn.L1Loss()
        elif self.args.loss_function_0 == "HuberLoss":
            criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.args.loss_function_0}")

        train_dataset = TensorDataset(torch.FloatTensor(np.concatenate((self.args.x_scaled_0, self.args.y_scaled_0), axis=1)), torch.FloatTensor(self.args.T_array_0))
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size_0, shuffle=True)
    
        print("Starting training with adaptive optimizer...")
        print(f"Device: {self.args.device_0}")
        print(f"Use DataParallel: {self.args.use_data_parallel_0}")
        print(f"GPU Device IDs: {self.args.gpu_device_ids_0}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"Batch size: {self.args.batch_size_0}")
        print(f"Epochs: {self.args.epochs_0}")
        print(f"Initial learning rate: {self.args.learning_rate_0}")
        print(f"Optimizer: {type(optimizer).__name__}")
        print(f"Scheduler: {type(scheduler).__name__}")
        print(f"Loss function: {self.args.loss_function_0}")
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
        patience = self.args.early_stopping_patience_0  # 使用早期停止耐心值
        patience_counter = 0

        for epoch in tqdm(range(self.args.epochs_0)):
            if isinstance(T_model, nn.DataParallel):
                T_model.module.train()
            else:
                T_model.train()
            total_loss = 0
            epoch_predictions = []
            epoch_targets = []

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.args.device_0)
                targets = targets.to(self.args.device_0)

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
                if self.args.scheduler_0 in ["OneCycleLR", "CyclicLR"]:
                    scheduler.step()

                total_loss += loss.item()
            
                # Collect predictions and targets
                # logging every 10 batches
                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{self.args.epochs_0}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, LR: {current_lr:.2e}")
                epoch_predictions.extend(outputs.squeeze().cpu().detach().numpy())
                epoch_targets.extend(targets.squeeze().cpu().detach().numpy())

                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{self.args.epochs_0}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, LR: {current_lr:.2e}")

            avg_loss = total_loss / len(train_loader)
        
            # 学习率调度器步进：根据类型调用
            if self.args.scheduler_0 == "ReduceLROnPlateau":
                scheduler.step(avg_loss)
            elif self.args.scheduler_0 == "OneCycleLR":
                # OneCycleLR需要在每个batch后调用
                pass  # 在batch循环中调用
            elif self.args.scheduler_0 == "CyclicLR":
                # CyclicLR需要在每个batch后调用
                pass  # 在batch循环中调用
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        
            # Calculate metrics for this epoch
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
                model_dir = f'../results/best_model_0_{self.args.optimizer_0}_{self.args.scheduler_0}_{self.args.loss_function_0}'
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
            
            print(f"Epoch {epoch+1}/{self.args.epochs_0}, Avg Loss: {avg_loss:.6f}, Avg R2: {avg_r2:.6f}, Avg MSE: {avg_mse:.6f}, Avg MAE: {avg_mae:.6f}")
            print(f"Learning Rate: {current_lr:.2e}, Best Loss: {best_loss:.6f}, Patience: {patience_counter}/{patience}")

            # Regular visualization：修正路径中的参数名
            if (epoch + 1) % 10 == 0:
                save_dir = f'../results/visualizations/0_{self.args.optimizer_0}_{self.args.scheduler_0}_{self.args.loss_function_0}/epoch_{epoch+1}'
                self.plot_prediction_analysis(epoch_predictions, epoch_targets, epoch + 1, save_dir)
                self.plot_training_metrics(history_loss, history_metrics, history_lr, save_dir)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs")
                break

        # Final visualization
        print("Plotting final training results...")
        final_save_dir = f'../results/visualizations/0_{self.args.optimizer_0}_{self.args.scheduler_0}_{self.args.loss_function_0}/epoch_{epoch+1}'
        self.plot_training_metrics(history_loss, history_metrics, history_lr, final_save_dir)
    
        # Save training history：确保目录存在
        history_dir = f'../results/training_history_0_{self.args.optimizer_0}_{self.args.scheduler_0}_{self.args.loss_function_0}'
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
    warnings.filterwarnings('ignore')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    start_time = datetime.now()
    print(f"Training start time: {start_time}")
    trainer0 = Trainer0()
    history_loss, history_metrics, history_lr = trainer0.train()
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"Training end time: {end_time}")
    print(f"Total training duration: {training_duration}")
    
    print(f'Final loss: {history_loss[-1]:.6f}')
    print(f'Final learning rate: {history_lr[-1]:.2e}')
    print(f"Training completed! All visualization results saved to ../results/visualizations/")
    print(f"Training history saved to ../results/training_history_0/")