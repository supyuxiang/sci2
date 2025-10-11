import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from typing import Dict,Any,List,Tuple,Optional,Union
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from utils.timer import Timer
from utils.optimizer import build_optimizer
from utils.loss_function import loss_base,PhysicsLoss
from utils.scheduler import build_scheduler
from utils.Metrics import MetricsManager
from utils.logger import Logger


class Trainer:

    '''
    Trainer
    Args:
        config: Dict
        model: nn.Module
        dataloader_train: DataLoader
        logger: Logger

    Attributes:
        dataloader_train: DataLoader
        model: nn.Module
        config: Dict
        logger: Logger
        optimizer: Optimizer
        loss_function: nn.Module
        scheduler: Scheduler
        metrics_manager: MetricsManager
        device: torch.device
    '''

    def __init__(self,config:Dict,model,dataloader_train,dataloader_eval,logger:Logger,save_dir:str):
        self.save_model_dir = save_dir
        self.model = model
        self.logger = logger
        self._set_device(config)
        if bool(self.config.get('swanlab',{}).get('use_swanlab',True)):
            self.init_swanlab(full_config=config)
        self.config = config.get('Train',{})
        self.dataloader_train = dataloader_train
        self.dataloader_eval = dataloader_eval
        self.config_check()
        self._build_optimizer()
        self._build_loss_function()
        self._build_scheduler()
        self._build_metrics_manager()
        self.train()


    def config_check(self):
        '''
        check config
        '''
        assert self.config.get('epochs',None) is not None,'epochs is not set'
        assert self.config.get('optimizer',None) is not None,'optimizer is not set'
        assert self.config.get('loss_function',None) is not None,'loss_function is not set'
        assert self.config.get('scheduler',None) is not None,'scheduler is not set'
        assert self.save_dir is not None,'save_dir is not set'
        assert self.config.get('log_interval',None) is not None,'log_interval is not set'
        assert self.config.get('is_log',None) is not None,'is_log is not set'
        assert self.config.get('is_save',None) is not None,'is_save is not set'
        assert self.config.get('save_freq',None) is not None,'save_freq is not set'
        assert self.config.get('log_dir',None) is not None,'log_dir is not set'
        assert self.config.get('device',None) is not None,'device is not set'
        assert self.config.get('metrics_dir',None) is not None,'metrics_dir is not set'
        assert self.config.get('early_stopping',None) is not None,'early_stopping is not set'
        assert self.config.get('patience',None) is not None,'patience is not set'
        assert self.config.get('min_delta',None) is not None,'min_delta is not set'
        assert self.config.get('save_best_only',None) is not None,'save_best_only is not set'
        assert self.config.get('log_freq',None) is not None,'log_freq is not set'
        assert self.config.get('optimizer_config',None) is not None,'optimizer_config is not set'
        assert self.config.get('loss_function_config',None) is not None,'loss_function_config is not set'
        assert self.config.get('scheduler_config',None) is not None,'scheduler_config is not set'
        assert self.config.get('metrics_dir',None) is not None,'metrics_dir is not set'
        assert self.config.get('is_pinn',None) is not None,'is_pinn is not set'
        assert self.config.get('physics_weight',None) is not None,'physics_weight is not set'
        assert self.config.get('original_loss_weight',None) is not None,'original_loss_weight is not set'
        self.logger.info(f"Config checked, config: {self.config}")
        

    def _set_device(self,config:Dict):
        self.device = config.get('device', 'cpu')
        self.logger.info(f"Device set, device: {self.device}")

    def _build_optimizer(self):
        optimizer_name = self.config.get('optimizer', 'adam')
        self.optimizer = build_optimizer(optimizer_name, self.model.parameters(), self.config.get('optimizer_config',{}))  
        self.logger.info(f"Optimizer set, optimizer: {self.optimizer}")
        

    def _build_loss_function(self):
        loss_function_name = self.config.get('loss_function', 'mse')
        self.loss_function = loss_base(loss_function_name)
        self.logger.info(f"Loss function set, loss_function: {self.loss_function}")

    def _build_scheduler(self):
        scheduler_name = self.config.get('scheduler', 'cosine')
        self.scheduler = build_scheduler(scheduler_name, self.optimizer, self.config.get('scheduler_config',{}))
        self.logger.info(f"Scheduler set, scheduler: {self.scheduler}")

    def _build_metrics_manager(self):
        self.metrics_manager = MetricsManager(self.logger,self.save_dir)
        self.logger.info(f"Metrics manager set, metrics_manager: {self.metrics_manager}")

    def save_model(self,steps:int):
        '''
        save model
        Args:
            steps: int
        '''
        epochs = self.config.get('epochs',None)
        save_model_path = Path(self.save_dir) / 'model' / f"ep{epochs}_steps_{steps}_{self.model.name}_model.pth"
        save_model_path.parent.mkdir(parents=True,exist_ok=True)

        torch.save(self.model.state_dict(),save_model_path)
        self.logger.info(f"Model saved, save_path: {save_model_path}")
    
    def save_fig(self, metrics: dict[str, Any]):
        '''
        Save training metrics visualization figures
        Args:
            metrics: dict containing training metrics history
        '''
        try:
            # 创建保存目录
            fig_dir = Path(self.save_model_dir) / 'figures'
            fig_dir.mkdir(parents=True, exist_ok=True)
            
            # 设置中文字体支持
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建2x2的子图布局
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Training Metrics Visualization', fontsize=16, fontweight='bold')
            
            # 1. 损失函数曲线 (左上)
            ax1 = axes[0, 0]
            if 'total_loss' in metrics and metrics['total_loss']:
                steps = list(range(len(metrics['total_loss'])))
                ax1.plot(steps, metrics['total_loss'], 'b-', label='Total Loss', linewidth=2)
                
                if 'physics_loss' in metrics and metrics['physics_loss']:
                    ax1.plot(steps, metrics['physics_loss'], 'r-', label='Physics Loss', linewidth=2)
                
                if 'original_loss' in metrics and metrics['original_loss']:
                    ax1.plot(steps, metrics['original_loss'], 'g-', label='Original Loss', linewidth=2)
            
            ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Loss Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')  # 使用对数坐标轴
            
            # 2. R² 指标曲线 (右上)
            ax2 = axes[0, 1]
            if 'r2' in metrics and metrics['r2']:
                # 提取各变量的R²值
                r2_data = metrics['r2']
                if isinstance(r2_data, list) and len(r2_data) > 0:
                    # 获取变量名称（除了'step'）
                    var_names = [k for k in r2_data[0].keys() if k != 'step']
                    
                    for var_name in var_names:
                        r2_values = [item[var_name] for item in r2_data if var_name in item]
                        if r2_values:
                            steps = list(range(len(r2_values)))
                            ax2.plot(steps, r2_values, label=f'R² {var_name}', linewidth=2)
            
            ax2.set_title('R² Score Curves', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('R² Score')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)  # R²通常在0-1之间
            
            # 3. MAE 指标曲线 (左下)
            ax3 = axes[1, 0]
            if 'mae' in metrics and metrics['mae']:
                mae_data = metrics['mae']
                if isinstance(mae_data, list) and len(mae_data) > 0:
                    var_names = [k for k in mae_data[0].keys() if k != 'step']
                    
                    for var_name in var_names:
                        mae_values = [item[var_name] for item in mae_data if var_name in item]
                        if mae_values:
                            steps = list(range(len(mae_values)))
                            ax3.plot(steps, mae_values, label=f'MAE {var_name}', linewidth=2)
            
            ax3.set_title('MAE Curves', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('MAE Value')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')  # 使用对数坐标轴
            
            # 4. RMSE 指标曲线 (右下)
            ax4 = axes[1, 1]
            if 'rmse' in metrics and metrics['rmse']:
                rmse_data = metrics['rmse']
                if isinstance(rmse_data, list) and len(rmse_data) > 0:
                    var_names = [k for k in rmse_data[0].keys() if k != 'step']
                    
                    for var_name in var_names:
                        rmse_values = [item[var_name] for item in rmse_data if var_name in item]
                        if rmse_values:
                            steps = list(range(len(rmse_values)))
                            ax4.plot(steps, rmse_values, label=f'RMSE {var_name}', linewidth=2)
            
            ax4.set_title('RMSE Curves', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Training Steps')
            ax4.set_ylabel('RMSE Value')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')  # 使用对数坐标轴
            
            # 调整子图间距
            plt.tight_layout()
            
            # 保存图片
            timestamp = int(time.time())
            fig_path = fig_dir / f'training_metrics_{timestamp}.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            # 同时保存为PDF格式（矢量图）
            pdf_path = fig_dir / f'training_metrics_{timestamp}.pdf'
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
            
            self.logger.info(f"Training metrics figures saved:")
            self.logger.info(f"  - PNG: {fig_path}")
            self.logger.info(f"  - PDF: {pdf_path}")
            
            # 关闭图形以释放内存
            plt.close(fig)
            
            # 如果启用了SwanLab，也上传图片
            if self.use_swanlab:
                try:
                    import swanlab
                    swanlab.log({"training_metrics_plot": swanlab.Image(str(fig_path))})
                    self.logger.info("Training metrics plot uploaded to SwanLab")
                except Exception as e:
                    self.logger.warning(f"Failed to upload plot to SwanLab: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to save training metrics figures: {e}")
            # 确保即使出错也关闭图形
            try:
                plt.close('all')
            except:
                pass

    def init_swanlab(self,full_config:Dict):
        '''
        Initialize SwanLab for experiment tracking
        '''
        try:
            import swanlab
            
            # 获取实验配置信息
            swanlab_config = full_config.get('swanlab',{})
            experiment_name = swanlab_config.get('experiment_name', f"experiment_{int(time.time())}")
            project_name = swanlab_config.get('project_name', 'sci2')
            description = swanlab_config.get('description', 'Scientific computing experiment with physics-informed neural network')
            
            # 初始化SwanLab
            self.swanlab_run = swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                description=description,
                config={
                    'model_name': full_config.get('Model',{}).get('model_name', 'unknown_model'),
                    'epochs': full_config.get('Train',{}).get('epochs', 'unknown_epochs'),
                    'batch_size': full_config.get('Train',{}).get('batch_size', 'unknown_batch_size'),
                    'optimizer': full_config.get('Train',{}).get('optimizer', 'unknown_optimizer'),
                    'loss_function': full_config.get('Train',{}).get('loss_function', 'unknown_loss_function'),
                    'scheduler': full_config.get('Train',{}).get('scheduler', 'unknown_scheduler'),
                    'learning_rate': full_config.get('Train',{}).get('learning_rate', 'unknown_learning_rate'),
                    'device': self.device,
                }
            )
            
            self.logger.info(f"SwanLab initialized successfully: {experiment_name}")
            self.logger.info(f"Project: {project_name}")
            self.logger.info(f"Description: {description}")
            self.use_swanlab = True
            
        except ImportError:
            self.logger.warning("SwanLab not installed. Please install with: pip install swanlab")
            self.use_swanlab = False
        except Exception as e:
            self.logger.error(f"Failed to initialize SwanLab: {e}")
            self.use_swanlab = False

    def save_swanlab(self,step_metrics:dict[str,Any],step:int):
        '''
        Log metrics to SwanLab for experiment tracking
        Args:
            step_metrics: dict containing metrics for current step
            step: int current training step
        '''
        if not self.use_swanlab:
            return
            
        try:
            import swanlab
            
            # 准备要记录的指标
            log_data = {}
            
            # 记录损失指标
            if 'total_loss' in step_metrics:
                log_data['loss/total_loss'] = step_metrics['total_loss']
            if 'physics_loss' in step_metrics:
                log_data['loss/physics_loss'] = step_metrics['physics_loss']
            if 'original_loss' in step_metrics:
                log_data['loss/original_loss'] = step_metrics['original_loss']
            
            # 记录各变量的R²指标
            if 'r2' in step_metrics and isinstance(step_metrics['r2'], dict):
                for var_name, r2_value in step_metrics['r2'].items():
                    if var_name != 'step' and isinstance(r2_value, (int, float)):
                        log_data[f'r2/{var_name}'] = r2_value
                        
                # 计算平均R²
                r2_values = [v for k, v in step_metrics['r2'].items() 
                           if k != 'step' and isinstance(v, (int, float))]
                if r2_values:
                    log_data['r2/mean'] = np.mean(r2_values)
            
            # 记录各变量的MAE指标
            if 'mae' in step_metrics and isinstance(step_metrics['mae'], dict):
                for var_name, mae_value in step_metrics['mae'].items():
                    if var_name != 'step' and isinstance(mae_value, (int, float)):
                        log_data[f'mae/{var_name}'] = mae_value
                        
                # 计算平均MAE
                mae_values = [v for k, v in step_metrics['mae'].items() 
                            if k != 'step' and isinstance(v, (int, float))]
                if mae_values:
                    log_data['mae/mean'] = np.mean(mae_values)
            
            # 记录各变量的RMSE指标
            if 'rmse' in step_metrics and isinstance(step_metrics['rmse'], dict):
                for var_name, rmse_value in step_metrics['rmse'].items():
                    if var_name != 'step' and isinstance(rmse_value, (int, float)):
                        log_data[f'rmse/{var_name}'] = rmse_value
                        
                # 计算平均RMSE
                rmse_values = [v for k, v in step_metrics['rmse'].items() 
                             if k != 'step' and isinstance(v, (int, float))]
                if rmse_values:
                    log_data['rmse/mean'] = np.mean(rmse_values)
            
            # 记录各变量的MSE指标
            if 'mse' in step_metrics and isinstance(step_metrics['mse'], dict):
                for var_name, mse_value in step_metrics['mse'].items():
                    if var_name != 'step' and isinstance(mse_value, (int, float)):
                        log_data[f'mse/{var_name}'] = mse_value
                        
                # 计算平均MSE
                mse_values = [v for k, v in step_metrics['mse'].items() 
                            if k != 'step' and isinstance(v, (int, float))]
                if mse_values:
                    log_data['mse/mean'] = np.mean(mse_values)
            
            # 记录学习率（如果可用）
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                try:
                    current_lr = self.scheduler.get_last_lr()[0]
                    log_data['learning_rate'] = current_lr
                except:
                    pass
            
            # 记录训练步数
            log_data['step'] = step
            
            # 发送数据到SwanLab
            if log_data:
                swanlab.log(log_data, step=step)
                
                # 只在特定步数记录详细信息
                if step % 100 == 0:
                    self.logger.info(f"SwanLab logged {len(log_data)} metrics at step {step}")
                    
        except ImportError:
            self.logger.warning("SwanLab not available for logging")
        except Exception as e:
            self.logger.error(f"Failed to log to SwanLab: {e}")
            # 不抛出异常，避免影响训练流程

    @Timer
    def train(self):
        '''
        train
        Args:
            dataloader_train: DataLoader
        '''
        epochs = int(self.config.get('epochs',None))
        total_steps = len(self.dataloader_train)*epochs
        step = 0
        is_eval = bool(self.config.get('is_eval',True))
        if is_eval:
            eval_freq = int(self.config.get('eval_freq',10))
            eval_dataloader = self.dataloader_eval
        
        self.logger.info(f"Training started, epochs: {epochs}, total_steps: {total_steps}, dataloader_train: {self.dataloader_train}")
        self.model.train()
        for epoch in tqdm(range(epochs),desc='Training'):
            self.logger.info(f"Epoch {epoch} started")
            start_time = time.time()
            for idx,(batch_data,batch_target) in enumerate(tqdm(self.dataloader_train, desc=f'Epoch {epoch}')):
                step += 1
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch_data)
                loss = self.loss_function(output,batch_target)
                if self.config.get('is_pinn',False):
                    w_physics = float(self.config.get('physics_weight',1.0))
                    w_original = float(self.config.get('original_loss_weight',1.0))
                    #如果权重之和不为一，则强行归一化并提示
                    if w_physics + w_original != 1:
                        self.logger.warning(f"Physics weight and original loss weight do not sum to 1, they are {w_physics} and {w_original}, they will be normalized to {w_physics / (w_physics + w_original)} and {w_original / (w_physics + w_original)}")
                        w_physics = w_physics / (w_physics + w_original)
                        w_original = w_original / (w_physics + w_original)
                        self.logger.info(f"Current Physics weight: {w_physics}, Original weight: {w_original}")
                    physics_loss = PhysicsLoss(output,batch_target).compute_physics_loss()
                    loss = w_physics * physics_loss + w_original * loss
                    self.logger.info(f"Physics loss: {physics_loss}, Original loss: {loss},Mixed loss: {loss}")
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                # 获取当前学习率
                current_lr = self.scheduler.get_last_lr()[0]
                
                # Calculate metrics for this batch, 要改
                is_save_metrics = bool(self.config.get('is_save_metrics',False)) and step % int(self.config.get('save_metrics_freq',10)) == 0
                with torch.no_grad():
                    metrics_current = self.metrics_manager.calculate_metrics(output,batch_target,loss,physics_loss,loss,step=step)
                    self.metrics_manager.update_metrics()
                    if is_save_metrics:
                        save_metrics_dir = Path(self.config.get('metrics_dir',None))
                        save_metrics_path = save_metrics_dir / f'ep{epoch}/{int(self.config.get('epochs',None))}_steps{step}/{total_steps}_{self.model.name}_metrics.yaml'
                        Path(save_metrics_path).parent.mkdir(parents=True,exist_ok=True)
                        self.metrics_manager.save_metrics(save_metrics_path,metrics_current)
                        self.logger.info(f"Metrics saved, save_path: {save_metrics_path}")
                    is_log_metrics = step % 10 ==0
                    if is_log_metrics:
                        self.logger.info(f"Epoch {epoch}, Step {step}, total_loss: {loss.item():.6f}, LR: {current_lr:.6f}, 'metrics_current': {metrics_current}")
                    is_save_model = bool(self.config.get('is_save_model',True)) and step % int(self.config.get('save_model_freq',500)) == 0
                    if is_save_model:
                        save_model_path = Path(self.save_dir) / 'model' / f"ep{epoch}/{int(self.config.get('epochs',None))}_steps{step}/{total_steps}_{self.model.name}_model.pth"
                        self.save_model(step)
                        self.logger.info(f"Model saved, save_path: {save_model_path}")
                    
                    if self.use_swanlab:
                        self.save_swanlab(metrics_current,step)
                    
                    if is_eval:
                        self.validate(step)
                    
            if self.config.get('early_stopping', True):
                loss_history = self.metrics_manager.metrics['total_loss']
                n = len(loss_history)
                patience = int(self.config.get('patience', 5))
                min_delta = float(self.config.get('min_delta', 1e-3))

                is_early_stopping = bool(n > patience and (max(loss_history[-patience:]) - loss_history[-1]) < (-min_delta))
                if is_early_stopping:
                    save_model_path = Path(self.save_dir) / 'model' / f"ep{epoch}/{int(self.config.get('epochs',None))}_steps{step}/{total_steps}_{self.model.name}_model.pth"
                    self.save_model(step)
                    self.logger.info(f"Model saved, save_path: {save_model_path}")
                    self.logger.info(f"Early stopping triggered, epoch: {epoch}")
                    break
        
        metrics_final_step = self.metrics_manager.metrics
        self.logger.info(f"Training completed, metrics_final_step: {metrics_final_step}")

        self.save_fig(self.metrics_manager.metrics)
        
        return_metrics = self.metrics_manager.metrics
        if self.use_swanlab:
            self.save_swanlab(return_metrics,step)
        return return_metrics

    @Timer
    def validate(self, step: int = None):
        '''
        Validate the model on validation dataset
        Args:
            dataloader_val: DataLoader for validation data
            step: int current training step (optional, for SwanLab logging)
        Returns:
            dict: validation results containing predictions and metrics
        '''
        self.model.eval()
        self.model.to(self.device)
        val_batch_size = int(self.config.get('val_batch_size', 50))
        
        # Initialize validation tracking
        val_predictions = []
        val_targets = []
        val_losses = []
        val_physics_losses = []
        val_original_losses = []
        
        self.logger.info("Starting validation...")
        
        with torch.no_grad():
            for batch_idx, (batch_in, batch_out) in enumerate(tqdm(self.dataloader_eval, desc='Validating')):
                # Move data to device
                batch_in = batch_in.to(self.device)
                batch_out = batch_out.to(self.device)
                
                # Forward pass
                output = self.model(batch_in)
                
                # Calculate original loss
                original_loss = self.loss_function(output, batch_out)
                
                # Calculate physics loss if PINN is enabled
                physics_loss = None
                total_loss = original_loss
                
                if bool(self.config.get('is_pinn', False)):
                    try:
                        physics_loss = PhysicsLoss(output, batch_out, phase=getattr(self.model, 'phase', None)).compute_physics_loss()
                        
                        # Get loss weights
                        w_physics = float(self.config.get('physics_weight', 1.0))
                        w_original = float(self.config.get('original_loss_weight', 1.0))

                        # Normalize weights if needed
                        if w_physics + w_original != 1:
                            w_physics = w_physics / (w_physics + w_original)
                            w_original = w_original / (w_physics + w_original)
                        
                        total_loss = w_physics * physics_loss + w_original * original_loss
                        
                        if batch_idx % val_batch_size == 0:  # Log every 10 batches
                            self.logger.info(f"Val Batch {batch_idx}: Physics loss: {physics_loss.item():.6f}, "
                                           f"Original loss: {original_loss.item():.6f}, Total loss: {total_loss.item():.6f}")
                    except Exception as e:
                        self.logger.warning(f"Physics loss calculation failed: {e}")
                        physics_loss = torch.tensor(0.0, device=self.device)
                
                # Store results
                val_predictions.append(output.detach().cpu())
                val_targets.append(batch_out.detach().cpu())
                val_losses.append(total_loss.detach().cpu().item())
                val_original_losses.append(original_loss.detach().cpu().item())
                if physics_loss is not None:
                    val_physics_losses.append(physics_loss.detach().cpu().item())
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(val_predictions, dim=0)
        all_targets = torch.cat(val_targets, dim=0)
        
        # Calculate comprehensive metrics using MetricsManager
        val_metrics = self.metrics_manager.calculate_metrics(
            all_predictions, all_targets, 
            torch.tensor(np.mean(val_losses)),  # total_loss
            torch.tensor(np.mean(val_physics_losses)) if val_physics_losses else torch.tensor(0.0),  # physics_loss
            torch.tensor(np.mean(val_original_losses)),  # original_loss
            step=step if step is not None else 0
        )
        
        # Prepare validation results
        val_results = {
            'predictions': all_predictions,
            'targets': all_targets,
            'total_loss': np.mean(val_losses),
            'physics_loss': np.mean(val_physics_losses) if val_physics_losses else 0.0,
            'original_loss': np.mean(val_original_losses),
            'metrics': val_metrics,
            'num_samples': len(all_predictions)
        }
        
        # Log validation results
        self.logger.info(f"Validation completed:")
        self.logger.info(f"  - Total loss: {val_results['total_loss']:.6f}")
        self.logger.info(f"  - Physics loss: {val_results['physics_loss']:.6f}")
        self.logger.info(f"  - Original loss: {val_results['original_loss']:.6f}")
        self.logger.info(f"  - Number of samples: {val_results['num_samples']}")
        
        # Log detailed metrics
        if 'r2' in val_metrics and isinstance(val_metrics['r2'], dict):
            r2_info = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics['r2'].items() if k != 'step'])
            self.logger.info(f"  - R² scores: {r2_info}")
        
        if 'mae' in val_metrics and isinstance(val_metrics['mae'], dict):
            mae_info = ", ".join([f"{k}: {v:.6f}" for k, v in val_metrics['mae'].items() if k != 'step'])
            self.logger.info(f"  - MAE scores: {mae_info}")
        
        # Log to SwanLab if enabled
        if self.use_swanlab and step is not None:
            try:
                # Prepare validation metrics for SwanLab
                val_log_data = {
                    'val/total_loss': val_results['total_loss'],
                    'val/physics_loss': val_results['physics_loss'],
                    'val/original_loss': val_results['original_loss'],
                    'val/num_samples': val_results['num_samples']
                }
                
                # Add detailed metrics
                if 'r2' in val_metrics and isinstance(val_metrics['r2'], dict):
                    for var_name, r2_value in val_metrics['r2'].items():
                        if var_name != 'step' and isinstance(r2_value, (int, float)):
                            val_log_data[f'val/r2/{var_name}'] = r2_value
                    
                    # Add mean R²
                    r2_values = [v for k, v in val_metrics['r2'].items() 
                               if k != 'step' and isinstance(v, (int, float))]
                    if r2_values:
                        val_log_data['val/r2/mean'] = np.mean(r2_values)
                
                if 'mae' in val_metrics and isinstance(val_metrics['mae'], dict):
                    for var_name, mae_value in val_metrics['mae'].items():
                        if var_name != 'step' and isinstance(mae_value, (int, float)):
                            val_log_data[f'val/mae/{var_name}'] = mae_value
                    
                    # Add mean MAE
                    mae_values = [v for k, v in val_metrics['mae'].items() 
                                if k != 'step' and isinstance(v, (int, float))]
                    if mae_values:
                        val_log_data['val/mae/mean'] = np.mean(mae_values)
                
                if 'rmse' in val_metrics and isinstance(val_metrics['rmse'], dict):
                    for var_name, rmse_value in val_metrics['rmse'].items():
                        if var_name != 'step' and isinstance(rmse_value, (int, float)):
                            val_log_data[f'val/rmse/{var_name}'] = rmse_value
                    
                    # Add mean RMSE
                    rmse_values = [v for k, v in val_metrics['rmse'].items() 
                                 if k != 'step' and isinstance(v, (int, float))]
                    if rmse_values:
                        val_log_data['val/rmse/mean'] = np.mean(rmse_values)
                
                # Log to SwanLab
                import swanlab
                swanlab.log(val_log_data, step=step)
                self.logger.info(f"Validation metrics logged to SwanLab at step {step}")
                
            except Exception as e:
                self.logger.error(f"Failed to log validation metrics to SwanLab: {e}")
        
        return val_results
 

    
    


    
        
        
        