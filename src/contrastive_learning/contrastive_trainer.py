"""
对比学习训练器
实现对比学习的两阶段训练：预训练和微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from tqdm import tqdm
import time
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import Logger
from src.utils.optimizer import build_optimizer
from src.utils.scheduler import build_scheduler
from src.utils.Metrics import MetricsManager
from .contrastive_model import create_contrastive_model
from .contrastive_loss import CombinedContrastiveLoss
from .augmentation import create_augmentation_pipeline


class ContrastiveTrainer:
    """对比学习训练器"""
    
    def __init__(self, config: Dict[str, Any], model: nn.Module, 
                 dataloader_train: DataLoader, dataloader_eval: DataLoader,
                 logger: Logger, save_dir: str):
        self.config = config
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_eval = dataloader_eval
        self.logger = logger
        self.save_dir = save_dir
        
        # 训练配置
        self.train_config = config.get('ContrastiveTrain', {})
        self.device = torch.device(self.train_config.get('device', 'cuda:2'))
        
        # 将模型移到设备
        self.model.to(self.device)
        
        # 创建数据增强器
        self.augmenter = create_augmentation_pipeline(config.get('Augmentation', {}))
        
        # 创建损失函数
        self.loss_fn = CombinedContrastiveLoss(config)
        
        # 创建优化器和调度器
        self._build_optimizer()
        self._build_scheduler()
        
        # 创建指标管理器
        self.metrics_manager = MetricsManager(logger, save_dir)
        
        # 训练状态
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        
        # SwanLab集成
        self.use_swanlab = config.get('swanlab', {}).get('use_swanlab', False)
        if self.use_swanlab:
            self._init_swanlab(config)
            
    def _build_optimizer(self):
        """构建优化器"""
        optimizer_name = self.train_config.get('optimizer', 'adam')
        optimizer_config = self.train_config.get('optimizer_config', {})
        
        # 根据训练阶段调整学习率
        if self.model.training_mode == 'pretrain':
            lr = self.train_config.get('pretrain_lr', 1e-3)
        else:
            lr = self.train_config.get('finetune_lr', 1e-4)
            
        optimizer_config['lr'] = lr
        
        self.optimizer = build_optimizer(optimizer_name, self.model.parameters(), optimizer_config)
        self.logger.info(f"Optimizer created: {optimizer_name}, lr: {lr}")
        
    def _build_scheduler(self):
        """构建学习率调度器"""
        scheduler_name = self.train_config.get('scheduler', 'cosine')
        scheduler_config = self.train_config.get('scheduler_config', {})
        
        self.scheduler = build_scheduler(scheduler_name, self.optimizer, scheduler_config)
        self.logger.info(f"Scheduler created: {scheduler_name}")
        
    def _init_swanlab(self, config: Dict[str, Any]):
        """初始化SwanLab"""
        try:
            import swanlab
            
            swanlab_config = config.get('swanlab', {})
            experiment_name = swanlab_config.get('experiment_name', f"contrastive_experiment_{int(time.time())}")
            project_name = swanlab_config.get('project_name', 'sci2_contrastive')
            description = swanlab_config.get('description', 'Contrastive learning enhanced predictor')
            
            self.swanlab_run = swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                description=description,
                config={
                    'model_name': config.get('ContrastiveModel', {}).get('name', 'contrastive_model'),
                    'base_model': config.get('ContrastiveModel', {}).get('base_model', 'mlp'),
                    'training_mode': self.model.training_mode,
                    'epochs': self.train_config.get('epochs', 1000),
                    'batch_size': self.train_config.get('batch_size', 64),
                    'optimizer': self.train_config.get('optimizer', 'adam'),
                    'device': str(self.device),
                }
            )
            
            self.logger.info(f"SwanLab initialized: {experiment_name}")
            
        except ImportError:
            self.logger.warning("SwanLab not available")
            self.use_swanlab = False
        except Exception as e:
            self.logger.error(f"Failed to initialize SwanLab: {e}")
            self.use_swanlab = False
            
    def _log_to_swanlab(self, metrics: Dict[str, Any], step: int):
        """记录指标到SwanLab"""
        if not self.use_swanlab:
            return
            
        try:
            import swanlab
            
            log_data = {}
            
            # 记录损失
            if 'total_loss' in metrics:
                log_data['loss/total_loss'] = metrics['total_loss']
            if 'contrastive_loss' in metrics:
                log_data['loss/contrastive_loss'] = metrics['contrastive_loss']
            if 'prediction_loss' in metrics:
                log_data['loss/prediction_loss'] = metrics['prediction_loss']
            if 'physics_loss' in metrics:
                log_data['loss/physics_loss'] = metrics['physics_loss']
                
            # 记录学习率
            if hasattr(self.scheduler, 'get_last_lr'):
                log_data['learning_rate'] = self.scheduler.get_last_lr()[0]
                
            # 记录训练步数
            log_data['step'] = step
            log_data['epoch'] = self.current_epoch
            
            swanlab.log(log_data, step=step)
            
        except Exception as e:
            self.logger.error(f"Failed to log to SwanLab: {e}")
            
    def pretrain(self) -> Dict[str, Any]:
        """对比学习预训练阶段"""
        self.logger.info("Starting contrastive learning pretraining...")
        
        # 设置为预训练模式
        self.model.set_training_mode('pretrain')
        self._build_optimizer()  # 重新构建优化器
        
        pretrain_epochs = self.train_config.get('pretrain_epochs', 200)
        self.model.train()
        
        for epoch in tqdm(range(pretrain_epochs), desc='Pretraining'):
            self.current_epoch = epoch
            epoch_losses = []
            
            for batch_idx, (batch_data, batch_target) in enumerate(tqdm(self.dataloader_train, desc=f'Pretrain Epoch {epoch}')):
                self.current_step += 1
                
                # 移动到设备
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)
                
                # 生成增强视图
                views = self.augmenter(batch_data, batch_target, num_views=2)
                anchor_data, anchor_target = views[0]
                positive_data, positive_target = views[1]
                
                # 前向传播
                self.optimizer.zero_grad()
                
                # 获取锚点和正样本的特征
                anchor_outputs = self.model(anchor_data, mode='contrastive')
                positive_outputs = self.model(positive_data, mode='contrastive')
                
                # 计算对比学习损失
                contrastive_loss = self.loss_fn.contrastive_loss(
                    anchor_outputs['projections'], 
                    positive_outputs['projections']
                )
                
                # 反向传播
                contrastive_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_losses.append(contrastive_loss.item())
                
                # 记录指标
                if self.current_step % self.train_config.get('log_interval', 10) == 0:
                    self.logger.info(f"Pretrain Epoch {epoch}, Step {self.current_step}, "
                                   f"Contrastive Loss: {contrastive_loss.item():.6f}")
                    
                    # 记录到SwanLab
                    if self.use_swanlab:
                        self._log_to_swanlab({'contrastive_loss': contrastive_loss.item()}, self.current_step)
                        
            # 记录epoch平均损失
            avg_loss = np.mean(epoch_losses)
            self.logger.info(f"Pretrain Epoch {epoch} completed, Average Loss: {avg_loss:.6f}")
            
            # 保存模型
            if epoch % self.train_config.get('save_freq', 50) == 0:
                self._save_model(epoch, 'pretrain')
                
        self.logger.info("Pretraining completed!")
        return {'pretrain_loss': avg_loss}
        
    def finetune(self) -> Dict[str, Any]:
        """微调阶段"""
        self.logger.info("Starting finetuning...")
        
        # 设置为微调模式
        self.model.set_training_mode('finetune')
        self._build_optimizer()  # 重新构建优化器
        
        finetune_epochs = self.train_config.get('finetune_epochs', 800)
        self.model.train()
        
        for epoch in tqdm(range(finetune_epochs), desc='Finetuning'):
            self.current_epoch = epoch
            epoch_losses = []
            
            for batch_idx, (batch_data, batch_target) in enumerate(tqdm(self.dataloader_train, desc=f'Finetune Epoch {epoch}')):
                self.current_step += 1
                
                # 移动到设备
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                
                # 预测模式
                outputs = self.model(batch_data, mode='predict')
                predictions = outputs['predictions']
                
                # 计算预测损失
                prediction_loss = self.loss_fn.prediction_loss(predictions, batch_target)
                
                # 计算物理约束损失（如果启用）
                physics_loss = torch.tensor(0.0, device=self.device)
                if self.train_config.get('is_pinn', False):
                    physics_loss = self.loss_fn.physics_loss.compute_physics_loss(predictions, batch_target)
                    
                # 总损失
                total_loss = (self.loss_fn.prediction_weight * prediction_loss + 
                             self.loss_fn.physics_weight * physics_loss)
                
                # 反向传播
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_losses.append(total_loss.item())
                
                # 记录指标
                if self.current_step % self.train_config.get('log_interval', 10) == 0:
                    self.logger.info(f"Finetune Epoch {epoch}, Step {self.current_step}, "
                                   f"Total Loss: {total_loss.item():.6f}, "
                                   f"Prediction Loss: {prediction_loss.item():.6f}, "
                                   f"Physics Loss: {physics_loss.item():.6f}")
                    
                    # 记录到SwanLab
                    if self.use_swanlab:
                        metrics = {
                            'total_loss': total_loss.item(),
                            'prediction_loss': prediction_loss.item(),
                            'physics_loss': physics_loss.item()
                        }
                        self._log_to_swanlab(metrics, self.current_step)
                        
            # 记录epoch平均损失
            avg_loss = np.mean(epoch_losses)
            self.logger.info(f"Finetune Epoch {epoch} completed, Average Loss: {avg_loss:.6f}")
            
            # 验证
            if epoch % self.train_config.get('eval_freq', 10) == 0:
                val_metrics = self.validate()
                self.logger.info(f"Validation metrics: {val_metrics}")
                
            # 保存最佳模型
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save_model(epoch, 'best')
                
            # 定期保存
            if epoch % self.train_config.get('save_freq', 50) == 0:
                self._save_model(epoch, 'checkpoint')
                
        self.logger.info("Finetuning completed!")
        return {'finetune_loss': avg_loss, 'best_loss': self.best_loss}
        
    def validate(self) -> Dict[str, Any]:
        """验证模型"""
        self.model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_data, batch_target in tqdm(self.dataloader_eval, desc='Validating'):
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)
                
                # 前向传播
                outputs = self.model(batch_data, mode='predict')
                predictions = outputs['predictions']
                
                # 计算损失
                loss = self.loss_fn.prediction_loss(predictions, batch_target)
                val_losses.append(loss.item())
                
                # 收集预测结果
                val_predictions.append(predictions.cpu())
                val_targets.append(batch_target.cpu())
                
        # 计算指标
        val_predictions = torch.cat(val_predictions, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        
        # 使用指标管理器计算详细指标
        metrics = self.metrics_manager.calculate_metrics(
            val_predictions, val_targets, 
            torch.tensor(np.mean(val_losses)),
            torch.tensor(0.0),  # physics_loss
            torch.tensor(np.mean(val_losses)),  # original_loss
            step=self.current_step
        )
        
        self.model.train()
        return metrics
        
    def _save_model(self, epoch: int, suffix: str = ''):
        """保存模型"""
        save_path = Path(self.save_dir) / 'model'
        save_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"contrastive_model_epoch_{epoch}_{suffix}.pth"
        if suffix:
            filename = f"contrastive_model_{suffix}.pth"
            
        torch.save({
            'epoch': epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }, save_path / filename)
        
        self.logger.info(f"Model saved: {save_path / filename}")
        
    def train(self) -> Dict[str, Any]:
        """完整训练流程"""
        self.logger.info("Starting contrastive learning training...")
        
        # 预训练阶段
        pretrain_results = self.pretrain()
        
        # 微调阶段
        finetune_results = self.finetune()
        
        # 最终验证
        final_metrics = self.validate()
        
        # 保存最终模型
        self._save_model(self.current_epoch, 'final')
        
        # 合并结果
        results = {
            **pretrain_results,
            **finetune_results,
            'final_metrics': final_metrics
        }
        
        self.logger.info("Contrastive learning training completed!")
        return results


def test_contrastive_trainer():
    """测试对比学习训练器"""
    # 创建测试数据
    batch_size = 32
    input_dim = 3
    output_dim = 4
    num_samples = 1000
    
    # 生成随机数据
    x = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, output_dim)
    
    # 创建数据集
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建配置
    config = {
        'ContrastiveModel': {
            'name': 'test_model',
            'base_model': 'mlp',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_dim': 64,
            'num_blocks': 2,
            'dropout': 0.1,
            'embedding_dim': 64,
            'projection_dim': 32
        },
        'ContrastiveTrain': {
            'epochs': 10,
            'pretrain_epochs': 5,
            'finetune_epochs': 5,
            'optimizer': 'adam',
            'optimizer_config': {'lr': 1e-3},
            'scheduler': 'cosine',
            'scheduler_config': {},
            'device': 'cpu',
            'log_interval': 1,
            'save_freq': 5,
            'is_pinn': False
        },
        'Augmentation': {
            'noise': {'enabled': True, 'noise_std': 0.01},
            'spatial': {'enabled': True, 'spatial_scale': 0.05}
        },
        'swanlab': {'use_swanlab': False}
    }
    
    # 创建模型
    model = create_contrastive_model(config['ContrastiveModel'])
    
    # 创建日志器
    logger = Logger('test_contrastive_trainer', config)
    
    # 创建训练器
    trainer = ContrastiveTrainer(
        config=config,
        model=model,
        dataloader_train=dataloader,
        dataloader_eval=dataloader,
        logger=logger,
        save_dir='./test_outputs'
    )
    
    # 运行训练
    results = trainer.train()
    
    print("Contrastive trainer test completed!")
    print(f"Results: {results}")


if __name__ == "__main__":
    test_contrastive_trainer()
