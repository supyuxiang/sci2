"""
对抗学习训练器
实现生成器、判别器和预测器的交替训练
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
from .adversarial_models import create_adversarial_model
from .adversarial_losses import CombinedAdversarialLoss
from .adversarial_augmentation import create_adversarial_augmentation_pipeline


class AdversarialTrainer:
    """对抗学习训练器"""
    
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
        self.train_config = config.get('AdversarialTrain', {})
        self.device = torch.device(self.train_config.get('device', 'cuda:2'))
        
        # 将模型移到设备
        self.model.to(self.device)
        
        # 创建数据增强器
        self.augmenter = create_adversarial_augmentation_pipeline(config)
        
        # 创建损失函数
        self.loss_fn = CombinedAdversarialLoss(config)
        
        # 创建优化器和调度器
        self._build_optimizers()
        self._build_schedulers()
        
        # 创建指标管理器
        self.metrics_manager = MetricsManager(logger, save_dir)
        
        # 训练状态
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        
        # GAN训练参数
        self.gan_config = self.train_config.get('gan_training', {})
        self.d_steps = self.gan_config.get('d_steps', 1)
        self.g_steps = self.gan_config.get('g_steps', 1)
        
        # SwanLab集成
        self.use_swanlab = config.get('swanlab', {}).get('use_swanlab', False)
        if self.use_swanlab:
            self._init_swanlab(config)
            
    def _build_optimizers(self):
        """构建优化器"""
        optimizer_name = self.train_config.get('optimizer', 'adam')
        optimizer_config = self.train_config.get('optimizer_config', {})
        
        # 根据训练阶段调整学习率
        if self.model.training_mode == 'pretrain':
            lr = self.train_config.get('pretrain_lr', 1e-3)
        elif self.model.training_mode == 'adversarial':
            lr = self.train_config.get('adversarial_lr', 1e-4)
        else:  # finetune
            lr = self.train_config.get('finetune_lr', 1e-5)
            
        optimizer_config['lr'] = lr
        
        # 预测器优化器
        self.predictor_optimizer = build_optimizer(
            optimizer_name, self.model.predictor.parameters(), optimizer_config
        )
        
        # 生成器优化器
        self.generator_optimizer = build_optimizer(
            optimizer_name, self.model.generator.parameters(), optimizer_config
        )
        
        # 判别器优化器
        self.discriminator_optimizer = build_optimizer(
            optimizer_name, self.model.discriminator.parameters(), optimizer_config
        )
        
        self.logger.info(f"Optimizers created: {optimizer_name}, lr: {lr}")
        
    def _build_schedulers(self):
        """构建学习率调度器"""
        scheduler_name = self.train_config.get('scheduler', 'cosine')
        scheduler_config = self.train_config.get('scheduler_config', {})
        
        self.predictor_scheduler = build_scheduler(
            scheduler_name, self.predictor_optimizer, scheduler_config
        )
        self.generator_scheduler = build_scheduler(
            scheduler_name, self.generator_optimizer, scheduler_config
        )
        self.discriminator_scheduler = build_scheduler(
            scheduler_name, self.discriminator_optimizer, scheduler_config
        )
        
        self.logger.info(f"Schedulers created: {scheduler_name}")
        
    def _init_swanlab(self, config: Dict[str, Any]):
        """初始化SwanLab"""
        try:
            import swanlab
            
            swanlab_config = config.get('swanlab', {})
            experiment_name = swanlab_config.get('experiment_name', f"adversarial_experiment_{int(time.time())}")
            project_name = swanlab_config.get('project_name', 'sci2_adversarial')
            description = swanlab_config.get('description', 'Adversarial learning enhanced predictor')
            
            self.swanlab_run = swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                description=description,
                config={
                    'model_name': config.get('AdversarialModel', {}).get('name', 'adversarial_model'),
                    'base_model': config.get('AdversarialModel', {}).get('base_model', 'mlp'),
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
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    log_data[f'loss/{key}'] = value.item()
                else:
                    log_data[f'loss/{key}'] = value
                
            # 记录学习率
            if hasattr(self.predictor_scheduler, 'get_last_lr'):
                log_data['learning_rate/predictor'] = self.predictor_scheduler.get_last_lr()[0]
            if hasattr(self.generator_scheduler, 'get_last_lr'):
                log_data['learning_rate/generator'] = self.generator_scheduler.get_last_lr()[0]
            if hasattr(self.discriminator_scheduler, 'get_last_lr'):
                log_data['learning_rate/discriminator'] = self.discriminator_scheduler.get_last_lr()[0]
                
            # 记录训练步数
            log_data['step'] = step
            log_data['epoch'] = self.current_epoch
            
            swanlab.log(log_data, step=step)
            
        except Exception as e:
            self.logger.error(f"Failed to log to SwanLab: {e}")
            
    def pretrain(self) -> Dict[str, Any]:
        """预训练阶段：只训练预测器"""
        self.logger.info("Starting pretraining...")
        
        # 设置为预训练模式
        self.model.set_training_mode('pretrain')
        self._build_optimizers()  # 重新构建优化器
        
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
                
                # 前向传播
                self.predictor_optimizer.zero_grad()
                
                # 预测模式
                outputs = self.model(batch_data, mode='predict')
                predictions = outputs['predictions']
                
                # 计算预测损失
                prediction_loss = self.loss_fn.prediction_loss(predictions, batch_target)
                
                # 计算物理约束损失（如果启用）
                physics_loss = torch.tensor(0.0, device=self.device)
                if self.train_config.get('is_pinn', False):
                    physics_loss = self.loss_fn.physics_loss.compute_physics_loss(predictions, batch_target, batch_data)
                    
                # 总损失
                total_loss = prediction_loss + self.loss_fn.physics_weight * physics_loss
                
                # 反向传播
                total_loss.backward()
                self.predictor_optimizer.step()
                self.predictor_scheduler.step()
                
                epoch_losses.append(total_loss.item())
                
                # 记录指标
                if self.current_step % self.train_config.get('log_interval', 10) == 0:
                    self.logger.info(f"Pretrain Epoch {epoch}, Step {self.current_step}, "
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
            self.logger.info(f"Pretrain Epoch {epoch} completed, Average Loss: {avg_loss:.6f}")
            
            # 保存模型
            if epoch % self.train_config.get('save_freq', 50) == 0:
                self._save_model(epoch, 'pretrain')
                
        self.logger.info("Pretraining completed!")
        return {'pretrain_loss': avg_loss}
        
    def adversarial_train(self) -> Dict[str, Any]:
        """对抗训练阶段：训练生成器、判别器和预测器"""
        self.logger.info("Starting adversarial training...")
        
        # 设置为对抗训练模式
        self.model.set_training_mode('adversarial')
        self._build_optimizers()  # 重新构建优化器
        
        adversarial_epochs = self.train_config.get('adversarial_epochs', 600)
        self.model.train()
        
        for epoch in tqdm(range(adversarial_epochs), desc='Adversarial Training'):
            self.current_epoch = epoch
            epoch_losses = {'generator': [], 'discriminator': [], 'predictor': []}
            
            for batch_idx, (batch_data, batch_target) in enumerate(tqdm(self.dataloader_train, desc=f'Adversarial Epoch {epoch}')):
                self.current_step += 1
                
                # 移动到设备
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)
                
                # 生成对抗样本
                with torch.no_grad():
                    adversarial_data = self.model.generate_adversarial_samples(
                        batch_data, batch_target, attack_type='fgsm'
                    )
                
                # 训练判别器
                for _ in range(self.d_steps):
                    self.discriminator_optimizer.zero_grad()
                    
                    # 真实样本
                    real_outputs = self.model(batch_data, mode='discriminate')
                    real_predictions = real_outputs['predictions']
                    real_discriminator_output = real_outputs['discriminator_output']
                    
                    # 生成样本
                    generated_outputs = self.model(batch_data, mode='generate')
                    generated = generated_outputs['generated']
                    fake_discriminator_output = self.model.discriminator(batch_data, generated)
                    
                    # 计算判别器损失
                    discriminator_losses = self.loss_fn.compute_discriminator_loss(
                        real_discriminator_output, fake_discriminator_output,
                        batch_data, generated, self.model.discriminator
                    )
                    
                    discriminator_loss = discriminator_losses['total_discriminator_loss']
                    discriminator_loss.backward()
                    self.discriminator_optimizer.step()
                    
                    epoch_losses['discriminator'].append(discriminator_loss.item())
                
                # 训练生成器
                for _ in range(self.g_steps):
                    self.generator_optimizer.zero_grad()
                    
                    # 生成样本
                    generated_outputs = self.model(batch_data, mode='generate')
                    generated = generated_outputs['generated']
                    fake_discriminator_output = self.model.discriminator(batch_data, generated)
                    
                    # 计算生成器损失
                    generator_losses = self.loss_fn.compute_generator_loss(
                        fake_discriminator_output, batch_target, generated
                    )
                    
                    generator_loss = generator_losses['total_generator_loss']
                    generator_loss.backward()
                    self.generator_optimizer.step()
                    
                    epoch_losses['generator'].append(generator_loss.item())
                
                # 训练预测器
                self.predictor_optimizer.zero_grad()
                
                # 原始预测
                original_outputs = self.model(batch_data, mode='predict')
                original_predictions = original_outputs['predictions']
                
                # 对抗预测
                adversarial_outputs = self.model(adversarial_data, mode='predict')
                adversarial_predictions = adversarial_outputs['predictions']
                
                # 计算预测器损失
                predictor_losses = self.loss_fn.compute_predictor_loss(
                    original_predictions, batch_target, adversarial_predictions, batch_data
                )
                
                predictor_loss = predictor_losses['total_predictor_loss']
                predictor_loss.backward()
                self.predictor_optimizer.step()
                
                epoch_losses['predictor'].append(predictor_loss.item())
                
                # 更新调度器
                self.discriminator_scheduler.step()
                self.generator_scheduler.step()
                self.predictor_scheduler.step()
                
                # 记录指标
                if self.current_step % self.train_config.get('log_interval', 10) == 0:
                    self.logger.info(f"Adversarial Epoch {epoch}, Step {self.current_step}, "
                                   f"Generator Loss: {generator_loss.item():.6f}, "
                                   f"Discriminator Loss: {discriminator_loss.item():.6f}, "
                                   f"Predictor Loss: {predictor_loss.item():.6f}")
                    
                    # 记录到SwanLab
                    if self.use_swanlab:
                        metrics = {
                            'generator_loss': generator_loss.item(),
                            'discriminator_loss': discriminator_loss.item(),
                            'predictor_loss': predictor_loss.item(),
                            'total_loss': generator_loss.item() + discriminator_loss.item() + predictor_loss.item()
                        }
                        self._log_to_swanlab(metrics, self.current_step)
                        
            # 记录epoch平均损失
            avg_generator_loss = np.mean(epoch_losses['generator'])
            avg_discriminator_loss = np.mean(epoch_losses['discriminator'])
            avg_predictor_loss = np.mean(epoch_losses['predictor'])
            
            self.logger.info(f"Adversarial Epoch {epoch} completed, "
                           f"Generator Loss: {avg_generator_loss:.6f}, "
                           f"Discriminator Loss: {avg_discriminator_loss:.6f}, "
                           f"Predictor Loss: {avg_predictor_loss:.6f}")
            
            # 验证
            if epoch % self.train_config.get('eval_freq', 10) == 0:
                val_metrics = self.validate()
                self.logger.info(f"Validation metrics: {val_metrics}")
                
            # 保存最佳模型
            total_avg_loss = avg_generator_loss + avg_discriminator_loss + avg_predictor_loss
            if total_avg_loss < self.best_loss:
                self.best_loss = total_avg_loss
                self._save_model(epoch, 'best')
                
            # 定期保存
            if epoch % self.train_config.get('save_freq', 50) == 0:
                self._save_model(epoch, 'checkpoint')
                
        self.logger.info("Adversarial training completed!")
        return {
            'generator_loss': avg_generator_loss,
            'discriminator_loss': avg_discriminator_loss,
            'predictor_loss': avg_predictor_loss,
            'best_loss': self.best_loss
        }
        
    def finetune(self) -> Dict[str, Any]:
        """微调阶段：只训练预测器"""
        self.logger.info("Starting finetuning...")
        
        # 设置为微调模式
        self.model.set_training_mode('finetune')
        self._build_optimizers()  # 重新构建优化器
        
        finetune_epochs = self.train_config.get('finetune_epochs', 200)
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
                self.predictor_optimizer.zero_grad()
                
                # 预测模式
                outputs = self.model(batch_data, mode='predict')
                predictions = outputs['predictions']
                
                # 计算预测损失
                prediction_loss = self.loss_fn.prediction_loss(predictions, batch_target)
                
                # 计算物理约束损失（如果启用）
                physics_loss = torch.tensor(0.0, device=self.device)
                if self.train_config.get('is_pinn', False):
                    physics_loss = self.loss_fn.physics_loss.compute_physics_loss(predictions, batch_target, batch_data)
                    
                # 总损失
                total_loss = prediction_loss + self.loss_fn.physics_weight * physics_loss
                
                # 反向传播
                total_loss.backward()
                self.predictor_optimizer.step()
                self.predictor_scheduler.step()
                
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
        
        filename = f"adversarial_model_epoch_{epoch}_{suffix}.pth"
        if suffix:
            filename = f"adversarial_model_{suffix}.pth"
            
        torch.save({
            'epoch': epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'predictor_optimizer_state_dict': self.predictor_optimizer.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'predictor_scheduler_state_dict': self.predictor_scheduler.state_dict(),
            'generator_scheduler_state_dict': self.generator_scheduler.state_dict(),
            'discriminator_scheduler_state_dict': self.discriminator_scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }, save_path / filename)
        
        self.logger.info(f"Model saved: {save_path / filename}")
        
    def train(self) -> Dict[str, Any]:
        """完整训练流程"""
        self.logger.info("Starting adversarial learning training...")
        
        # 预训练阶段
        pretrain_results = self.pretrain()
        
        # 对抗训练阶段
        adversarial_results = self.adversarial_train()
        
        # 微调阶段
        finetune_results = self.finetune()
        
        # 最终验证
        final_metrics = self.validate()
        
        # 保存最终模型
        self._save_model(self.current_epoch, 'final')
        
        # 合并结果
        results = {
            **pretrain_results,
            **adversarial_results,
            **finetune_results,
            'final_metrics': final_metrics
        }
        
        self.logger.info("Adversarial learning training completed!")
        return results


def test_adversarial_trainer():
    """测试对抗学习训练器"""
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
        'AdversarialModel': {
            'name': 'test_model',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_dim': 64,
            'num_blocks': 2,
            'dropout': 0.1,
            'generator': {
                'hidden_dim': 64,
                'num_layers': 3,
                'dropout': 0.1,
                'noise_dim': 16,
                'output_activation': 'tanh'
            },
            'discriminator': {
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.1,
                'output_activation': 'sigmoid'
            },
            'predictor': {
                'hidden_dim': 64,
                'num_layers': 3,
                'dropout': 0.1,
                'output_activation': 'none'
            },
            'adversarial': {
                'epsilon': 0.1,
                'num_adv_steps': 3,
                'adv_lr': 0.01,
                'pgd_steps': 2,
                'fgsm_alpha': 0.01
            }
        },
        'AdversarialTrain': {
            'epochs': 10,
            'pretrain_epochs': 3,
            'adversarial_epochs': 4,
            'finetune_epochs': 3,
            'optimizer': 'adam',
            'optimizer_config': {'lr': 1e-3},
            'scheduler': 'cosine',
            'scheduler_config': {},
            'device': 'cpu',
            'log_interval': 1,
            'save_freq': 5,
            'is_pinn': False,
            'gan_training': {
                'd_steps': 1,
                'g_steps': 1,
                'd_lr': 1e-4,
                'g_lr': 1e-4
            }
        },
        'Augmentation': {
            'enabled': True,
            'noise': {'enabled': True, 'noise_std': 0.01},
            'spatial': {'enabled': True, 'spatial_scale': 0.05}
        },
        'AdversarialAttack': {
            'enabled': True,
            'attack_types': ['fgsm'],
            'fgsm': {'epsilon': 0.1, 'targeted': False}
        },
        'swanlab': {'use_swanlab': False}
    }
    
    # 创建模型
    model = create_adversarial_model(config['AdversarialModel'])
    
    # 创建日志器
    logger = Logger('test_adversarial_trainer', config)
    
    # 创建训练器
    trainer = AdversarialTrainer(
        config=config,
        model=model,
        dataloader_train=dataloader,
        dataloader_eval=dataloader,
        logger=logger,
        save_dir='./test_outputs'
    )
    
    # 运行训练
    results = trainer.train()
    
    print("Adversarial trainer test completed!")
    print(f"Results: {results}")


if __name__ == "__main__":
    test_adversarial_trainer()
