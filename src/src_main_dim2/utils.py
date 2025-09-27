import torch.optim as optim
import torch.nn as nn

# 优化器
class Optimizer:
    def __init__(self, optimizer_type, learning_rate, weight_decay):
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def get_optimizer(self, model):
        if self.optimizer_type == "AdamW":
            return optim.AdamW(model.parameters(), 
                               lr=self.learning_rate, 
                               weight_decay=self.weight_decay,
                               betas=(0.9, 0.999),
                               eps=1e-8)
        elif self.optimizer_type == "Adam":
            return optim.Adam(model.parameters(), 
                               lr=self.learning_rate, 
                               weight_decay=self.weight_decay,
                               betas=(0.9, 0.999),
                               eps=1e-8)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")


# 学习率调度器
class Scheduler:
    def __init__(self, scheduler_type, learning_rate, weight_decay, patience, T_max, T_0, T_mult, gamma, step_size, milestones, max_lr, epochs, steps_per_epoch, pct_start, anneal_strategy, base_lr, start_factor, end_factor, total_iters):
        self.scheduler_type = scheduler_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.T_max = T_max
        self.T_0 = T_0
        self.T_mult = T_mult
        self.gamma = gamma
        self.step_size = step_size
        self.milestones = milestones
        self.max_lr = max_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.base_lr = base_lr
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters

    def get_scheduler(self, optimizer):
        if self.scheduler_type == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         factor=0.5, 
                                                         patience=self.patience, 
                                                         verbose=True, 
                                                         min_lr=1e-8)
        elif self.scheduler_type == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=1e-8)
        elif self.scheduler_type == "CosineAnnealingWarmRestarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=1e-8)
        elif self.scheduler_type == "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif self.scheduler_type == "StepLR":
            return optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=0.5)
        elif self.scheduler_type == "MultiStepLR":
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.5)
        elif self.scheduler_type == "OneCycleLR":
            return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, pct_start=0.3, anneal_strategy='cos')
        elif self.scheduler_type == "CyclicLR":
            return optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.base_lr, max_lr=self.max_lr, step_size_up=self.step_size, mode='triangular')
        elif self.scheduler_type == "LinearLR":
            return optim.lr_scheduler.LinearLR(optimizer, start_factor=self.start_factor, end_factor=self.end_factor, total_iters=self.total_iters)
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_type}")




# 损失函数
class LossFunction:
    def __init__(self, loss_function_type):
        self.loss_function_type = loss_function_type

    def get_loss_function(self):
        if self.loss_function_type == "MSELoss":
            return nn.MSELoss()
        elif self.loss_function_type == "L1Loss":
            return nn.L1Loss()
        elif self.loss_function_type == "HuberLoss":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function_type}")
        

if __name__ == "__main__":
    pass