import torch
import torch.optim as optim
from typing import Dict, Any


def build_scheduler(name: str, optimizer: optim.Optimizer, config: Dict[str, Any]):
    """
    Build a learning rate scheduler based on the given name.
    
    Args:
        name: Name of the scheduler ('cosine', 'step', 'exponential', 'plateau')
        optimizer: The optimizer to schedule
        config: Configuration dictionary for the scheduler
    
    Returns:
        Scheduler instance
    """
    if name == 'cosine':
        T_max = config.get('T_max', 100)
        eta_min = config.get('eta_min', 1e-6)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif name == 'step':
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif name == 'exponential':
        gamma = config.get('gamma', 0.95)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif name == 'plateau':
        mode = config.get('mode', 'min')
        factor = config.get('factor', 0.1)
        patience = config.get('patience', 10)
        threshold = config.get('threshold', 1e-4)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {name}. Available schedulers: 'cosine', 'step', 'exponential', 'plateau'")