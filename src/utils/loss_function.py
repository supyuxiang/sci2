import torch
import torch.nn as nn
from typing import Dict,Any


def build_loss_function(name: str):
    """
    Build a loss function based on the given name.
    
    Args:
        name: Name of the loss function ('mse', 'mae', 'crossentropy')
    
    Returns:
        Loss function instance
    """
    if name == 'mse':
        return nn.MSELoss()
    elif name == 'mae':
        return nn.L1Loss()
    elif name == 'crossentropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid loss function: {name}. Available loss functions: 'mse', 'mae', 'crossentropy'")