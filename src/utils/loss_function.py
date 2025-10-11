import torch
import torch.nn as nn
from typing import Dict,Any
import CoolProp as CP  


def loss_base(name: str):
    """
    Build a loss function based on the given name.
    
    Args:
        name: Name of the loss function ('mse', 'mae', 'crossentropy')
    
    Returns:
        Loss function instance
    """
    if name == 'mse':
        return nn.MSELoss(reduction='mean')
    elif name == 'mae':
        return nn.L1Loss(reduction='mean')
    elif name == 'crossentropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid loss function: {name}. Available loss functions: 'mse', 'mae', 'crossentropy'")


def build_loss_function(config: Dict[str, Any]):
    """
    Build loss function from config
    
    Args:
        config: Configuration dictionary containing loss function settings
    
    Returns:
        Loss function instance
    """
    loss_name = config.get('loss_function', 'mse')
    return loss_base(loss_name)







def rho_func(T,p):
    assert T is not None and p is not None, "T and p are required"
    try:
        import CoolProp as CP
        return torch.tensor([CP.PropsSI('D', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(T, p)],
                        dtype=torch.float32, device=T.device)
    except ImportError:
        return -0.005343 * T**2 + 3.0787 * T + 283.56

def mu_func(T,p):
    assert T is not None and p is not None, "T and p are required"
    try:
        import CoolProp as CP
        return torch.tensor([CP.PropsSI('V', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(T, p)],
                           dtype=torch.float32, device=T.device)
    except ImportError:
        return torch.exp(-0.0091114 * T - 4.3961)

def Cp_func(T,p):
    assert T is not None and p is not None, "T and p are required"
    try:
        import CoolProp as CP
        return torch.tensor([CP.PropsSI('C', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(T, p)],
                           dtype=torch.float32, device=T.device)
    except ImportError:
        return 4.1587 * T + 947.66

def k_func(T,p):
    assert T is not None and p is not None, "T and p are required"
    try:
        import CoolProp as CP
        return torch.tensor([CP.PropsSI('K', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(T, p)],
                           dtype=torch.float32, device=T.device)
    except ImportError:
        return 0.141 * T + 0.000193 * T**2 - 0.0000000657 * T**3



class PhysicsLoss:
    def __init__(self,batch):
        self.x,self.y,self.z,self.T,self.p = batch
    
    @classmethod
    def compute_physics_loss(self,w_rho=0.25,w_mu=0.25,w_Cp=0.25,w_k=0.25):
        if self.T and self.p is not None:
            return w_rho * rho_func(self.T,self.p) + w_mu * mu_func(self.T,self.p) + w_Cp * Cp_func(self.T,self.p) + w_k * k_func(self.T,self.p)
        else:
            return 0

