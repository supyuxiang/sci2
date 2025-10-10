import torch
import torch.nn as nn
from typing import Dict,Any
import CoolProp as CP


def build_loss_function(name: str):
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



class PhysicsLoss:
    def __init__(self,batch,phase):
        if phase == 1:
            self.x,self.y,self.z,self.T,self.p = batch
        elif phase == 2:
            self.x,self.y,self.z,self.T,self.p = batch
        else:
            raise ValueError(f"Invalid phase: {phase}. Available phases: 1, 2")
    
    @classmethod
    def rho_func(self,T,p):
        assert self.T is not None and self.p is not None, "T and p are required"
        if CP is not None:
            return torch.tensor([CP.PropsSI('D', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(self.T, self.p)],
                           dtype=torch.float32, device=self.T.device)
        else:
            return -0.005343 * self.T**2 + 3.0787 * self.T + 283.56
        
    @classmethod
    def mu_func(self,T,p):
        assert self.T is not None and self.p is not None, "T and p are required"
        if CP is not None:
            return torch.tensor([CP.PropsSI('V', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(self.T, self.p)],
                           dtype=torch.float32, device=self.T.device)
        else:
            return torch.exp(-0.0091114 * self.T - 4.3961)

    @classmethod
    def Cp_func(self):
        assert self.T is not None and self.p is not None, "T and p are required"
        if CP is not None:
            return torch.tensor([CP.PropsSI('C', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(self.T, self.p)],
                           dtype=torch.float32, device=self.T.device)
        else:
            return 4.1587 * self.T + 947.66
        
    @classmethod
    def k_func(self):
        assert self.T is not None and self.p is not None, "T and p are required"
        if CP is not None:
            return torch.tensor([CP.PropsSI('K', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(self.T, self.p)],
                           dtype=torch.float32, device=self.T.device)
        else:
            return 0.141 * self.T + 0.000193 * self.T**2 - 0.0000000657 * self.T**3
    
    @classmethod
    def physics_loss(self,w_rho=0.25,w_mu=0.25,w_Cp=0.25,w_k=0.25):
        if self.T and self.p is not None:
            return w_rho * self.rho_func() + w_mu * self.mu_func() + w_Cp * self.Cp_func() + w_k * self.k_func()
        else:
            return 0

