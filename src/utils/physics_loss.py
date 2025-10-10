import torch
import torch.nn as nn
import CoolProp as CP

class PhysicsLoss:
    def __init__(self,x,y,z,T,p,w):
        self.x = x
        self.y = y
        self.z = z
        self.T = T
        self.p = p
        self.w = w
    
    @classmethod
    def rho_func(self,T,p):
        if CP is not None:
            return torch.tensor([CP.PropsSI('D', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(self.T, self.p)],
                           dtype=torch.float32, device=self.T.device)
        else:
            return -0.005343 * self.T**2 + 3.0787 * self.T + 283.56
        
    @classmethod
    def mu_func(self,T,p):
        if CP is not None:
            return torch.tensor([CP.PropsSI('V', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(self.T, self.p)],
                           dtype=torch.float32, device=self.T.device)
        else:
            return torch.exp(-0.0091114 * self.T - 4.3961)

    @classmethod
    def Cp_func(self):
        if CP is not None:
            return torch.tensor([CP.PropsSI('C', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(self.T, self.p)],
                           dtype=torch.float32, device=self.T.device)
        else:
            return 4.1587 * self.T + 947.66
        
    @classmethod
    def k_func(self):
        if CP is not None:
            return torch.tensor([CP.PropsSI('K', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(self.T, self.p)],
                           dtype=torch.float32, device=self.T.device)
        else:
            return 0.141 * self.T + 0.000193 * self.T**2 - 0.0000000657 * self.T**3
    
    
        
