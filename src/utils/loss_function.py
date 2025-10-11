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
        # 确保T和p是张量
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=torch.float32)
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float32)
        return -0.005343 * T**2 + 3.0787 * T + 283.56

def mu_func(T,p):
    assert T is not None and p is not None, "T and p are required"
    try:
        import CoolProp as CP
        return torch.tensor([CP.PropsSI('V', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(T, p)],
                           dtype=torch.float32, device=T.device)
    except ImportError:
        # 确保T和p是张量
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=torch.float32)
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float32)
        return torch.exp(-0.0091114 * T - 4.3961)

def Cp_func(T,p):
    assert T is not None and p is not None, "T and p are required"
    try:
        import CoolProp as CP
        return torch.tensor([CP.PropsSI('C', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(T, p)],
                           dtype=torch.float32, device=T.device)
    except ImportError:
        # 确保T和p是张量
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=torch.float32)
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float32)
        return 4.1587 * T + 947.66

def k_func(T,p):
    assert T is not None and p is not None, "T and p are required"
    try:
        import CoolProp as CP
        return torch.tensor([CP.PropsSI('K', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(T, p)],
                           dtype=torch.float32, device=T.device)
    except ImportError:
        # 确保T和p是张量
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=torch.float32)
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float32)
        return 0.141 * T + 0.000193 * T**2 - 0.0000000657 * T**3



class PhysicsLoss:
    def __init__(self, batch):
        """
        初始化物理损失函数
        
        Args:
            batch: 预测结果张量 [batch_size, 4] - [T, spf.U, u, p]
        """
        
        if batch is not None and batch.dim() >= 2 and batch.shape[1] >= 4:
            self.T = batch[:, 0]      # 温度 T (K)
            self.spfu = batch[:, 1]   # 速度分量 spf.U (m/s)
            self.u = batch[:, 2]      # 速度分量 u (m/s)
            self.p = batch[:, 3]      # 压力 p (Pa)
        else:
            self.T = self.spfu = self.u = self.p = None
    
    def compute_physics_loss(self, w_rho=0.25, w_mu=0.25, w_Cp=0.25, w_k=0.25):
        """
        计算物理约束损失，保持原有的物理约束逻辑
        
        Args:
            w_rho: 密度权重
            w_mu: 粘度权重  
            w_Cp: 比热容权重
            w_k: 热导率权重
            
        Returns:
            物理损失值
        """
        if self.T is None or self.spfu is None or self.u is None or self.p is None:
            return torch.tensor(0.0, device=self.T.device if self.T is not None else 'cpu')
        
        try:
            # 计算物理属性
            rho = rho_func(self.T, self.p)
            mu = mu_func(self.T, self.p)
            cp = Cp_func(self.T, self.p)
            k = k_func(self.T, self.p)
            
            # 将物理属性转换为损失形式
            # 这里我们计算物理属性的变化率作为约束
            # 保持原有的物理约束逻辑，但转换为损失函数形式
            
            # 密度约束：密度应该合理变化
            rho_loss = torch.mean(torch.abs(rho - rho.detach().mean()))
            
            # 粘度约束：粘度应该合理变化
            mu_loss = torch.mean(torch.abs(mu - mu.detach().mean()))
            
            # 比热容约束：比热容应该合理变化
            cp_loss = torch.mean(torch.abs(cp - cp.detach().mean()))
            
            # 热导率约束：热导率应该合理变化
            k_loss = torch.mean(torch.abs(k - k.detach().mean()))
            
            # 组合物理约束损失，保持原有的权重结构
            physics_loss = w_rho * rho_loss + w_mu * mu_loss + w_Cp * cp_loss + w_k * k_loss
            
            return physics_loss
            
        except Exception as e:
            # 如果物理计算失败，返回零损失
            return torch.tensor(0.0, device=self.T.device if self.T is not None else 'cpu')

