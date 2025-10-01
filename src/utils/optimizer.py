import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict,Any,List

class Optimizer(nn.Module):
    def __init__(self,name:str,params,config:Dict[str,Any]):
        super(Optimizer,self).__init__()
        self.name = name
        self.params = params
        self.lr = config.get('lr',1e-3)
        self.weight_decay = config.get('weight_decay',1e-5)
        self.betas = config.get('betas',(0.9,0.999))
        self.momentum = config.get('momentum',0.9)
        self.eps = config.get('eps',1e-8)
    
    def build(self) -> optim.Optimizer:
        if self.name == 'adam':
            return optim.Adam(self.params,lr=self.lr,weight_decay=self.weight_decay,betas=self.betas,eps=self.eps)
        elif self.name == 'sgd':
            return optim.SGD(self.params,lr=self.lr,momentum=self.momentum,weight_decay=self.weight_decay)
        elif self.name == 'adamw':
            return optim.AdamW(self.params,lr=self.lr,weight_decay=self.weight_decay,betas=self.betas,eps=self.eps)
        else:
            raise ValueError(f"Invalid optimizer: {self.name}")

def build_optimizer(name:str,params,config:Dict[str,Any]) -> optim.Optimizer:
    lr = config.get('lr',1e-3)
    weight_decay = config.get('weight_decay',1e-5)
    betas = config.get('betas',(0.9,0.999))
    momentum = config.get('momentum',0.9)
    eps = config.get('eps',1e-8)
    
    if name == 'adam':
        return optim.Adam(params,lr=lr,weight_decay=weight_decay,betas=betas,eps=eps)
    elif name == 'sgd':
        return optim.SGD(params,lr=lr,momentum=momentum,weight_decay=weight_decay)
    elif name == 'adamw':
        return optim.AdamW(params,lr=lr,weight_decay=weight_decay,betas=betas,eps=eps)
    else:
        raise ValueError(f"Invalid optimizer: {name}")