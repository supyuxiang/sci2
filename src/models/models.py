import os
import sys
import logging
import time
import yaml
import torch
from typing import Dict, Any
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP
from .GatedResMLP import GatedResMLP
from .lstm import LSTM
from .AdaptiveGatedTransformerMLP import AdaptiveGatedTransformerMLP


class ModelFactory(torch.nn.Module):
    '''
    Model Factory: mlp, lstm
    '''
    def __init__(self,logger,phase:int,model_name:str='mlp'):
        super(ModelFactory,self).__init__()
        self.model_name = model_name
        self.logger = logger
        self.phase = phase
        self._load_config()
        self._check_config()
        self.build_model()
        self.logger.info(f"Model {self.model_name} initialized successfully")

    def _load_config(self) -> Dict:
        try:
            setting_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
            with open(setting_path, 'r') as f:
                self.model_setting = yaml.safe_load(f).get('Model',{})
            self.logger.info(f'Model setting loaded successfully!')
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise ValueError(f"Error loading config: {e}")

    def _check_config(self):
        if self.model_name == 'mlp':
            mlp_cfg = self.model_setting.get('mlp',None)
            if not mlp_cfg:
                raise ValueError('mlp setting is not set')
            self.indim = int(self.model_setting.get('input_dim',False))
            self.outdim1 = int(self.model_setting.get('output_dim1',False))
            self.outdim2 = int(self.model_setting.get('output_dim2',False))
            self.logger.info(f'mlp_cfg: {mlp_cfg}')
            self.hidden_dim = int(mlp_cfg.get('hidden_dim',False))
            self.num_blocks = int(mlp_cfg.get('num_blocks',False))
            self.dropout = float(mlp_cfg.get('dropout',False))
            assert all([self.indim,self.outdim1,self.outdim2,self.hidden_dim,self.num_blocks,self.dropout]),'input_dim, output_dim1, output_dim2, hidden_dim, num_blocks and dropout are not set'
        
        elif self.model_name == 'lstm':
            lstm_cfg = self.model_setting.get('lstm',None)
            if not lstm_cfg:
                raise ValueError('lstm setting is not set')
            self.indim = int(self.model_setting.get('input_dim',False))
            self.outdim1 = int(self.model_setting.get('output_dim1',False))
            self.outdim2 = int(self.model_setting.get('output_dim2',False))
            self.hidden_dim = int(lstm_cfg.get('hidden_dim',False))
            self.num_blocks = int(lstm_cfg.get('num_blocks',False))
            self.dropout = float(lstm_cfg.get('dropout',False))
            assert all([self.indim,self.outdim1,self.outdim2,self.hidden_dim,self.num_blocks,self.dropout]),'input_dim, output_dim1, output_dim2, hidden_dim, num_blocks and dropout are not set'
        
        elif self.model_name == 'grmlp':
            grmlp_cfg = self.model_setting.get('grmlp',None)
            if not grmlp_cfg:
                raise ValueError('grmlp setting is not set')
            self.indim = int(self.model_setting.get('input_dim',False))
            self.outdim1 = int(self.model_setting.get('output_dim1',False))
            self.outdim2 = int(self.model_setting.get('output_dim2',False))
            self.hidden_dim = int(grmlp_cfg.get('hidden_dim',False))
            self.depth = int(grmlp_cfg.get('depth',False))
            self.dropout = float(grmlp_cfg.get('dropout',False))
            assert all([self.indim,self.outdim1,self.outdim2,self.hidden_dim,self.depth,self.dropout]),'input_dim, output_dim1, output_dim2, hidden_dim, depth and dropout are not set'
        
        elif self.model_name == 'agtmlp':
            agtmlp_cfg = self.model_setting.get('agtmlp',None)
            if not agtmlp_cfg:
                raise ValueError('agtmlp setting is not set')
            self.indim = int(self.model_setting.get('input_dim',False))
            self.outdim1 = int(self.model_setting.get('output_dim1',False))
            self.outdim2 = int(self.model_setting.get('output_dim2',False))
            self.hidden_dim = int(agtmlp_cfg.get('hidden_dim',False))
            self.depth = int(agtmlp_cfg.get('depth',False))
            self.dropout = float(agtmlp_cfg.get('dropout',False))
            self.num_heads = int(agtmlp_cfg.get('num_heads',8))
            self.num_experts = int(agtmlp_cfg.get('num_experts',6))
            assert all([self.indim,self.outdim1,self.outdim2,self.hidden_dim,self.depth,self.dropout]),'input_dim, output_dim1, output_dim2, hidden_dim, depth and dropout are not set'
        else:
            raise ValueError(f"Model name {self.model_name} not supported")


    def build_model(self):
        if self.model_name == 'mlp':
            if self.phase == 1:
                self.model = MLP(
                    indim = self.indim,
                    outdim = self.outdim1,
                    hidden_dim = self.hidden_dim,
                    num_blocks = self.num_blocks,
                    dropout = self.dropout
                )
            elif self.phase == 2:
                self.model = MLP(
                    indim = self.indim,
                    outdim = self.outdim2,
                    hidden_dim = self.hidden_dim,
                    num_blocks = self.num_blocks,
                    dropout = self.dropout
                )
            else:
                raise ValueError(f"Phase {self.phase} not supported")
        
        
        elif self.model_name == 'lstm':
            if self.phase == 1:
                self.model = LSTM(
                    indim = self.indim,
                    outdim = self.outdim1,
                    hidden_dim = self.hidden_dim,
                    num_blocks = self.num_blocks,
                    dropout = self.dropout
                )
            elif self.phase == 2:
                self.model = LSTM(
                    indim = self.indim,
                    outdim = self.outdim2,
                    hidden_dim = self.hidden_dim,
                    num_blocks = self.num_blocks,
                    dropout = self.dropout
                )
            else:
                raise ValueError(f"Phase {self.phase} not supported")
            
        elif self.model_name == 'grmlp':
            if self.phase == 1:
                self.model = GatedResMLP(
                    indim = self.indim,
                    outdim = self.outdim1,
                    hidden_dim = self.hidden_dim,
                    depth = self.depth,
                    dropout = self.dropout
                )

            elif self.phase == 2:
                self.model = GatedResMLP(
                    indim = self.indim,
                    outdim = self.outdim2,
                    hidden_dim = self.hidden_dim,
                    depth = self.depth,
                    dropout = self.dropout
                )
            else:
                raise ValueError(f"Phase {self.phase} not supported")
        
        elif self.model_name == 'agtmlp':
            if self.phase == 1:
                self.model = AdaptiveGatedTransformerMLP(
                    indim = self.indim,
                    outdim = self.outdim1,
                    hidden_dim = self.hidden_dim,
                    depth = self.depth,
                    num_heads = self.num_heads,
                    num_experts = self.num_experts,
                    dropout = self.dropout
                )
            elif self.phase == 2:
                self.model = AdaptiveGatedTransformerMLP(
                    indim = self.indim,
                    outdim = self.outdim2,
                    hidden_dim = self.hidden_dim,
                    depth = self.depth,
                    num_heads = self.num_heads,
                    num_experts = self.num_experts,
                    dropout = self.dropout
                )
            else:
                raise ValueError(f"Phase {self.phase} not supported")
        
        else:
            self.logger.error(f"Model name {self.model_name} not supported")
            raise ValueError(f"Model name {self.model_name} not supported")
        self.logger.info(f"Model {self.model_name} built successfully")

    def parameters_count(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    


        
        





