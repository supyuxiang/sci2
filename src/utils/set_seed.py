import torch
import numpy as np
import random

def set_seed(seed=42):  
    torch.manual_seed(seed)  
    np.random.seed(seed)  
    random.seed(seed)