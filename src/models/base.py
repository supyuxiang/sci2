from abc import ABC, abstractmethod
from typing import Any,Dict
import numpy as np

from utils.logger import Logger

class BaseFactory(ABC):
    def __init__(self,config:Dict,logger:Logger):
        self.config = config
        self.logger = logger
        self.logger.info(f"BaseFactory initialized, config: {config}")
    @abstractmethod
    def build_model(self):
        pass
    
    
   
