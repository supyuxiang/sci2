import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import yaml
import logging
sys.path.insert(0,str(Path(__file__).parent.parent))
print(sys.path[0])

from src.utils.logger import Logger
from src.models.models import ModelFactory

try:
    config_path = Path(__file__).parent.parent / 'src' / 'config.yaml'
    with open(config_path,'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("config.yaml loaded successfully")
    print(f"Data config: {config.get('Data', {})}")
except Exception as e:
    print(f"config.yaml loaded failed: {e}")
    sys.exit(1)
'''
try:
    from src.data.data_manager import DataManager
    print("DataManager imported successfully")
    print('*'*1000)
    logger = Logger("test_script", config)
    logger.info("DataManager initialized successfully")
    logger.info('*'*100)
    data_manager = DataManager(config)
    print("DataManager initialized successfully")
    print('*'*1000)
except Exception as e:
    print(f"DataManager imported failed: {e}")
    sys.exit(1)



try:  
    model = ModelFactory(phase=1,logger=logger,model_name="mlp").model
    print("Model initialized successfully")
    print('*'*1000)
except Exception as e:
    print(f"Model initialized failed: {e}")
    sys.exit(1)
try:
    dataloader1_train,dataloader1_test = data_manager.dataloader_phase1_train,data_manager.dataloader_phase1_test  
    print(f"Dataloader1_train: {dataloader1_train}")
    print(f"Dataloader1_test: {dataloader1_test}")
    print(f'length of dataloader1_train: {len(dataloader1_train)}')
    print(f'length of dataloader1_test: {len(dataloader1_test)}')
    print('*'*1000)
except Exception as e:
    print(f"Dataloader initialized failed: {e}")
    sys.exit(1)



try:
    from src.trainer.train import Trainer
    print("Trainer imported successfully")
    print('*'*1000)
except Exception as e:
    print(f"Trainer imported failed: {e}")
    sys.exit(1)



try:
    trainer = Trainer(config=config,
        model=model,
        dataloader_train=dataloader1_train,
        logger=logger
    ).validate(dataloader_val = dataloader1_test)
    print("Trainer initialized successfully")
    print('*'*1000)
    print(f"Metrics: {trainer.metrics}")
except Exception as e:
    print(f"Trainer initialized failed: {e}")
    sys.exit(1)'''


try:
    from main import create_dataloaders
    dataloader1_train,dataloader1_test,dataloader2_train,dataloader2_test = create_dataloaders(config)
    print(f"Dataloader1_train: {dataloader1_train}")
    print(f"Dataloader1_test: {dataloader1_test}")
    print(f"Dataloader2_train: {dataloader2_train}")
    print(f"Dataloader2_test: {dataloader2_test}")
    print('*'*1000)
    print("create_dataloaders imported successfully")
    print('*'*1000)
    for batch_in,batch_out in dataloader2_train:
        print(f"Batch in: {batch_in}")
        print(f"Batch out: {batch_out}")
        print('*'*1000)
except Exception as e:
    print(f"create_dataloaders imported failed: {e}")
    sys.exit(1)






