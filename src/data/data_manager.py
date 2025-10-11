'''
Data manager, load data, split data, preprocess data, build dataloader
'''
import sys
from pathlib import Path
import time
from dataclasses import dataclass

from seaborn.relational import _DashType
import torch
from typing import Dict
import numpy as np
import pandas as pd
from torch.utils.data import SubsetRandomSampler, TensorDataset, DataLoader

sys.path.insert(0,str(Path(__file__).parent.parent))

from utils.scaler import build_scaler
from utils.logger import Logger


class DataManager:
    """Load excel and produce single-phase dataset.

    input:
    config: DataConfig

    output:
    dataloader_train: Dataloader
    dataloader_test: Dataloader

    """

    def __init__(self, config: Dict,logger: Logger,data_path:str):
        """
        init data manager
        """
        start_time = time.time()
        self.config:Dict = config.get('Data',{})
        self.logger:Logger = logger
        self.data_path:str = self.config.get('data_path', None)  # 先设置data_path
        self.data_load()  # 检查配置并加载数据
        self.split_data()
        self.data_preprocessing(scaler_name=self.config.get('scaler_name','standardscaler'))  # 处理
        self.build_dataloader()
        end_time = time.time()
        self.logger.info(f"DataManager initialized, config: {self.config}, time: {end_time - start_time}")



    def data_load(self) -> None:
        '''
        check config and load data from data_path
        '''
        # 1. 检查基本配置
        assert self.data_path is not None, "data_path is not set"
        assert Path(self.data_path).exists(), 'data_path is not exists'
        assert self.config.get('features', None) is not None, 'features is not set'
        self.logger.info("Basic config checked completed")
        
        # 2. 加载数据
        if self.data_path.endswith('.xlsx'):  # 读取Excel文件
            # 读取Excel文件，跳过前面的元数据行，使用第一行作为列名
            self.df = pd.read_excel(self.data_path, sheet_name=self.config.get('sheet_name', '数据100mm流体域'), skiprows=7, header=0)
            # 重新设置列名（去掉第一行，使用第二行作为列名）
            self.df.columns = self.df.iloc[0]
            self.df = self.df.drop(self.df.index[0]).reset_index(drop=True)
            self.data = torch.tensor(self.df[self.config.get('features', None)].values, dtype=torch.float32, requires_grad=True)
            self.targets = torch.tensor(self.df[self.config.get('targets', None)].values, dtype=torch.float32, requires_grad=True)
        
        elif self.data_path.endswith('.csv'):  # 读取CSV文件
            self.logger.warning(f"Data loaded from csv file, data_path: {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            self.data = torch.tensor(self.df[self.config.get('features', None)].values, dtype=torch.float32, requires_grad=True)
            self.targets = torch.tensor(self.df[self.config.get('targets', None)].values, dtype=torch.float32, requires_grad=True)
        else:
            self.logger.error(f"Unsupported file extension: {self.data_path}")
            raise ValueError(f"Unsupported file extension: {self.data_path}")
        
        # 3. 检查数据相关配置
        output_cols = self.config.get('targets', None)
        assert output_cols is not None, "targets is not set"
        self.logger.info(f"Data config checked completed, output_cols: {output_cols}")
        self.logger.info(f"Data loaded completed, data_path: {self.data_path}")

    def split_data(self) -> None:
        '''
        split data into train and test
        '''
        from sklearn.model_selection import train_test_split
        
        test_ratio = self.config.get('test_ratio', 0.2)
        random_state = self.config.get('random_state', 42)
        
        # Clean NaNs in data (features or targets). Keep only rows with all finite values
        mask = torch.all(torch.isfinite(self.data), axis=1) & torch.all(torch.isfinite(self.targets), axis=1)
        removed = int((~mask).sum())
        if removed > 0:
            self.logger.warning(f"Detected and removed {removed} rows containing NaN/Inf.")
        
        clean_data = self.data[mask]
        clean_targets = self.targets[mask]
        
        # Split into train/test
        self.input_train, self.input_test, self.output_train, self.output_test = train_test_split(
            clean_data, clean_targets, test_size=test_ratio, random_state=random_state
        )
        
        self.logger.info(f'Data split completed, test_ratio: {test_ratio}, train_size: {len(self.input_train)}, test_size: {len(self.input_test)}')

    def data_preprocessing(self, scaler_name : str) -> None:
        '''
        process data
        '''
        scaler = build_scaler(scaler_name)
        self.input_train = scaler.fit_transform(self.input_train)
        self.input_test = scaler.transform(self.input_test)
        self.logger.info(f"Data preprocessing completed, scaler: {scaler_name}, input_train: {self.input_train.shape}, input_test: {self.input_test.shape}")
    
    def build_dataloader(self):
        '''
        build dataloader
        '''
        self.dataloader_train, self.dataloader_test = Dataloader(
            self.input_train,
            self.output_train,
            self.input_test,
            self.output_test,
            self.config.get('batch_size', 64),
            self.logger
        ).build_dataloader()
        
        self.logger.info(f"Dataloader built, train_size: {len(self.dataloader_train.dataset)}, test_size: {len(self.dataloader_test.dataset)}")



class Dataloader:
    '''
    build dataloader
    input:
    - input_train, output_train
    - input_test, output_test
    output:
    - dataloader_train, dataloader_test
    '''

    def __init__(self, input_train, output_train, input_test, output_test, batch_size, logger):
        self.batch_size = batch_size
        self.input_train = input_train
        self.output_train = output_train
        self.input_test = input_test
        self.output_test = output_test
        self.logger = logger
        self.build_datasets()
        self.build_sampler()

    def build_datasets(self):
        '''
        build datasets
        '''
        # 构建训练和测试数据集
        inputs_train = torch.from_numpy(self.input_train).float()
        targets_train = torch.from_numpy(self.output_train).float()
        inputs_test = torch.from_numpy(self.input_test).float()
        targets_test = torch.from_numpy(self.output_test).float()

        # 保证 target 为二维形状 [N, C]
        if targets_train.dim() == 1:
            targets_train = targets_train.unsqueeze(1)
        if targets_test.dim() == 1:
            targets_test = targets_test.unsqueeze(1)

        self.dataset_train = TensorDataset(inputs_train, targets_train)
        self.dataset_test = TensorDataset(inputs_test, targets_test)
        self.logger.info(f"Build datasets, train_size: {len(self.dataset_train)}, test_size: {len(self.dataset_test)}")
    
    def build_sampler(self):
        '''
        build sampler
        '''
        # 构建训练和测试的采样器
        n_train = len(self.dataset_train)
        n_test = len(self.dataset_test)
        
        indices_train = list(range(n_train))
        indices_test = list(range(n_test))
        
        self.sampler_train = SubsetRandomSampler(indices_train)
        self.sampler_test = SubsetRandomSampler(indices_test)
        
        self.logger.info(f"Build sampler, train_size: {n_train}, test_size: {n_test}")

    def build_dataloader(self):
        '''
        build dataloader
        '''
        # 构建训练和测试的DataLoader
        dataloader_train = DataLoader(self.dataset_train, sampler=self.sampler_train, batch_size=self.batch_size, shuffle=False)
        dataloader_test = DataLoader(self.dataset_test, sampler=self.sampler_test, batch_size=self.batch_size, shuffle=False)
        
        self.logger.info(f"Build dataloader, train_batches: {len(dataloader_train)}, test_batches: {len(dataloader_test)}")
        return dataloader_train, dataloader_test








