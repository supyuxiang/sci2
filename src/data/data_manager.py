'''
Data manager, load data, split data, preprocess data, build dataloader
'''
import sys
from pathlib import Path
import time
from dataclasses import dataclass
import torch
from typing import Dict
import numpy as np
import pandas as pd
from torch.utils.data import SubsetRandomSampler, TensorDataset, DataLoader

sys.path.insert(0,str(Path(__file__).parent.parent))

from utils.scaler import build_scaler
from utils.logger import Logger


class DataManager:
    """Load excel and produce phase-specific datasets.

    - Phase 1: rows [0:index_phase1_end) => predict T
    - Phase 2: rows [index_phase1_end: ) => predict T,u,v,p

    input:
    config: DataConfig

    output:
    phase1_dataloader: Dataloader
    phase2_dataloader: Dataloader

    """

    def __init__(self, config: Dict):
        start_time = time.time()
        self.config = config.get('Data',{})
        self.logger = Logger("DataManager", config)
        self.data_path = self.config.get('data_path', None)  # 先设置data_path
        self._check_basic_config()  # 只检查基本配置
        self._data_load()  # 加载数据
        self._check_data_config()  # 检查数据相关的配置
        self._split_data_based_on_phase()
        self._data_preprocessing(scaler_name=self.config.get('scaler_name','standardscaler'))
        self._build_dataloader()
        end_time = time.time()
        self.logger.info(f"DataManager initialized, config: {self.config}, time: {end_time - start_time}")



    def _check_basic_config(self) -> None:
        '''
        check basic config (before data loading)
        '''
        assert self.data_path is not None, "data_path is not set"
        assert self.config.get('index_start',None) is not None, "index_start is not set"
        assert Path(self.data_path).exists(),'data_path is not exists'
        assert self.config.get('phase1_index_end',None) is not None, "phase1_index_end is not set"
        assert self.config.get('features',None) is not None, 'features is not set'
        self.logger.info("Basic config checked completed")

    def _check_data_config(self) -> None:
        '''
        check data-related config (after data loading)
        '''
        output_cols_phase1 = self.config.get('phase1_targets',None)
        output_cols_phase2 = self.config.get('phase2_targets',None)
        assert output_cols_phase1 is not None, "phase1_targets is not set"
        assert output_cols_phase2 is not None, 'phase2_targets is not set'
        error_cols1 = [col for col in output_cols_phase1 if col not in self.df.columns]
        error_cols2 = [col for col in output_cols_phase2 if col not in self.df.columns]
        assert len(error_cols1) == 0, f"error_cols1: {error_cols1}"
        assert len(error_cols2) == 0, f"error_cols2: {error_cols2}"
        self.logger.info(f"Data config checked completed, output_cols_phase1: {output_cols_phase1}, output_cols_phase2: {output_cols_phase2}")


    def _data_load(self) -> None:
        '''
        laod data form data_path
        '''
        # self.data_path already set in __init__
        
        if self.data_path.endswith('.xlsx'):

            '''self.df = pd.read_excel(self.data_path)
            self.data = self.df[self.config.get('features',None)].values.to_numpy(dtype=np.float32)
            self.targets1 = self.df[self.config.get('phase1_targets',None)].values.to_numpy(dtype=np.float32)
            self.targets2 = self.df[self.config.get('phase2_targets',None)].values.to_numpy(dtype=np.float32)'''
            # 读取Excel文件，跳过前面的元数据行，使用第一行作为列名
            self.df = pd.read_excel(self.data_path, sheet_name='三维双热通量(2)', skiprows=7, header=0)
            # 重新设置列名（去掉第一行，使用第二行作为列名）
            self.df.columns = self.df.iloc[0]
            self.df = self.df.drop(self.df.index[0]).reset_index(drop=True)
            self.data = self.df[self.config.get('features',None)].values.astype(np.float32)
            self.targets1 = self.df[self.config.get('phase1_targets',None)].values.astype(np.float32)
            self.targets2 = self.df[self.config.get('phase2_targets',None)].values.astype(np.float32)
        elif self.data_path.endswith('.csv'):
            self.logger.warning(f"Data loaded from csv file, data_path: {self.data_path}")
            self.df = pd.read_csv(self.data_path)

            '''self.data = self.df[self.config.get('features',None)].values.to_numpy(dtype=np.float32)
            self.targets1 = self.df[self.config.get('phase1_targets',None)].values.to_numpy(dtype=np.float32)
            self.targets2 = self.df[self.config.get('phase2_targets',None)].values.to_numpy(dtype=np.float32)'''
            self.data = self.df[self.config.get('features',None)].values.astype(np.float32)
            self.targets1 = self.df[self.config.get('phase1_targets',None)].values.astype(np.float32)
            self.targets2 = self.df[self.config.get('phase2_targets',None)].values.astype(np.float32)
        else:
            self.logger.error(f"Unsupported file extension: {self.data_path}")
            raise ValueError(f"Unsupported file extension: {self.data_path}")
        self.logger.info(f"Data loaded completed, data_path: {self.data_path}")

    def _split_data_based_on_phase(self) -> None:
        '''
        split data into train and test
        '''
        from sklearn.model_selection import train_test_split
        
        index_start = self.config.get('index_start')
        '''phase1_index_end = self.config.get('phase1_index_end')
        self.phase1_input = self.data[index_start:phase1_index_end,:]
        self.phase1_output = self.targets1[index_start:phase1_index_end,:]
        self.phase2_input = self.data[phase1_index_end:,:]
        self.phase2_output = self.targets2[phase1_index_end:,:]
        self.logger.info(f'Data split based on phase completed, phase1_index_end: {phase1_index_end}')'''
        
        phase1_index_end = self.config.get('phase1_index_end')
        test_ratio = self.config.get('test_ratio', 0.2)
        random_state = self.config.get('random_state', 42)
        
        # Phase 1 data
        phase1_input = self.data[index_start:phase1_index_end,:]
        phase1_output = self.targets1[index_start:phase1_index_end,:]
        
        # Phase 2 data
        phase2_input = self.data[phase1_index_end+1:,:]
        phase2_output = self.targets2[phase1_index_end+1:,:]
        
        # Split phase 1 into train/test
        self.phase1_input_train, self.phase1_input_test, self.phase1_output_train, self.phase1_output_test = train_test_split(
            phase1_input, phase1_output, test_size=test_ratio, random_state=random_state
        )
        
        # Split phase 2 into train/test
        self.phase2_input_train, self.phase2_input_test, self.phase2_output_train, self.phase2_output_test = train_test_split(
            phase2_input, phase2_output, test_size=test_ratio, random_state=random_state
        )
        
        self.logger.info(f'Data split based on phase completed, phase1_index_end: {phase1_index_end}, test_ratio: {test_ratio}')

    def _data_preprocessing(self, scaler_name : str) -> None:
        '''
        process data
        '''
        scaler = build_scaler(scaler_name)
        self.phase1_input_train = scaler.fit_transform(self.phase1_input_train)
        self.phase1_input_test = scaler.transform(self.phase1_input_test)
        self.phase2_input_train = scaler.fit_transform(self.phase2_input_train)
        self.phase2_input_test = scaler.transform(self.phase2_input_test)
        self.logger.info(f"Data preprocessing completed, scaler: {scaler_name}, phase1_input_train: {self.phase1_input_train.shape}, phase1_input_test: {self.phase1_input_test.shape}, phase2_input_train: {self.phase2_input_train.shape}, phase2_input_test: {self.phase2_input_test.shape}")
    
    def _build_dataloader(self):
        '''
        build dataloader
        '''
        # 原实现：直接将 (inputs, targets) 二元组交给 DataLoader，导致后续取 batch 解包时报错
        # self.dataloader_phase1, self.dataloader_phase2 = Dataloader(self.phase1_input_train, self.phase1_output_train, self.phase2_input_train, self.phase2_output_train, self.config.get('batch_size1',64), self.logger).build_dataloader()
        
        # 新实现：先包装为逐样本返回 (x_i, y_i) 的 TensorDataset，再交给 DataLoader
        self.dataloader_phase1_train, self.dataloader_phase1_test = Dataloader(
            self.phase1_input_train,
            self.phase1_output_train,
            self.phase1_input_test,
            self.phase1_output_test,
            self.config.get('batch_size1', 64),
            self.logger
        ).build_dataloader()

        self.dataloader_phase2_train,self.dataloader_phase2_test = Dataloader(
            self.phase2_input_train,
            self.phase2_output_train,
            self.phase2_input_test,
            self.phase2_output_test,
            self.config.get('batch_size2', 64),
            self.logger
        ).build_dataloader()
        
        self.logger.info(f"Dataloader built, dataloader_phase1_train: {self.dataloader_phase1_train}, dataloader_phase1_test: {self.dataloader_phase1_test}, dataloader_phase2_train: {self.dataloader_phase2_train}, dataloader_phase2_test: {self.dataloader_phase2_test}")



class Dataloader:
    '''
    build dataloader
    input:
    - phase1_datasets
    - phase2_datasets
    output:
    - dataloader_phase1
    - dataloader_phase2
    '''

    def __init__(self,phase1_input,phase1_output,phase2_input,phase2_output,batch_size,logger):
        self.batch_size = batch_size
        self.phase1_input = phase1_input
        self.phase1_output = phase1_output
        self.phase2_input = phase2_input
        self.phase2_output = phase2_output
        self.logger = logger
        self.build_datasets()
        self.build_sampler()

    def build_datasets(self):
        '''
        build datasets
        '''
        # 旧实现：使用 numpy 二元组 (X, y)
        # self.datasets1 = (self.phase1_input, self.phase1_output)
        # self.datasets2 = (self.phase2_input, self.phase2_output)

        # 新实现：显式构建 TensorDataset，逐样本返回 (x_i, y_i)
        inputs1 = torch.from_numpy(self.phase1_input).float()
        targets1 = torch.from_numpy(self.phase1_output).float()
        inputs2 = torch.from_numpy(self.phase2_input).float()
        targets2 = torch.from_numpy(self.phase2_output).float()

        # 保证 target 为二维形状 [N, C]
        if targets1.dim() == 1:
            targets1 = targets1.unsqueeze(1)
        if targets2.dim() == 1:
            targets2 = targets2.unsqueeze(1)

        self.datasets1 = TensorDataset(inputs1, targets1)
        self.datasets2 = TensorDataset(inputs2, targets2)
        self.logger.info(f"Build datasets, datasets1: {self.datasets1}, datasets2: {self.datasets2}")
    
    def build_sampler(self):
        '''
        build sampler
        '''
        # 旧实现：基于二元组计算长度
        # n = len(self.datasets1[0])
        # n2 = len(self.datasets2[0])

        # 新实现：直接使用 Dataset 的长度
        n = len(self.datasets1)
        n2 = len(self.datasets2)
        self.logger.info(f"Build sampler, datasets1: {self.datasets1}, datasets2: {self.datasets2}")
        indices1 = list(range(n))
        indices2 = list(range(n2))
        self.sampler1 = SubsetRandomSampler(indices1)
        self.sampler2 = SubsetRandomSampler(indices2)
        self.logger.info(f"Build sampler, sampler1: {self.sampler1}, sampler2: {self.sampler2}")

    def build_dataloader(self):
        '''
        build dataloader
        '''
        # 旧实现：直接把 (X, y) 二元组交给 DataLoader
        # dataloader1 = torch.utils.data.DataLoader(self.datasets1, sampler=self.sampler1, batch_size=self.batch_size, shuffle=False)
        # dataloader2 = torch.utils.data.DataLoader(self.datasets2, sampler=self.sampler2, batch_size=self.batch_size, shuffle=False)

        # 新实现：对 TensorDataset 构建 DataLoader
        dataloader1 = DataLoader(self.datasets1, sampler=self.sampler1, batch_size=self.batch_size, shuffle=False)
        dataloader2 = DataLoader(self.datasets2, sampler=self.sampler2, batch_size=self.batch_size, shuffle=False)
        self.logger.info(f"Build dataloader, dataloader1: {dataloader1}, dataloader2: {dataloader2}")
        return dataloader1,dataloader2








