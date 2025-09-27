import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler,Normalizer,QuantileTransformer,PowerTransformer

def _get_scaler(method: str):
    if method == 'standard':
        return StandardScaler()
    if method == 'minmax':
        return MinMaxScaler()
    if method == 'robust':
        return RobustScaler()
    if method == 'maxabs':
        return MaxAbsScaler()
    if method == 'normalize':
        return Normalizer()
    if method == 'quantile':
        return QuantileTransformer(output_distribution='uniform', random_state=0)
    if method == 'power':
        return PowerTransformer(method='yeo-johnson', standardize=True)
    raise ValueError(f"Invalid method: {method}")

class Datapreprocessing1:
    '''
    data preprocessing class for 1-dimensional data
    '''
    def __init__(self,file_path: str):
        '''
        init method
        return: None
        '''
        self.file_path_read = file_path
        self.df = pd.read_excel(self.file_path_read,header=None)
        self.df_preprocessed = self.df.iloc[584:,:].values
        

    def data_load(self):
        '''
        data loading method
        return: x,y,T
        '''
        df = pd.DataFrame(self.df_preprocessed)
        x = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        y = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        u = pd.to_numeric(df.iloc[:, 2], errors='coerce')
        v = pd.to_numeric(df.iloc[:, 3], errors='coerce')
        T = pd.to_numeric(df.iloc[:, 4], errors='coerce')
        p = pd.to_numeric(df.iloc[:, 5], errors='coerce')
        mask = x.notna() & y.notna() & u.notna() & v.notna() & T.notna() & p.notna()
        self.x = x[mask].to_numpy(dtype=np.float32).reshape(-1, 1)
        self.y = y[mask].to_numpy(dtype=np.float32).reshape(-1, 1)
        self.T = T[mask].to_numpy(dtype=np.float32).reshape(-1, 1)
        self.u = u[mask].to_numpy(dtype=np.float32).reshape(-1, 1)
        self.v = v[mask].to_numpy(dtype=np.float32).reshape(-1, 1)
        self.p = p[mask].to_numpy(dtype=np.float32).reshape(-1, 1)

    def data_scale(self,method:str):
        '''
        data scaling method: standard, minmax, robust, maxabs, normalize, quantile, power
        standard: standard scaling
        minmax: min-max scaling
        robust: robust scaling
        maxabs: max-abs scaling
        normalize: normalization
        quantile: quantile transformation
        power: power transformation
        method: str
        return: x_scaled,y_scaled,T_scaled
        '''
        if method == 'standard':
            self.scaler = StandardScaler()
            self.x_scaled = self.scaler.fit_transform(self.x).astype(np.float32)
            self.y_scaled = self.scaler.fit_transform(self.y).astype(np.float32)
            self.u_scaled = self.scaler.fit_transform(self.u).astype(np.float32)
            self.v_scaled = self.scaler.fit_transform(self.v).astype(np.float32)
            self.p_scaled = self.scaler.fit_transform(self.p).astype(np.float32)
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
            self.x_scaled = self.scaler.fit_transform(self.x).astype(np.float32)
            self.y_scaled = self.scaler.fit_transform(self.y).astype(np.float32)
            self.u_scaled = self.scaler.fit_transform(self.u).astype(np.float32)
            self.v_scaled = self.scaler.fit_transform(self.v).astype(np.float32)
            self.p_scaled = self.scaler.fit_transform(self.p).astype(np.float32)
        elif method == 'robust':
            self.scaler = RobustScaler()
            self.x_scaled = self.scaler.fit_transform(self.x).astype(np.float32)
            self.y_scaled = self.scaler.fit_transform(self.y).astype(np.float32)
            self.u_scaled = self.scaler.fit_transform(self.u).astype(np.float32)
            self.v_scaled = self.scaler.fit_transform(self.v).astype(np.float32)
            self.p_scaled = self.scaler.fit_transform(self.p).astype(np.float32)
        elif method == 'maxabs':
            self.scaler = MaxAbsScaler()
            self.x_scaled = self.scaler.fit_transform(self.x).astype(np.float32)
            self.y_scaled = self.scaler.fit_transform(self.y).astype(np.float32)
            self.u_scaled = self.scaler.fit_transform(self.u).astype(np.float32)
            self.v_scaled = self.scaler.fit_transform(self.v).astype(np.float32)
            self.p_scaled = self.scaler.fit_transform(self.p).astype(np.float32)
        elif method == 'normalize':
            self.scaler = Normalizer()
            self.x_scaled = self.scaler.fit_transform(self.x).astype(np.float32)
            self.y_scaled = self.scaler.fit_transform(self.y).astype(np.float32)
            self.u_scaled = self.scaler.fit_transform(self.u).astype(np.float32)
            self.v_scaled = self.scaler.fit_transform(self.v).astype(np.float32)
            self.p_scaled = self.scaler.fit_transform(self.p).astype(np.float32)
        elif method == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='uniform', random_state=0)
            self.x_scaled = self.scaler.fit_transform(self.x).astype(np.float32)
            self.y_scaled = self.scaler.fit_transform(self.y).astype(np.float32)
            self.u_scaled = self.scaler.fit_transform(self.u).astype(np.float32)
            self.v_scaled = self.scaler.fit_transform(self.v).astype(np.float32)
            self.p_scaled = self.scaler.fit_transform(self.p).astype(np.float32)
        elif method == 'power':
            self.scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            self.x_scaled = self.scaler.fit_transform(self.x).astype(np.float32)
            self.y_scaled = self.scaler.fit_transform(self.y).astype(np.float32)
            self.u_scaled = self.scaler.fit_transform(self.u).astype(np.float32)
            self.v_scaled = self.scaler.fit_transform(self.v).astype(np.float32)
            self.p_scaled = self.scaler.fit_transform(self.p).astype(np.float32)
        else:
            raise ValueError(f"Invalid method: {method}")
        return self.x_scaled,self.y_scaled,self.T,self.u_scaled,self.v_scaled,self.p_scaled

    def run(self, method: str = 'standard'):
        '''end-to-end preprocessing: load -> scale -> return'''
        self.data_load()
        # 返回顺序与 config.py 期望一致：x,y,T,atm,u,v
        x_s, y_s, T, u_s, v_s, p_s = self.data_scale(method)
        atm_s = p_s
        return x_s, y_s, T, atm_s, u_s, v_s


 




