



from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Union


def build_scaler(scaler_name: str) -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
    """
    Build a scaler based on the given name.
    
    Args:
        scaler_name: Name of the scaler ('standardscaler', 'minmaxscaler', 'robustscaler')
    
    Returns:
        Scaler instance
    """
    scaler_name = scaler_name.lower()
    
    if scaler_name == 'standardscaler':
        return StandardScaler()
    elif scaler_name == 'minmaxscaler':
        return MinMaxScaler()
    elif scaler_name == 'robustscaler':
        return RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler_name}. Available scalers: 'standardscaler', 'minmaxscaler', 'robustscaler'")


class Scaler:
    """Base scaler class for future extensions"""
    pass











