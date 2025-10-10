"""
Hydra使用示例 - 展示正确的Hydra用法
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# 设置日志
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="src", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    主函数 - 使用Hydra装饰器
    
    Args:
        cfg: Hydra配置对象，包含所有配置信息
    """
    # 打印配置信息
    print("=== 完整配置 ===")
    print(OmegaConf.to_yaml(cfg))
    
    # 访问配置的不同方式
    print("\n=== 配置访问示例 ===")
    
    # 1. 直接访问
    print(f"数据路径: {cfg.Data.data_path}")
    print(f"训练轮数: {cfg.Train.epochs}")
    print(f"模型名称: {cfg.Model.model_name}")
    
    # 2. 使用get方法（推荐，更安全）
    batch_size = cfg.get('Data', {}).get('batch_size1', 32)
    print(f"批次大小: {batch_size}")
    
    # 3. 检查配置是否存在
    if 'phase_ls' in cfg.Train:
        print(f"训练阶段: {cfg.Train.phase_ls}")
    
    # 4. 类型转换
    epochs = int(cfg.Train.epochs)
    print(f"训练轮数（整数）: {epochs}")
    
    # 5. 配置验证
    required_keys = ['Data', 'Train', 'Model']
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"缺少必需的配置节: {key}")
    
    print("\n=== 配置验证通过 ===")

if __name__ == "__main__":
    main()
