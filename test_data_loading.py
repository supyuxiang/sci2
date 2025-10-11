#!/usr/bin/env python3
"""
数据加载模块测试脚本
测试DataManager的数据加载功能，验证配置和数据的兼容性
"""

import sys
import os
import traceback
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_manager import DataManager
from src.utils.logger import Logger


def create_test_config() -> Dict[str, Any]:
    """创建测试配置"""
    return {
        'Data': {
            'data_path': "/home/yxfeng/project2/sci2/data/数据100mm流体域.xlsx",
            'sheet_name': "数据100mm流体域",
            'features': ["x", "y", "z"],
            'targets': ["T (K)", "spf. U (m/s)", "u (m/s)", "p (Pa)"],
            'test_ratio': 0.2,
            'random_state': 42,
            'scaler_name': 'standardscaler',
            'batch_size': 64
        }
    }


def test_file_existence(config: Dict[str, Any]) -> bool:
    """测试文件是否存在"""
    print("=" * 60)
    print("🔍 测试1: 文件存在性检查")
    print("=" * 60)
    
    data_path = config['Data']['data_path']
    if Path(data_path).exists():
        print(f"✅ 数据文件存在: {data_path}")
        file_size = Path(data_path).stat().st_size / (1024 * 1024)  # MB
        print(f"📊 文件大小: {file_size:.2f} MB")
        return True
    else:
        print(f"❌ 数据文件不存在: {data_path}")
        return False


def test_excel_reading(config: Dict[str, Any]) -> bool:
    """测试Excel文件读取"""
    print("\n" + "=" * 60)
    print("📖 测试2: Excel文件读取")
    print("=" * 60)
    
    try:
        data_path = config['Data']['data_path']
        sheet_name = config['Data']['sheet_name']
        
        # 测试不同的skiprows值
        for skiprows in [7, 8, 9]:
            try:
                print(f"\n🔍 尝试 skiprows={skiprows}:")
                df = pd.read_excel(data_path, sheet_name=sheet_name, skiprows=skiprows, header=0)
                print(f"   📊 数据形状: {df.shape}")
                print(f"   📋 列名: {list(df.columns)}")
                
                # 检查是否包含期望的列
                expected_columns = config['Data']['features'] + config['Data']['targets']
                missing_columns = [col for col in expected_columns if col not in df.columns]
                
                if not missing_columns:
                    print(f"   ✅ 所有期望列都存在")
                    print(f"   📝 前5行数据预览:")
                    print(df.head().to_string())
                    return True, skiprows, df
                else:
                    print(f"   ⚠️  缺少列: {missing_columns}")
                    
            except Exception as e:
                print(f"   ❌ skiprows={skiprows} 失败: {e}")
        
        return False, None, None
        
    except Exception as e:
        print(f"❌ Excel读取失败: {e}")
        traceback.print_exc()
        return False, None, None


def test_data_manager_creation(config: Dict[str, Any]) -> bool:
    """测试DataManager创建"""
    print("\n" + "=" * 60)
    print("🏗️  测试3: DataManager创建")
    print("=" * 60)
    
    try:
        # 创建临时日志器
        logger = Logger(name='test_data_loading', config=config)
        
        # 创建DataManager
        print("🔧 创建DataManager...")
        data_manager = DataManager(config, logger)
        
        print("✅ DataManager创建成功")
        print(f"📊 训练数据形状: {data_manager.data.shape}")
        print(f"📊 目标数据形状: {data_manager.targets.shape}")
        print(f"📊 训练集大小: {len(data_manager.dataloader_train.dataset)}")
        print(f"📊 测试集大小: {len(data_manager.dataloader_test.dataset)}")
        
        return True, data_manager
        
    except Exception as e:
        print(f"❌ DataManager创建失败: {e}")
        traceback.print_exc()
        return False, None


def test_data_quality(data_manager) -> bool:
    """测试数据质量"""
    print("\n" + "=" * 60)
    print("🔍 测试4: 数据质量检查")
    print("=" * 60)
    
    try:
        data = data_manager.data
        targets = data_manager.targets
        
        # 检查数据类型
        print(f"📊 数据类型: {data.dtype}")
        print(f"📊 目标类型: {targets.dtype}")
        
        # 检查NaN和Inf值
        data_nan = torch.isnan(data).sum().item()
        data_inf = torch.isinf(data).sum().item()
        targets_nan = torch.isnan(targets).sum().item()
        targets_inf = torch.isinf(targets).sum().item()
        
        print(f"📊 数据NaN值: {data_nan}")
        print(f"📊 数据Inf值: {data_inf}")
        print(f"📊 目标NaN值: {targets_nan}")
        print(f"📊 目标Inf值: {targets_inf}")
        
        if data_nan == 0 and data_inf == 0 and targets_nan == 0 and targets_inf == 0:
            print("✅ 数据质量良好，无NaN/Inf值")
        else:
            print("⚠️  数据中存在NaN/Inf值")
        
        # 检查数据范围
        print(f"\n📊 特征数据统计:")
        print(f"   x: min={data[:, 0].min():.3f}, max={data[:, 0].max():.3f}")
        print(f"   y: min={data[:, 1].min():.3f}, max={data[:, 1].max():.3f}")
        print(f"   z: min={data[:, 2].min():.3f}, max={data[:, 2].max():.3f}")
        
        print(f"\n📊 目标数据统计:")
        print(f"   T (K): min={targets[:, 0].min():.3f}, max={targets[:, 0].max():.3f}")
        print(f"   spf. U (m/s): min={targets[:, 1].min():.3f}, max={targets[:, 1].max():.3f}")
        print(f"   u (m/s): min={targets[:, 2].min():.3f}, max={targets[:, 2].max():.3f}")
        print(f"   p (Pa): min={targets[:, 3].min():.3f}, max={targets[:, 3].max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据质量检查失败: {e}")
        traceback.print_exc()
        return False


def test_dataloader_functionality(data_manager) -> bool:
    """测试DataLoader功能"""
    print("\n" + "=" * 60)
    print("🔄 测试5: DataLoader功能")
    print("=" * 60)
    
    try:
        # 测试训练DataLoader
        print("🔍 测试训练DataLoader...")
        train_loader = data_manager.dataloader_train
        test_loader = data_manager.dataloader_test
        
        # 获取一个batch
        for batch_idx, (features, targets) in enumerate(train_loader):
            print(f"📊 Batch {batch_idx}:")
            print(f"   特征形状: {features.shape}")
            print(f"   目标形状: {targets.shape}")
            print(f"   特征范围: [{features.min():.3f}, {features.max():.3f}]")
            print(f"   目标范围: [{targets.min():.3f}, {targets.max():.3f}]")
            
            if batch_idx >= 2:  # 只测试前3个batch
                break
        
        # 测试测试DataLoader
        print("\n🔍 测试测试DataLoader...")
        for batch_idx, (features, targets) in enumerate(test_loader):
            print(f"📊 Test Batch {batch_idx}:")
            print(f"   特征形状: {features.shape}")
            print(f"   目标形状: {targets.shape}")
            
            if batch_idx >= 1:  # 只测试前2个batch
                break
        
        print("✅ DataLoader功能正常")
        return True
        
    except Exception as e:
        print(f"❌ DataLoader测试失败: {e}")
        traceback.print_exc()
        return False


def test_memory_usage(data_manager) -> bool:
    """测试内存使用情况"""
    print("\n" + "=" * 60)
    print("💾 测试6: 内存使用情况")
    print("=" * 60)
    
    try:
        import psutil
        import gc
        
        # 获取当前进程
        process = psutil.Process(os.getpid())
        
        # 记录内存使用
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        print(f"📊 加载前内存使用: {memory_before:.2f} MB")
        
        # 强制垃圾回收
        gc.collect()
        
        # 计算数据大小
        data_size = data_manager.data.numel() * data_manager.data.element_size() / 1024 / 1024
        targets_size = data_manager.targets.numel() * data_manager.targets.element_size() / 1024 / 1024
        
        print(f"📊 数据张量大小: {data_size:.2f} MB")
        print(f"📊 目标张量大小: {targets_size:.2f} MB")
        print(f"📊 总数据大小: {data_size + targets_size:.2f} MB")
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        print(f"📊 加载后内存使用: {memory_after:.2f} MB")
        print(f"📊 内存增长: {memory_after - memory_before:.2f} MB")
        
        if memory_after - memory_before < 1000:  # 小于1GB
            print("✅ 内存使用合理")
            return True
        else:
            print("⚠️  内存使用较高，请检查数据大小")
            return False
            
    except ImportError:
        print("⚠️  psutil未安装，跳过内存测试")
        return True
    except Exception as e:
        print(f"❌ 内存测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始数据加载模块测试")
    print("=" * 80)
    
    # 创建测试配置
    config = create_test_config()
    
    # 测试结果统计
    test_results = []
    
    # 测试1: 文件存在性
    result1 = test_file_existence(config)
    test_results.append(("文件存在性", result1))
    
    if not result1:
        print("\n❌ 文件不存在，终止测试")
        return
    
    # 测试2: Excel读取
    result2, skiprows, df = test_excel_reading(config)
    test_results.append(("Excel读取", result2))
    
    if not result2:
        print("\n❌ Excel读取失败，终止测试")
        return
    
    # 测试3: DataManager创建
    result3, data_manager = test_data_manager_creation(config)
    test_results.append(("DataManager创建", result3))
    
    if not result3:
        print("\n❌ DataManager创建失败，终止测试")
        return
    
    # 测试4: 数据质量
    result4 = test_data_quality(data_manager)
    test_results.append(("数据质量", result4))
    
    # 测试5: DataLoader功能
    result5 = test_dataloader_functionality(data_manager)
    test_results.append(("DataLoader功能", result5))
    
    # 测试6: 内存使用
    result6 = test_memory_usage(data_manager)
    test_results.append(("内存使用", result6))
    
    # 输出测试总结
    print("\n" + "=" * 80)
    print("📋 测试总结")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！数据加载模块工作正常。")
    else:
        print("⚠️  部分测试失败，请检查相关配置和数据。")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
