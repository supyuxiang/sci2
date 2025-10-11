#!/usr/bin/env python3
"""
测试main.py脚本
确保在cuda:2上正常运行
"""

import sys
import os
import torch
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_cuda_availability():
    """测试CUDA可用性"""
    print("🔍 测试CUDA可用性")
    print("=" * 60)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"设备 {i}: {torch.cuda.get_device_name(i)}")
            print(f"  内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        # 测试cuda:2
        if torch.cuda.device_count() > 2:
            print(f"\n🎯 测试cuda:2...")
            device = torch.device('cuda:2')
            print(f"使用设备: {device}")
            
            # 创建测试张量
            test_tensor = torch.randn(1000, 1000).to(device)
            print(f"测试张量设备: {test_tensor.device}")
            print(f"测试张量形状: {test_tensor.shape}")
            
            # 简单计算测试
            result = torch.matmul(test_tensor, test_tensor.T)
            print(f"计算结果设备: {result.device}")
            print(f"计算结果形状: {result.shape}")
            
            print("✅ cuda:2 工作正常")
            return True
        else:
            print("❌ 没有足够的CUDA设备，cuda:2不可用")
            return False
    else:
        print("❌ CUDA不可用")
        return False


def test_main_imports():
    """测试main.py的导入"""
    print("\n" + "=" * 60)
    print("🔍 测试main.py导入")
    print("=" * 60)
    
    try:
        # 测试主要模块导入
        from src.data.data_manager import DataManager
        from src.utils.logger import Logger
        from src.models.models import ModelFactory
        from src.trainer.train import Trainer
        from src.utils.timer import Timer
        from src.utils.set_seed import set_seed
        
        print("✅ 所有主要模块导入成功")
        
        # 测试配置加载
        import yaml
        with open('src/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 配置文件加载成功")
        print(f"设备配置: {config['Train']['device']}")
        
        return True, config
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_data_loading_with_cuda():
    """测试数据加载和CUDA"""
    print("\n" + "=" * 60)
    print("🔍 测试数据加载和CUDA")
    print("=" * 60)
    
    try:
        from src.data.data_manager import DataManager
        from src.utils.logger import Logger
        import yaml
        
        # 加载配置
        with open('src/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建DataManager
        logger = Logger(name='test_main', config=config)
        data_path = config['Data']['data_path']
        data_manager = DataManager(config, logger, data_path)
        
        print("✅ DataManager创建成功")
        
        # 测试DataLoader
        device = torch.device(config['Train']['device'])
        print(f"使用设备: {device}")
        
        # 获取一个batch并移动到GPU
        for batch_idx, (features, targets) in enumerate(data_manager.dataloader_train):
            features_gpu = features.to(device)
            targets_gpu = targets.to(device)
            
            print(f"📊 Batch {batch_idx}:")
            print(f"   特征设备: {features_gpu.device}")
            print(f"   目标设备: {targets_gpu.device}")
            print(f"   特征形状: {features_gpu.shape}")
            print(f"   目标形状: {targets_gpu.shape}")
            
            # 简单计算测试
            dummy_output = torch.randn_like(targets_gpu)
            loss = torch.nn.functional.mse_loss(dummy_output, targets_gpu)
            print(f"   损失值: {loss.item():.6f}")
            print(f"   损失设备: {loss.device}")
            
            if batch_idx >= 1:  # 只测试前2个batch
                break
        
        print("✅ 数据加载和CUDA测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 数据加载和CUDA测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("🔍 测试模型创建")
    print("=" * 60)
    
    try:
        from src.models.models import ModelFactory
        from src.utils.logger import Logger
        import yaml
        
        # 加载配置
        with open('src/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger = Logger(name='test_model', config=config)
        
        # 创建模型
        model_factory = ModelFactory(logger=logger, phase=1, model_name='grmlp')
        model = model_factory.model
        
        print(f"✅ 模型创建成功: {type(model).__name__}")
        
        # 移动到GPU
        device = torch.device(config['Train']['device'])
        model = model.to(device)
        print(f"✅ 模型移动到设备: {next(model.parameters()).device}")
        
        # 测试前向传播
        dummy_input = torch.randn(32, 3).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ 前向传播成功")
        print(f"   输入形状: {dummy_input.shape}")
        print(f"   输出形状: {output.shape}")
        print(f"   输出设备: {output.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_execution():
    """测试main.py执行"""
    print("\n" + "=" * 60)
    print("🔍 测试main.py执行")
    print("=" * 60)
    
    try:
        # 设置环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        
        # 导入main模块
        import main
        
        print("✅ main.py导入成功")
        
        # 测试主要函数
        from main import set_logger, create_model, create_dataloaders
        
        # 加载配置
        import yaml
        with open('src/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 测试日志器
        logger = set_logger(config)
        print("✅ 日志器创建成功")
        
        # 测试数据加载器
        train_loader, test_loader, eval_loader = create_dataloaders(logger, config)
        print("✅ 数据加载器创建成功")
        
        # 测试模型创建
        model = create_model(logger, 'grmlp', 1)
        print("✅ 模型创建成功")
        
        print("✅ main.py所有组件测试成功")
        return True
        
    except Exception as e:
        print(f"❌ main.py执行测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 开始main.py测试")
    print("=" * 80)
    
    # 测试结果统计
    test_results = []
    
    # 测试1: CUDA可用性
    result1 = test_cuda_availability()
    test_results.append(("CUDA可用性", result1))
    
    if not result1:
        print("\n❌ CUDA不可用，终止测试")
        return
    
    # 测试2: 导入测试
    result2, config = test_main_imports()
    test_results.append(("导入测试", result2))
    
    if not result2:
        print("\n❌ 导入失败，终止测试")
        return
    
    # 测试3: 数据加载和CUDA
    result3 = test_data_loading_with_cuda()
    test_results.append(("数据加载和CUDA", result3))
    
    # 测试4: 模型创建
    result4 = test_model_creation()
    test_results.append(("模型创建", result4))
    
    # 测试5: main.py执行
    result5 = test_main_execution()
    test_results.append(("main.py执行", result5))
    
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
        print("🎉 所有测试通过！main.py可以在cuda:2上正常运行。")
    else:
        print("⚠️  部分测试失败，请检查相关配置。")


if __name__ == "__main__":
    main()
