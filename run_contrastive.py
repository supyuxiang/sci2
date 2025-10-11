#!/usr/bin/env python3
"""
对比学习增强预测器运行脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """主函数"""
    print("=" * 60)
    print("对比学习增强预测器")
    print("=" * 60)
    
    # 检查是否在正确的目录
    if not Path("src/contrastive_learning").exists():
        print("❌ 错误: 请在项目根目录运行此脚本")
        sys.exit(1)
    
    # 运行对比学习主程序
    try:
        print("🚀 启动对比学习增强预测器...")
        
        # 切换到对比学习目录
        os.chdir("src/contrastive_learning")
        
        # 导入并运行主程序
        from main import main as contrastive_main
        contrastive_main()
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断训练")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        sys.exit(1)
    
    print("✅ 对比学习训练完成!")


if __name__ == "__main__":
    main()
