#!/bin/bash

# 设置fyx_sci环境的安装脚本
# 使用方法: bash setup_environment.sh

echo "🚀 开始设置fyx_sci环境..."

# 删除现有环境（如果存在）
echo "🗑️  删除现有环境..."
conda env remove -n fyx_sci -y

# 创建新环境
echo "📦 创建新环境..."
conda create -n fyx_sci python=3.10 -y

# 激活环境
echo "🔧 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fyx_sci

# 安装PyTorch
echo "🔥 安装PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 安装数据科学包
echo "📊 安装数据科学包..."
conda install pandas scikit-learn scipy matplotlib seaborn openpyxl tqdm -c conda-forge -y

# 安装配置管理
echo "⚙️  安装配置管理..."
conda install hydra-core -c conda-forge -y

# 安装物理库
echo "🧪 安装物理库..."
conda install coolprop -c conda-forge -y

# 安装实验跟踪
echo "📈 安装实验跟踪..."
pip install --user swanlab

echo "✅ 环境设置完成！"
echo "使用方法: conda activate fyx_sci"
echo "测试: python scripts/test_main.py"
