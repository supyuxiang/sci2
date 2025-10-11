# SCI2: Scientific Computing with Intelligent Neural Networks

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Research-purple.svg)](LICENSE)

## 项目简介

SCI2是一个先进的科学计算项目，专注于流体动力学仿真和物理信息神经网络（PINN）研究。项目集成了多种深度学习架构，包括MLP、LSTM、GRU、Transformer等，并支持对比学习增强预测器，为科学计算提供强大的AI驱动解决方案。

### 核心特性

- 🧠 **多架构支持**: MLP、LSTM、GRU、CNN、Transformer、Wide&Deep等
- 🔬 **物理信息神经网络**: 集成物理约束的PINN框架
- 🎯 **对比学习增强**: 先进的对比学习预测器提升模型性能
- ⚡ **GPU加速**: 完整的CUDA支持，支持多GPU训练
- 📊 **实验跟踪**: 集成SwanLab进行实验管理和可视化
- 🐳 **容器化部署**: 完整的Docker支持，一键部署
- 🔧 **配置管理**: 基于Hydra的灵活配置系统

## 环境要求

### 系统要求
- Python 3.10+
- CUDA 12.4+ (可选，用于GPU加速)
- 10GB+ 可用磁盘空间
- Docker (可选，用于容器化部署)

### 硬件推荐
- NVIDIA GPU with 8GB+ VRAM (推荐RTX 3080或更高)
- 16GB+ RAM
- 多核CPU

## 快速开始

### 方式一：Docker部署 (推荐)

#### 1. 构建Docker镜像

```bash
# 构建镜像
docker build -t sci2:latest .

# 或使用docker-compose
docker-compose build
```

#### 2. 运行容器

```bash
# 直接运行
docker run -it --gpus all -v $(pwd):/workspace sci2:latest

# 或使用docker-compose
docker-compose up -d
docker-compose exec sci2 bash
```

#### 3. 在容器内运行项目

```bash
# 激活conda环境
conda activate fyx_sci

# 运行训练
python main.py

# 运行测试
python scripts/test_main.py
```

### 方式二：本地安装

#### 1. 自动安装环境

```bash
# 运行自动安装脚本
bash setup_environment.sh
```

#### 2. 手动安装环境

```bash
# 创建conda环境
conda create -n fyx_sci python=3.10 -y

# 激活环境
conda activate fyx_sci

# 安装PyTorch (CUDA版本)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 安装其他依赖
conda install pandas scikit-learn scipy matplotlib seaborn openpyxl tqdm -c conda-forge -y
conda install hydra-core coolprop -c conda-forge -y
pip install --user swanlab
```

#### 3. 使用pip安装

```bash
# 激活环境
conda activate fyx_sci

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 激活环境

```bash
conda activate fyx_sci
```

### 运行测试

```bash
# 测试所有功能
python scripts/test_main.py

# 测试数据加载
python scripts/test_data_loading.py
```

### 运行主程序

```bash
# 使用默认配置运行
python main.py

# 使用自定义配置
python main.py --config-path=src --config-name=config

# 运行对比学习增强预测器
python main.py --config-path=src/contrastive_learning --config-name=config
```

### Docker使用

```bash
# 构建并运行
docker-compose up --build

# 后台运行
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 项目结构

```
sci2/
├── main.py                    # 主程序入口
├── requirements.txt           # Python依赖
├── setup_environment.sh       # 环境安装脚本
├── Dockerfile                 # Docker镜像构建文件
├── docker-compose.yml         # Docker Compose配置
├── .dockerignore              # Docker忽略文件
├── src/                       # 源代码
│   ├── config.yaml           # 主配置文件
│   ├── data/                 # 数据处理模块
│   │   └── data_manager.py   # 数据管理器
│   ├── models/               # 模型定义
│   │   ├── models.py         # 模型工厂
│   │   ├── mlp.py           # MLP模型
│   │   ├── lstm.py          # LSTM模型
│   │   ├── transformer.py   # Transformer模型
│   │   └── ...              # 其他模型
│   ├── trainer/              # 训练器
│   │   └── train.py         # 训练逻辑
│   ├── utils/                # 工具函数
│   │   ├── logger.py        # 日志系统
│   │   ├── optimizer.py     # 优化器
│   │   ├── loss_function.py # 损失函数
│   │   └── ...              # 其他工具
│   ├── contrastive_learning/ # 对比学习模块
│   │   ├── config.yaml      # 对比学习配置
│   │   ├── contrastive_model.py # 对比学习模型
│   │   ├── contrastive_trainer.py # 对比学习训练器
│   │   └── ...              # 对比学习相关代码
│   └── adversarial_learning/ # 对抗学习模块
├── scripts/                  # 测试脚本
│   └── test_main.py         # 主测试脚本
└── data/                     # 数据文件
    └── 数据100mm流体域.xlsx  # 流体动力学数据
```

## 配置说明

主要配置在 `src/config.yaml` 中：

- **Data**: 数据路径和参数
- **Model**: 模型配置
- **Train**: 训练参数
- **SwanLab**: 实验跟踪配置

## GPU使用

项目默认使用 `cuda:2` 设备。可以在配置文件中修改：

```yaml
Train:
  device: 'cuda:2'  # 或 'cpu', 'cuda:0', 'cuda:1' 等
```

## 实验跟踪

项目集成了SwanLab进行实验跟踪：

1. 在配置文件中启用SwanLab
2. 设置项目名称和描述
3. 运行训练时会自动记录指标

## 故障排除

### 常见问题

1. **CUDA不可用**: 检查CUDA驱动和PyTorch安装
2. **依赖冲突**: 重新创建conda环境
3. **权限问题**: 使用 `--user` 标志安装包

### 重新安装环境

```bash
# 删除现有环境
conda env remove -n fyx_sci -y

# 重新运行安装脚本
bash setup_environment.sh
```

## 版本信息

- PyTorch: 2.5.1
- CUDA: 12.4
- Python: 3.10
- Hydra: 0.11.3
- CoolProp: 6.4.1

## 许可证

本项目仅供学习和研究使用。
