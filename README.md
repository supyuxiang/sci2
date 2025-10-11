# SCI2 项目

这是一个基于PyTorch的科学计算项目，支持CUDA加速和实验跟踪。

## 环境要求

- Python 3.10
- CUDA 12.4+ (可选，用于GPU加速)
- 10GB+ 可用磁盘空间

## 快速开始

### 1. 自动安装环境

```bash
# 运行自动安装脚本
bash setup_environment.sh
```

### 2. 手动安装环境

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

### 3. 使用pip安装

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
```

## 项目结构

```
sci2/
├── main.py                 # 主程序入口
├── requirements.txt        # Python依赖
├── setup_environment.sh    # 环境安装脚本
├── src/                    # 源代码
│   ├── config.yaml        # 配置文件
│   ├── data/              # 数据处理
│   ├── models/            # 模型定义
│   ├── trainer/           # 训练器
│   └── utils/             # 工具函数
├── scripts/               # 测试脚本
└── data/                  # 数据文件
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
