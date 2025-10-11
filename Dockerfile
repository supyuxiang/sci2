# 使用Ubuntu 22.04作为基础镜像
FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# 设置conda路径
ENV PATH=/opt/conda/bin:${PATH}

# 创建fyx_sci环境
RUN conda create -n fyx_sci python=3.10 -y

# 激活环境并安装PyTorch (CUDA版本)
RUN /bin/bash -c "source activate fyx_sci && \
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y"

# 安装数据科学包
RUN /bin/bash -c "source activate fyx_sci && \
    conda install pandas scikit-learn scipy matplotlib seaborn openpyxl tqdm -c conda-forge -y"

# 安装配置管理
RUN /bin/bash -c "source activate fyx_sci && \
    conda install hydra-core -c conda-forge -y"

# 安装物理库
RUN /bin/bash -c "source activate fyx_sci && \
    conda install coolprop -c conda-forge -y"

# 安装实验跟踪
RUN /bin/bash -c "source activate fyx_sci && \
    pip install swanlab"

# 设置工作目录
WORKDIR /workspace

# 复制项目文件
COPY . /workspace/

# 设置conda环境激活脚本
RUN echo "source activate fyx_sci" >> ~/.bashrc

# 默认命令
CMD ["/bin/bash"]
