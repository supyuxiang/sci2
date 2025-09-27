#!/bin/bash

set -x

# 配置参数
if command -v conda >/dev/null 2>&1; then
    # 尝试激活 conda 环境
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/data1/chzhang/anaconda3/etc/profile.d/conda.sh" ]; then
        source "/data1/chzhang/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate fyx_sci || true
fi

GPU_ID=4  # 主GPU
GPU_DEVICE_IDS="4,5,6,7"  # 所有GPU ID列表
MODEL_NAME="EnhancedCARNet"
SCHEDULER="ReduceLROnPlateau"

# 运行单任务训练
python3 /data1/chzhang/sci826/src/src_main_dim3/train0.py \
    --model_0 "$MODEL_NAME" \
    --scheduler_0 "$SCHEDULER" \
    --device_0 "cuda:$GPU_ID" \
    --gpu_device_ids_0 "$GPU_DEVICE_IDS" \
    --epochs_0 1000 \
    --batch_size_0 32 
