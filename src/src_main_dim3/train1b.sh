#!/bin/bash

set -x

# 配置参数
GPU_ID=4  # 主GPU
GPU_DEVICE_IDS="4,5,6,7"  # 所有GPU ID列表
MODEL_NAME="EnhancedCARNet"
SCHEDULER="ReduceLROnPlateau"

# 运行多任务训练
python3 /data1/chzhang/sci826/src/src_main_dim3/train_1b.py \
    --model_1 "$MODEL_NAME" \
    --scheduler_1 "$SCHEDULER" \
    --device_1 "cuda:$GPU_ID" \
    --gpu_device_ids_1 "$GPU_DEVICE_IDS"
