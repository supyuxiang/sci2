#!/bin/bash

set -x

# 配置参数
GPU_ID=1  # 主GPU（使用空闲的GPU）
GPU_DEVICE_IDS="1,2"  # 所有GPU ID列表（使用空闲的GPU）
MODEL_NAME="EnhancedCARNet"
SCHEDULER="ReduceLROnPlateau"

# 运行单任务训练
python3 /data1/chzhang/sci826/src/src_main_dim2/train0.py \
    --model_0 "$MODEL_NAME" \
    --scheduler_0 "$SCHEDULER" \
    --device_0 "cuda:$GPU_ID" \
    --gpu_device_ids_0 "$GPU_DEVICE_IDS"
