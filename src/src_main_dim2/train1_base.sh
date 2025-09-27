#!/bin/bash

set -x

# 配置参数
GPU_ID=3  # 主GPU（使用空闲的GPU）
GPU_DEVICE_IDS="3,4,5,2"  # 所有GPU ID列表（使用空闲的GPU）
MODEL_NAME="EnhancedCARNet"
SCHEDULER="ReduceLROnPlateau"

# 运行多任务训练
python3 /data1/chzhang/sci826/src/src_main_dim2/train1_base.py \
    --model_1 "$MODEL_NAME" \
    --scheduler_1 "$SCHEDULER" \
    --device_1 "cuda:$GPU_ID" \
    --gpu_device_ids_1 "$GPU_DEVICE_IDS"
