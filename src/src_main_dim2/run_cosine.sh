#!/bin/bash

set -x

# 配置参数
GPU_ID=2  # 主GPU（使用空闲的GPU）
GPU_DEVICE_IDS="2,3,4,5"  # 所有GPU ID列表（使用空闲的GPU）
MODEL_NAME="EnhancedCARNet"
SCHEDULER="CosineAnnealingLR"

# 运行单任务训练
python3 /data1/chzhang/sci826/src/src_main_dim2/train0.py \
    --model_0 "$MODEL_NAME" \
    --scheduler_0 "$SCHEDULER" \
    --device_0 "cuda:$GPU_ID" \
    --gpu_device_ids_0 "$GPU_DEVICE_IDS"

# 运行多任务训练
python3 /data1/chzhang/sci826/src/src_main_dim2/train1_base.py \
    --model_1 "$MODEL_NAME" \
    --scheduler_1 "$SCHEDULER" \
    --device_1 "cuda:$GPU_ID" \
    --gpu_device_ids_1 "$GPU_DEVICE_IDS"
