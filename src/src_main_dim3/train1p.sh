#!/bin/bash

set -x

# 配置参数
GPU_ID=4  # 主GPU（必须是GPU列表中的第一个）
GPU_DEVICE_IDS="4,5,6,7"  # 所有GPU ID列表

# 运行物理约束训练
python3 /data1/chzhang/sci826/src/src_main_dim3/train1_p.py \
    --device_1 "cuda:$GPU_ID" \
    --gpu_device_ids_1 "$GPU_DEVICE_IDS"
