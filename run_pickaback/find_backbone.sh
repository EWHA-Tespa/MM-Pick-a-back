#!/bin/bash

# 사용법: ./find_backbone.sh dataset_name
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 dataset_config"
    exit 1
fi

DATASET_NAME=$1  

echo "Running with dataset config: $DATASET_NAME"  

GPU_ID=5
ARCH='perceiver'

for TARGET_ID in {1..6}; do
    echo "Find backbone for task $TARGET_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID LOG_FILE=$LOG_FILE python3 pickaback_cifar100.py --dataset_config $DATASET_NAME \
                --arch $ARCH \
                --target_id $TARGET_ID
done
