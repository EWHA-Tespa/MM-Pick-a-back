#!/bin/bash

# 사용법: ./find_backbone.sh dataset_name
DATASET_NAME=$1  

echo "Running with dataset config: $DATASET_NAME"  
GPU_ID=0
ARCH='perceiver'
for TARGET_ID in {15..20}; do
    echo "Find backbone for task $TARGET_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 pickaback_cifar100.py --dataset_config $DATASET_NAME \
                --arch $ARCH \
                --target_id $TARGET_ID
done   