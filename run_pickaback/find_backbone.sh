#!/bin/bash

# 사용법: ./find_backbone.sh dataset_name
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 dataset_config"
    exit 1
fi

DATASET_NAME=$1  

echo "Running with dataset config: $DATASET_NAME"  

GPU_ID=0
ARCH='perceiver'

for TARGET_ID in {1..12}; do
    if [ $TASK_ID -le 6 ]; then
        MODALITY='image'
    else
        MODALITY='text'
    fi
    MODALITY='image'
    echo "Find backbone for task $TARGET_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID LOG_FILE=$LOG_FILE python3 pickaback_MM.py --dataset_config $DATASET_NAME \
                --arch $ARCH \
                --target_id $TARGET_ID \
                --modality $MODALITY
done
