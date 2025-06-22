#!/bin/bash

# 사용법: ./find_backbone.sh dataset_name
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 dataset_config"
    exit 1
fi
export TMPDIR=/data5/tmp_$USER
mkdir -p $TMPDIR

# wandb도 같은 임시폴더 사용
export WANDB_DIR=$TMPDIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


DATASET_NAME=$1  
echo "Running with dataset config: $DATASET_NAME"  

GPU_ID=0
ARCH='perceiver_io'

for TARGET_ID in 1; do
    if [ $TARGET_ID -le 6 ]; then
        MODALITY='image'
    else
        MODALITY='text'
    fi
    echo "Find backbone for task $TARGET_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID LOG_FILE=$LOG_FILE python3 pickaback_MM_no_train.py --dataset_config $DATASET_NAME \
                --arch $ARCH \
                --target_id $TARGET_ID \
                # --modality $MODALITY
done