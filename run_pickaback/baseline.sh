#!/bin/bash

# 사용법: ./run_experiment.sh dataset_config
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 dataset_config"
    exit 1
fi

DATASET_CONFIG=$1

GPU_ID=0
ARCH='perceiver_io'
FINETUNE_EPOCHS=100
seed=2
EXPNAME='baseline'

for TASK_ID in {8..12}; do  # change according to the number of classes in the dataset
    if [ $TASK_ID -le 6 ]; then
        MODALITY='image'
    else
        MODALITY='text'
    fi
    DATASET=$(python3 get_dataset_name.py $DATASET_CONFIG $TASK_ID)
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 packnet_cifar100_main_normal.py \
        --arch $ARCH \
        --expname $EXPNAME \
        --modality $MODALITY \
        --dataset_config $DATASET_CONFIG \
        --dataset $DATASET \
        --num_classes -1 \
        --lr 1e-2 \
        --weight_decay 4e-5 \
        --save_folder checkpoints_${ARCH}/baseline_scratch/$ARCH/${DATASET_CONFIG}/${DATASET} \
        --epochs $FINETUNE_EPOCHS \
        --mode finetune \
        --logfile logs_${ARCH}/baseline_${DATASET_CONFIG}_acc.txt \
        --seed $seed
done
