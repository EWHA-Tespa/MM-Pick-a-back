#!/bin/bash

# 사용법: ./run_experiment.sh dataset_config
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 dataset_config"
    exit 1
fi

DATASET_CONFIG=$1

GPU_ID=0
ARCH='perceiver_io'
FINETUNE_EPOCHS=60
ARCH='perceiver_io'
FINETUNE_EPOCHS=60
seed=2

####################
##### Baseline #####
####################
TASK_ID=4

for TASK_ID in {17..20}; do
CUDA_VISIBLE_DEVICES=$GPU_ID python3 packnet_cifar100_main_normal.py \
    --arch $ARCH \
    --dataset ${DATASETS[TASK_ID]} --num_classes 5 \
    --lr 1e-2 \
    --weight_decay 4e-5 \
    --save_folder checkpoints_${ARCH}/baseline_scratch/$ARCH/${DATASETS[TASK_ID]} \
    --epochs $FINETUNE_EPOCHS \
    --mode finetune \
    --logfile logs_${ARCH}/baseline_cifar100_acc_scratch.txt \
    --seed $seed        
done
for TASK_ID in {1..15}; do  # change according to the number of classes in the dataset
    DATASET=$(python3 get_dataset_name.py $DATASET_CONFIG $TASK_ID)
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 packnet_cifar100_main_normal.py \
        --arch $ARCH \
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
