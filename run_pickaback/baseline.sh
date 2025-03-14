#!/bin/bash

# 사용법: ./run_experiment.sh dataset_config
# export CUDA_HOME=/usr/local/cuda-11.5
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# echo "CUDA_HOME: $CUDA_HOME"
# echo "PATH: $PATH"
# echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
# nvcc --version

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 dataset_config modality"
    exit 1
fi

DATASET_CONFIG=$1
MODALITY=$2

GPU_ID=0
ARCH='perceiver'
EXPIREMENT_GROUP='baseline'
FINETUNE_EPOCHS=100
seed=2

for TASK_ID in {1..15}; do  # change according to the number of classes in the dataset
    DATASET=$(python3 get_dataset_name.py $DATASET_CONFIG $TASK_ID)
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 packnet_cifar100_main_normal.py \
        --arch $ARCH \
        --modality $MODALITY \
        --experiment_group $EXPIREMENT_GROUP \
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
