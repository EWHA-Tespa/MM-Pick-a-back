#!/bin/bash

DATASETS=(
    'Art_Design'
    'Automobiles'
    'Books'
    'Dance'
    'Economy'
    'Education'
    'Fashion_Style'
    'Food'
    'Global_Business'
    'Health'
    'Media'
    'Movies'
    'Music'
    'Opinion'
    'Real_Estate'
    'Science'
    'Sports'
    'Style'
    'Technology'
    'Television'
    'Theater'
    'Travel'
    'Well'
    'Your_Money'
)

GPU_ID=0
ARCH='lenet5'
FINETUNE_EPOCHS=100
seed=2

####################
##### Baseline #####
####################
TASK_ID=4

CUDA_VISIBLE_DEVICES=$GPU_ID python3 packnet_cifar100_main_normal.py \
    --arch $ARCH \
    --dataset ${DATASETS[TASK_ID]} --num_classes 24 \
    --lr 1e-2 \
    --weight_decay 4e-5 \
    --save_folder checkpoints_${ARCH}/baseline_scratch/$ARCH/${DATASETS[TASK_ID]} \
    --epochs $FINETUNE_EPOCHS \
    --mode finetune \
    --logfile logs_${ARCH}/baseline_n24news_acc_scratch.txt \
    --seed $seed        
