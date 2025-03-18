#!/bin/bash

# 사용법: ./select_pruning_ratio_of_backbone.sh dataset_config
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 dataset_config"
    exit 1
fi

DATASET_CONFIG=$1

for TASK_ID in {1..12}; do
    # TARGET_TASK_ID=1

    # 각 TASK_ID에 맞는 데이터셋 이름을 가져옴
    DATASET=$(python3 get_dataset_name.py $DATASET_CONFIG $TASK_ID)

    GPU_ID=0
    arch='perceiver'
    FINETUNE_EPOCHS=100
    NUM_CLASSES=-1   
    INIT_LR=1e-2
    PRUNING_LR=1e-3
    LR_MASK=1e-4
    NETWORK_WIDTH_MULTIPLIER=1.0
    MAX_NETWORK_WIDTH_MULTIPLIER=1.0
    PRUNING_RATIO_INTERVAL=0.1
    TOTAL_NUM_TASKS=5
    seed=2

    VERSION_NAME='CPG_single_scratch_woexp'
    CHECKPOINTS_NAME="checkpoints_${arch}"
    BASELINE_FILE="logs_${arch}/baseline_${DATASET_CONFIG}_acc.txt"

    python3 tools/choose_appropriate_pruning_ratio_for_next_task_notrmfiles.py \
        --pruning_ratio_to_acc_record_file ${CHECKPOINTS_NAME}/${VERSION_NAME}/${arch}/${DATASET_CONFIG}/${DATASET}/gradual_prune/record.txt \
        --baseline_acc_file $BASELINE_FILE \
        --allow_acc_loss 0.001 \
        --dataset $DATASET \
        --network_width_multiplier $NETWORK_WIDTH_MULTIPLIER \
        --max_allowed_network_width_multiplier $MAX_NETWORK_WIDTH_MULTIPLIER \
        --log_path ${CHECKPOINTS_NAME}/${VERSION_NAME}/${arch}/${DATASET_CONFIG}/${DATASET}/train.log
done
