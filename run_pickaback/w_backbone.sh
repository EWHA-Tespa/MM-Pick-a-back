#!/bin/bash

# 사용법: ./w_backbone.sh dataset_config
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 dataset_config"
    exit 1
fi

DATASET_CONFIG=$1

NUM_CLASSES=-1
INIT_LR=1e-2
PRUNING_LR=1e-3

GPU_ID=0
ARCH='perceiver'
EXPNAME='w_backbone'

FINETUNE_EPOCHS=100
DEFAULT_NETWORK_WIDTH_MULTIPLIER=1.0
NETWORK_WIDTH_MULTIPLIER=$DEFAULT_NETWORK_WIDTH_MULTIPLIER
MAX_NETWORK_WIDTH_MULTIPLIER=2.0
PRUNING_RATIO_INTERVAL=0.1
LR_MASK=1e-4
TOTAL_NUM_TASKS=5
seed=2

version_name='CPG_fromsingle_scratch_woexp_target'
single_version_name='CPG_single_scratch_woexp'
baseline_file="logs_${ARCH}/baseline_${DATASET_CONFIG}_acc.txt"
checkpoints_name="checkpoints_${ARCH}"
PICKABACK_CSV="pickaback_${DATASET_CONFIG}_result.csv"

# CSV 파일의 첫 행(헤더)을 건너뛰고, 두 번째 행부터 읽습니다.
tail -n +2 "$PICKABACK_CSV" | while IFS=',' read -r csv_target_id csv_task_id; do
    # CSV에서 읽은 값에서 CR, 따옴표, 불필요한 공백 제거
    target_id=$(echo "$csv_target_id" | tr -d '\r"' | xargs)
    task_id=$(echo "$csv_task_id" | tr -d '\r"' | xargs)

    echo "Starting training for task_id=$task_id, target_id=$target_id"

    # 각각 get_dataset_name.py를 호출하여 dataset 이름을 가져옵니다.
    DATASET_TASK=$(python3 get_dataset_name.py $DATASET_CONFIG $task_id)
    DATASET_TARGET=$(python3 get_dataset_name.py $DATASET_CONFIG $target_id)

    echo "DATASET_TASK: $DATASET_TASK"
    echo "DATASET_TARGET: $DATASET_TARGET"

    # 네트워크 폭 초기화
    NETWORK_WIDTH_MULTIPLIER=$DEFAULT_NETWORK_WIDTH_MULTIPLIER

    state=2
    while [ $state -eq 2 ]; do
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 CPG_cifar100_main_normal.py \
           --arch $ARCH \
           --expname $EXPNAME \
           --dataset $DATASET_TASK --num_classes $NUM_CLASSES \
           --lr $INIT_LR \
           --lr_mask $LR_MASK \
           --weight_decay 4e-5 \
           --save_folder $checkpoints_name/$version_name/$ARCH/${DATASET_TASK}/${DATASET_TARGET}/scratch \
           --load_folder $checkpoints_name/$single_version_name/$ARCH/${DATASET_TASK}/gradual_prune \
           --epochs $FINETUNE_EPOCHS \
           --mode finetune \
           --network_width_multiplier $NETWORK_WIDTH_MULTIPLIER \
           --max_allowed_network_width_multiplier $MAX_NETWORK_WIDTH_MULTIPLIER \
           --pruning_ratio_to_acc_record_file $checkpoints_name/$version_name/$ARCH/${DATASET_TASK}/${DATASET_TARGET}/gradual_prune/record.txt \
           --jsonfile $baseline_file \
           --log_path $checkpoints_name/$version_name/$ARCH/${DATASET_TASK}/${DATASET_TARGET}/train.log \
           --total_num_tasks $TOTAL_NUM_TASKS \
           --seed $seed

        state=$?
        if [ $state -eq 2 ]; then
            if [[ "$NETWORK_WIDTH_MULTIPLIER" == "$MAX_NETWORK_WIDTH_MULTIPLIER" ]]; then
                break
            fi
            
            NETWORK_WIDTH_MULTIPLIER=$(bc <<< $NETWORK_WIDTH_MULTIPLIER+0.5)
            echo "New network_width_multiplier: $NETWORK_WIDTH_MULTIPLIER"
            continue
        elif [ $state -eq 3 ]; then
            echo "You should provide the baseline_${DATASET_CONFIG}_acc.txt as criterion to decide whether the capacity of network is enough for new task"
            exit 0
        fi
    done
done
