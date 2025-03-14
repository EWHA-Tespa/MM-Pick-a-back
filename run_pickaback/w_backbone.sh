#!/bin/bash


TARGET_TASK_ID=1

dataset=(
    'None'     # dummy
    'aquatic_mammals' #1
    'fish' #2
    'flowers' #3
    'food_containers' #4
    'fruit_and_vegetables' #5
    'household_electrical_devices' #6
    'household_furniture' #7
    'insects' #8
    'large_carnivores' #9
    'large_man-made_outdoor_things' #10
    'large_natural_outdoor_scenes' #11
    'large_omnivores_and_herbivores' #12
    'medium_mammals' #13
    'non-insect_invertebrates' #14
    'people' #15
    'reptiles' #16
    'small_mammals' #17
    'trees' #18
    'vehicles_1' #19
    'vehicles_2' #20
)

num_classes=(
    5
)

init_lr=(
    1e-2
)

pruning_lr=(
    1e-3
)

GPU_ID=0
arch='perceiver'
finetune_epochs=100
network_width_multiplier=1.0
max_network_width_multiplier=2.0
pruning_ratio_interval=0.1
lr_mask=1e-4
total_num_tasks=5

task_id=4 # backbone
target_id=9 
seed=2

echo "task_id=$task_id, target_id=$target_id"

version_name='CPG_fromsingle_scratch_woexp_target'
single_version_name='CPG_single_scratch_woexp'
baseline_file="logs_${arch}/baseline_cifar100_acc_scratch.txt"
checkpoints_name="checkpoints_${arch}"
PICKABACK_CSV="checkpoints_${arch}/pickaback_result.csv"
####################
##### Training #####
####################
state=2
while [ $state -eq 2 ]; do
        CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_cifar100_main_normal.py \
           --arch $arch \
           --dataset ${dataset[target_id]} --num_classes ${num_classes[0]} \
           --lr ${init_lr[0]} \
           --lr_mask $lr_mask \
           --weight_decay 4e-5 \
           --save_folder $checkpoints_name/$version_name/$arch/${dataset[task_id]}/${dataset[target_id]}/scratch \
           --load_folder $checkpoints_name/$single_version_name/$arch/${dataset[task_id]}/gradual_prune \
           --epochs $finetune_epochs \
           --mode finetune \
           --network_width_multiplier $network_width_multiplier \
           --max_allowed_network_width_multiplier $max_network_width_multiplier \
           --pruning_ratio_to_acc_record_file $checkpoints_name/$version_name/$arch/${dataset[task_id]}/${dataset[target_id]}/gradual_prune/record.txt \
           --jsonfile $baseline_file \
           --log_path $checkpoints_name/$version_name/$arch/${dataset[task_id]}/${dataset[target_id]}/train.log \
           --total_num_tasks $total_num_tasks \
           --seed $seed

    state=$?
    if [ $state -eq 2 ]
    then
        if [[ "$network_width_multiplier" == "$max_network_width_multiplier" ]]
        then
            break
        fi
        network_width_multiplier=$(bc <<< $network_width_multiplier+0.5)
        echo "New network_width_multiplier: $network_width_multiplier"
        continue
    elif [ $state -eq 3 ]
    then
        echo "You should provide the baseline_cifar100_acc.txt as criterion to decide whether the capacity of network is enough for new task"
        exit 0
    fi
done
