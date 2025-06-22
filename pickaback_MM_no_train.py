import argparse
import json
import warnings
import logging
import os
import sys
import math
import copy
import pdb
import csv

import numpy as np
from tqdm import tqdm
from scipy import spatial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.parameter import Parameter

import yaml

def get_kv_modules(perceiver_model, modality):
    if modality=='image':   
        kv_proj = perceiver_model.cross_attend_blocks[0].fn.to_kv_image
    else:
        kv_proj = perceiver_model.cross_attend_blocks[0].fn.to_kv_text
    return kv_proj

# def set_kv_modules(perceiver_model, new_kv, modality):
#     if modality == 'image':
#         perceiver_model.cross_attend_blocks[0].fn.to_kv_image = new_kv
#     elif modality == 'text':
#         perceiver_model.cross_attend_blocks[0].fn.to_kv_text = new_kv
#     else:
#         raise ValueError("modality must be 'image' or 'text'")
#     return perceiver_model

def set_kv_modules(m, kv_src, modality):
    block = m.cross_attend_blocks[0].fn
    tgt = block.to_kv_image if modality == 'image' else block.to_kv_text
    with torch.no_grad():
        tgt.weight.copy_(kv_src.weight)
        if tgt.bias is not None:
            tgt.bias.copy_(kv_src.bias)
    return m

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='lenet5', choices=['lenet5', 'perceiver_io'])
parser.add_argument('--dataset', type=str, default='', help='Name of dataset (or subfolder for datasets with subfolders)')
parser.add_argument('--dataset_config', type=str, default='oxford', choices=["cifar100", "n24news", "mscoco", "cub", "oxford"],
                   help='Dataset configuration key defined in dataset_config.yaml (e.g., cifar100, n24news)')
parser.add_argument('--target_id', type=int, default=1)
parser.add_argument('--modality', type=str, default='image', help='Modality of data')
args = parser.parse_args()

print(f"Architecture: {args.arch}")
print(f"Received dataset_config: '{args.dataset_config}'") 
print(f"Target ID: {args.target_id}")
# dataset_config.yaml 파일 경로 설정
config_file = "utils/dataset_config.yaml"

# YAML 파일에서 전체 데이터셋 설정 불러오기
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# args.dataset_config는 실제로는 dataset_config.yaml에 존재하는 키 이름
dataset_config = config.get(args.dataset_config)

# 해당 데이터셋 설정이 존재하는지 확인
if not dataset_config:
    raise ValueError(f"Dataset configuration '{args.dataset_config}' not found in dataset_config.yaml")

# 데이터셋 설정에서 DATASETS 및 num_classes 추출
DATASETS = dataset_config["DATASETS"]
num_classes_in_config = dataset_config["num_classes"] # 왜 num_class가 그룹개수인거지...?
num_groups = dataset_config["num_groups"]

start_index = 1

from utils_pickaback.dataset import train_loader as train_loader_fn
from utils_pickaback.dataset import val_loader as val_loader_fn

import utils_pickaback as utils
from utils_pickaback.packnet_manager import Manager
import packnet_models_pickaback as packnet_models
################################
# 기본 설정
################################
arch = args.arch
num_classes = -1
lr = 0.1
batch_size = 32
val_batch_size = 100
workers = 24
weight_decay = 4e-5
train_path = ''
val_path = ''
cuda = True
seed = 1
checkpoint_format = './{save_folder}/checkpoint-{epoch}.pth.tar'
epochs = 160
restore_epoch = 0
save_folder = ''
load_folder = ''
one_shot_prune_perc = 0.5
mode = ''
logfile = ''
initial_from_task = ''

################################
val_batch_size = 32              # CUDA Out of memory 뜨면 여길 바꿀 것
epsilon = 0.1
max_iterations = 100
################################

# target dataset 인덱스 (DATASETS 목록 내 번호; 필요에 따라 조정)
target_id = args.target_id

table_rows = []
ddvcc_list = []
ddvec_list = []
tasks = [] 

# Iterate over the datasets
for task_id in range(start_index, num_groups + 1):

    if task_id == target_id:
        continue
    
    # target_modality='image'
    # task_modality='image'
    if target_id <= (num_groups / 2):
        target_modality = 'image'
    else:
        target_modality = 'text'

    if task_id <= (num_groups / 2):
        task_modality = 'image'
    else:
        task_modality = 'text'
    
    dataset_name = DATASETS[task_id]
    dataset_name_target = DATASETS[target_id]
    dataset_name_test = DATASETS[task_id]
    dataset_name_test_target = DATASETS[target_id]
    num_classes = dataset_config["num_classes"]
    lr = 1e-2
    weight_decay = 4e-5
    load_folder = 'checkpoints_' + arch + '/baseline_scratch/' + arch + '/' + args.dataset_config + '/' + dataset_name
    load_folder2 = 'checkpoints_' + arch + '/baseline_scratch/' + arch + '/' + args.dataset_config + '/' + dataset_name_target
    epochs = 100
    mode = 'inference'
    logfile = 'logs' + arch + '/baseline_oxford_acc_temp.txt'

    if save_folder and not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        cuda = False

    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    resume_from_epoch = 0
    resume_folder = load_folder
    for try_epoch in range(200, 0, -1):
        if os.path.exists(checkpoint_format.format(save_folder=resume_folder, epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    resume_from_epoch2 = 0
    resume_folder2 = load_folder2
    for try_epoch2 in range(200, 0, -1):
        if os.path.exists(checkpoint_format.format(save_folder=resume_folder2, epoch=try_epoch2)):
            resume_from_epoch2 = try_epoch2
            break

    if restore_epoch:
        resume_from_epoch = restore_epoch
        resume_from_epoch2 = restore_epoch

    if resume_from_epoch == 0:
        logging.warning(f"[{dataset_name}] checkpoint not found! using random init")
    if resume_from_epoch:
        filepath = checkpoint_format.format(save_folder=resume_folder, epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        checkpoint_keys = checkpoint.keys()
        dataset_history = checkpoint['dataset_history']
        dataset2num_classes = checkpoint['dataset2num_classes']
        masks = checkpoint['masks']
        if 'shared_layer_info' in checkpoint_keys:
            shared_layer_info = checkpoint['shared_layer_info']
        else:
            shared_layer_info = {}
        if 'num_for_construct' in checkpoint_keys:
            num_for_construct = checkpoint['num_for_construct']
    else:
        dataset_history = []
        dataset2num_classes = {}
        masks = {}
        shared_layer_info = {}
    if resume_from_epoch == 0:
        logging.warning(f"[{dataset_name_target}] checkpoint not found! using random init")
    if resume_from_epoch2:
        filepath2 = checkpoint_format.format(save_folder=resume_folder2, epoch=resume_from_epoch2)
        checkpoint2 = torch.load(filepath2)
        checkpoint_keys2 = checkpoint2.keys()
        dataset_history2 = checkpoint2['dataset_history']
        dataset2num_classes2 = checkpoint2['dataset2num_classes']
        masks2 = checkpoint2['masks']
        if 'shared_layer_info' in checkpoint_keys2:
            shared_layer_info2 = checkpoint2['shared_layer_info']
        else:
            shared_layer_info2 = {}
        if 'num_for_construct' in checkpoint_keys2:
            num_for_construct2 = checkpoint2['num_for_construct']
    else:
        dataset_history2 = []
        dataset2num_classes2 = {}
        masks2 = {}
        shared_layer_info2 = {}

    if arch == 'vgg16_bn_cifar100':
        model = packnet_models.__dict__[arch](
            pretrained=False,
            dataset_history=dataset_history,
            dataset2num_classes=dataset2num_classes
        )
        model2 = packnet_models.__dict__[arch](
            pretrained=False,
            dataset_history=dataset_history2,
            dataset2num_classes=dataset2num_classes2
        )
    elif arch == 'lenet5':
        custom_cfg = [6, 'A', 16, 'A']
        model = packnet_models.__dict__[arch](custom_cfg,
                                              dataset_history=dataset_history,
                                              dataset2num_classes=dataset2num_classes)
        model2 = packnet_models.__dict__[arch](custom_cfg,
                                               dataset_history=dataset_history2,
                                               dataset2num_classes=dataset2num_classes2)
    elif arch == 'mobilenetv1':
        model = packnet_models.__dict__[arch]([],
                                              dataset_history=dataset_history,
                                              dataset2num_classes=dataset2num_classes)
        model2 = packnet_models.__dict__[arch]([],
                                               dataset_history=dataset_history2,
                                               dataset2num_classes=dataset2num_classes2)
    elif 'mobilenetv2' in arch:
        model = packnet_models.__dict__[arch]([],
                                              dataset_history=dataset_history,
                                              dataset2num_classes=dataset2num_classes)
        model2 = packnet_models.__dict__[arch]([],
                                               dataset_history=dataset_history2,
                                               dataset2num_classes=dataset2num_classes2)
    elif 'efficientnet' in arch:
        model = packnet_models.__dict__[arch]([],
                                              dataset_history=dataset_history,
                                              dataset2num_classes=dataset2num_classes)
        model2 = packnet_models.__dict__[arch]([],
                                               dataset_history=dataset_history2,
                                               dataset2num_classes=dataset2num_classes2)
    elif arch == 'resnet50':
        model = packnet_models.__dict__[arch](dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
        model2 = packnet_models.__dict__[arch](dataset_history=dataset_history2, dataset2num_classes=dataset2num_classes2)
    elif arch == 'perceiver':
        image_input_channels=3
        image_input_axis=2
        text_input_channels=768
        text_input_axis=1
        model = packnet_models.__dict__[arch](
                                    num_freq_bands=6,
                                    depth=5,
                                    max_freq=10,
                                    image_input_channels=image_input_channels,
                                    image_input_axis=image_input_axis,
                                    text_input_channels=text_input_channels,
                                    text_input_axis=text_input_axis,
                                    num_latents=256,
                                    latent_dim=512,
                                    cross_heads=1,
                                    latent_heads=8,
                                    cross_dim_head=64,
                                    latent_dim_head=64,
                                    attn_dropout=0.,
                                    ff_dropout=0.,
                                    weight_tie_layers=False,
                                    fourier_encode_data=True,
                                    self_per_cross_attn=1,
                                    final_classifier_head=False,
                                    dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
        model2 = packnet_models.__dict__[arch](
                                    num_freq_bands=6,
                                    depth=5,
                                    max_freq=10,
                                    image_input_channels=image_input_channels,
                                    image_input_axis=image_input_axis,
                                    text_input_channels=text_input_channels,
                                    text_input_axis=text_input_axis,
                                    num_latents=256,
                                    latent_dim=512,
                                    cross_heads=1,
                                    latent_heads=8,
                                    cross_dim_head=64,
                                    latent_dim_head=64,
                                    attn_dropout=0.,
                                    ff_dropout=0.,
                                    weight_tie_layers=False,
                                    fourier_encode_data=True,
                                    self_per_cross_attn=1,
                                    final_classifier_head=False,
                                    dataset_history=dataset_history2, dataset2num_classes=dataset2num_classes2)
        model.set_modality(task_modality)
        model2.set_modality(target_modality)
    elif arch == 'perceiver_io':
        image_input_channels=3
        image_input_axis=2
        text_input_axis=1
        text_input_channels=768
        model = packnet_models.__dict__[args.arch](
            num_freq_bands=6,
            depth=4,
            max_freq=10,
            init_weights=True,
            image_input_channels=image_input_channels,
            image_input_axis=image_input_axis,
            text_input_channels=text_input_channels,
            text_input_axis=text_input_axis,
            max_text_length=512,
            queries_dim=512, 
            dataset_history=dataset_history2, 
            dataset2num_classes=dataset2num_classes2, 
            num_latents=256,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
            fourier_encode_data=True,
            decoder_ff=True,
            final_classifier_head=False
        )
        model2 = packnet_models.__dict__[args.arch](
            num_freq_bands=6,
            depth=4,
            max_freq=10,
            init_weights=True,
            image_input_channels=image_input_channels,
            image_input_axis=image_input_axis,
            text_input_channels=text_input_channels,
            text_input_axis=text_input_axis,
            max_text_length=512,
            queries_dim=512, 
            dataset_history=dataset_history, 
            dataset2num_classes=dataset2num_classes, 
            num_latents=256,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
            fourier_encode_data=True,
            decoder_ff=True,
            final_classifier_head=False
        )
        model.set_modality(task_modality)
        model2.set_modality(target_modality)

    else:
        print('Error!')
        sys.exit(0)

    model.add_dataset(dataset_name, num_classes)
    model.set_dataset(dataset_name)
    model2.add_dataset(dataset_name_target, num_classes)
    model2.set_dataset(dataset_name_target)

    if not masks:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if 'classifiers' in name:
                    continue
                mask_ = torch.ByteTensor(module.weight.data.size()).fill_(1)
                if 'cuda' in module.weight.data.type():
                    mask_ = mask_.cuda()
                masks[name] = mask_

    if not masks2:
        for name, module in model2.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if 'classifiers' in name:
                    continue
                mask_ = torch.ByteTensor(module.weight.data.size()).fill_(1)
                if 'cuda' in module.weight.data.type():
                    mask_ = mask_.cuda()
                masks2[name] = mask_

    if dataset_name not in shared_layer_info:
        shared_layer_info[dataset_name] = {
            'conv_bias': {},
            'bn_layer_running_mean': {},
            'bn_layer_running_var': {},
            'bn_layer_weight': {},
            'bn_layer_bias': {},
            'fc_bias': {}
        }
    if dataset_name_target not in shared_layer_info2:
        shared_layer_info2[dataset_name_target] = {
            'conv_bias': {},
            'bn_layer_running_mean': {},
            'bn_layer_running_var': {},
            'bn_layer_weight': {},
            'bn_layer_bias': {},
            'fc_bias': {}
        }

    model = model.cuda()
    model2 = model2.cuda()

    train_loader = train_loader_fn(args.dataset_config, batch_size, dataset_name) # task의 validset
    val_loader = val_loader_fn(args.dataset_config, val_batch_size, dataset_name)
    train_loader2 = train_loader_fn(args.dataset_config, batch_size, dataset_name_target) # target의 validset
    val_loader2 = val_loader_fn(args.dataset_config, val_batch_size, dataset_name_target)

    if save_folder != load_folder:
        start_epoch = 0
    else:
        start_epoch = resume_from_epoch

    manager = Manager(dataset_name,
                      checkpoint_format,
                      weight_decay,
                      cuda,
                      model,
                      shared_layer_info,
                      masks,
                      train_loader,
                      val_loader)

    manager2 = Manager(dataset_name_target,
                       checkpoint_format,
                       weight_decay,
                       cuda,
                       model2,
                       shared_layer_info2,
                       masks2,
                       train_loader2,
                       val_loader2)

    manager.load_checkpoint_for_inference(resume_from_epoch, resume_folder)
    manager2.load_checkpoint_for_inference(resume_from_epoch2, resume_folder2)

    print('======================')

    manager.pruner.apply_mask()
    manager.model.eval()
    manager2.pruner.apply_mask()
    manager2.model.eval()

    with torch.no_grad():
        data1, target1 = next(iter(manager.val_loader)) # task 모델
        data2, target2 = next(iter(manager2.val_loader)) # target모델
        if manager.cuda:
            data1, target1 = data1.cuda(), target1.cuda()
            data2, target2 = data2.cuda(), target2.cuda()
            # inputs = np.concatenate([data1.cpu(), data2.cpu()])
            # outputs1 = manager.model(torch.Tensor(inputs).cuda()).to('cpu').tolist()
            # outputs2 = manager2.model(torch.Tensor(inputs).cuda()).to('cpu').tolist()

            outputs1_1 = manager.model(data1.cuda()).to('cpu').numpy()
            outputs2_2 = manager2.model(data2.cuda()).to('cpu').numpy()

            ### kv projection layer 바꿔서 나온 output ###
            
            NewModel1 = copy.deepcopy(manager.model) # task모델
            NewModel2 = copy.deepcopy(manager2.model) # target모델

            kv_1 = get_kv_modules(manager.model, modality=task_modality)
            kv_2 = get_kv_modules(manager2.model, modality=target_modality)
            
            set_kv_modules(NewModel1, kv_2, modality=target_modality)
            set_kv_modules(NewModel2, kv_1,   modality=task_modality)
            NewModel1.set_modality(target_modality)
            NewModel2.set_modality(task_modality)

            outputs1_2 = NewModel1(data2.cuda()).to('cpu').numpy()
            outputs2_1 = NewModel2(data1.cuda()).to('cpu').numpy()

            outputs1 = np.concatenate([outputs1_1, outputs1_2], axis=0).tolist()
            outputs2 = np.concatenate([outputs2_1, outputs2_2], axis=0).tolist()

    initial_outputs1 = copy.deepcopy(outputs1)
    initial_outputs2 = copy.deepcopy(outputs2)

    def input_metrics(x_inputs):
        with torch.no_grad():
            outs1 = manager.model(torch.Tensor(x_inputs).cuda()).to('cpu').tolist()
            outs2 = manager2.model(torch.Tensor(x_inputs).cuda()).to('cpu').tolist()
        dist1 = spatial.distance.cdist(outs1, outs1)
        dist2 = spatial.distance.cdist(outs2, outs2)
        return np.mean(dist1), np.mean(dist2)

    def evaluate_inputs(x_inputs):
        with torch.no_grad():
            outs1 = manager.model(torch.Tensor(x_inputs).cuda()).to('cpu').tolist()
            outs2 = manager2.model(torch.Tensor(x_inputs).cuda()).to('cpu').tolist()
        m1, m2 = input_metrics(x_inputs)
        odist1 = np.mean(spatial.distance.cdist(outs1, initial_outputs1).diagonal())
        odist2 = np.mean(spatial.distance.cdist(outs2, initial_outputs2).diagonal())
        return odist1 * odist2 * m1 * m2

    # input_shape = inputs[0].shape
    # n_inputs = inputs.shape[0]
    # ndims = np.prod(input_shape)
    inputs1 = data1.cpu().numpy().copy()
    inputs2 = data2.cpu().numpy().copy()

    input_shape1 = inputs1[0].shape
    n_inputs1 = inputs1.shape[0]
    ndims1 = np.prod(input_shape1)

    input_shape2 = inputs2[0].shape
    n_inputs2 = inputs2.shape[0]
    ndims2 = np.prod(input_shape2)    
    
    def mutate_inputs(inputs, input_shape, n_inputs, ndims, evaluate_fn, epsilon, max_iterations):
        score = evaluate_fn(inputs)
        for iteration in range(max_iterations):
            torch.cuda.empty_cache()
            mutation_pos = np.random.randint(0, ndims)
            mutation = np.zeros(ndims, dtype=np.float32)
            mutation[mutation_pos] = epsilon
            mutation = np.reshape(mutation, input_shape)
            mutation_batch = np.zeros(inputs.shape, dtype=np.float32)
            mutation_idx = np.random.randint(0, n_inputs)
            mutation_batch[mutation_idx] = mutation

            mutate_right = inputs + mutation_batch
            mutate_left  = inputs - mutation_batch
            score_right = evaluate_fn(mutate_right)
            score_left  = evaluate_fn(mutate_left)
            
            if score_right <= score and score_left <= score:
                continue
            if score_right > score_left:
                inputs = mutate_right
                score = score_right
            else:
                inputs = mutate_left
                score = score_left
        return inputs, score

    def evaluate_fn1(x):
        tensor_x = torch.from_numpy(x).cuda()
        outs = manager.model(tensor_x).detach().to('cpu').numpy()
        return np.mean(spatial.distance.cdist(outs, outs))

    def evaluate_fn2(x):
        tensor_x = torch.from_numpy(x).cuda()
        outs = manager2.model(tensor_x).detach().to('cpu').numpy()
        return np.mean(spatial.distance.cdist(outs, outs))

    mutated1, score1 = mutate_inputs(inputs1, input_shape1, n_inputs1, ndims1, evaluate_fn1, epsilon, max_iterations)
    mutated2, score2 = mutate_inputs(inputs2, input_shape2, n_inputs2, ndims2, evaluate_fn2, epsilon, max_iterations)
    outputs1 = manager.model(torch.from_numpy(mutated1).cuda()).detach().to('cpu').numpy()
    outputs2 = manager2.model(torch.from_numpy(mutated2).cuda()).detach().to('cpu').numpy()
    # print(outputs1.shape, outputs2.shape)
    
    combined_outputs = np.concatenate([outputs1, outputs2], axis=0)
    
    # profiling_inputs = inputs
    # input_metrics_1, input_metrics_2 = input_metrics(profiling_inputs) # 용도불명. 일단 주석처리리

    def compute_ddv_cos(outputs1, outputs2):
        dists = []
        n_pairs = int(len(outputs1) / 2)
        for i in range(n_pairs):
            ya = outputs1[i]
            yb = outputs1[i + n_pairs]
            dist = spatial.distance.cosine(ya, yb)
            dists.append(dist)
        dists2 = []
        for i in range(n_pairs):
            ya = outputs2[i]
            yb = outputs2[i + n_pairs]
            dist = spatial.distance.cosine(ya, yb)
            dists2.append(dist)
        return np.array(dists), np.array(dists2)

    def compute_ddv_euc(outputs1, outputs2):
        dists = []
        n_pairs = int(len(outputs1) / 2)
        for i in range(n_pairs):
            ya = outputs1[i]
            yb = outputs1[i + n_pairs]
            dist = spatial.distance.euclidean(ya, yb)
            dists.append(dist)
        dists2 = []
        for i in range(n_pairs):
            ya = outputs2[i]
            yb = outputs2[i + n_pairs]
            dist = spatial.distance.euclidean(ya, yb)
            dists2.append(dist)
        return np.array(dists), np.array(dists2)

    def compute_sim_cos(ddv1, ddv2):
        return spatial.distance.cosine(ddv1, ddv2)

    ddv1, ddv2 = compute_ddv_cos(outputs1, outputs2)
    ddv_cos_distance = compute_sim_cos(ddv1, ddv2)
    print('DDV cos-cos [%d => %d] %.5f' % (task_id, target_id, ddv_cos_distance))
    ddvcc_list.append(ddv_cos_distance)

    ddv1, ddv2 = compute_ddv_euc(outputs1, outputs2)
    ddv_euc_distance = compute_sim_cos(ddv1, ddv2)
    print('DDV euc-cos [%d => %d] %.5f' % (task_id, target_id, ddv_euc_distance))
    ddvec_list.append(ddv_euc_distance)
    tasks.append(task_id)  

    table_rows.append({
        'target_id': target_id,
        'task_id': task_id,  # selected_backbone
        'ddv_cos': ddv_cos_distance,
        'ddv_euc': ddv_euc_distance
    })

# ddvec_list에서 최대값의 인덱스를 찾고, 그 인덱스를 tasks 리스트에 적용해 실제 task_id를 얻음
best_idx = np.argmax(ddvec_list)
best_task = tasks[best_idx]

print('Selected backbone for target ' + str(target_id) +
      ' = (euc) ' + str(best_task))

result_csv = f"pickaback_{args.dataset_config}_result.csv"
table_csv = f"pickback_{args.dataset_config}_table.csv"

write_header = not os.path.exists(result_csv)
with open(result_csv, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['target_id', 'task_id'])
    if write_header:
        writer.writeheader()
    writer.writerow({'target_id': target_id, 'task_id': best_task})

write_header_table = not os.path.exists(table_csv)
with open(table_csv, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['target_id', 'task_id', 'ddv_cos', 'ddv_euc'])
    if write_header_table:
        writer.writeheader()
    for row in table_rows:
        writer.writerow(row)