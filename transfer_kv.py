#!/usr/bin/env python
import torch
import argparse
import csv
import os

# perceiver 모델과 관련 함수들을 import합니다.
# (모델 파일 구조에 따라 import 경로는 조정 필요)
from models import perceiver

def get_kv_modules(perceiver_model, modality):
    """각 레이어에서 modality에 맞는 key-value 모듈을 추출합니다."""
    kv_list = []
    for layer in perceiver_model.layers:
        cross_attn = layer[0]  # PreNorm으로 래핑된 MultiModalAttention
        if modality == 'image':
            kv_list.append(cross_attn.fn.to_kv_image)
        elif modality == 'text':
            kv_list.append(cross_attn.fn.to_kv_text)
        else:
            raise ValueError("modality must be 'image' or 'text'")
    return kv_list

def set_kv_modules(perceiver_model, new_kv_list, modality):
    """각 레이어의 modality에 맞는 key-value 모듈을 새 모듈로 설정합니다."""
    for layer, new_kv in zip(perceiver_model.layers, new_kv_list):
        cross_attn = layer[0]
        if modality == 'image':
            cross_attn.fn.to_kv_image = new_kv
        elif modality == 'text':
            cross_attn.fn.to_kv_text = new_kv
        else:
            raise ValueError("modality must be 'image' or 'text'")

def get_modality(group_id):
    """
    oxford 데이터셋에서 그룹 1~6은 image modality, 7~12는 text modality로 간주합니다.
    group_id는 문자열 형태의 정수여야 합니다.
    """
    try:
        gid = int(group_id)
    except Exception as e:
        raise ValueError("group id must be an integer") from e
    return 'image' if gid <= 6 else 'text'

def load_perceiver_checkpoint(ckpt_path, modality):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    print("Checkpoint layer names:")
    
    model = perceiver(
        num_freq_bands=6,
        depth=5,
        max_freq=10,
        dataset_history=['oxford'],
        dataset2num_classes={'oxford': 12},
        init_weights=False,
        image_input_channels=3,
        image_input_axis=2,
        text_input_channels=768,
        text_input_axis=1,
        num_latents=256,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        fourier_encode_data=True,
        self_per_cross_attn=1,
        final_classifier_head=False,
        modality=modality
    )
    curr_model_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in curr_model_state_dict:
            if curr_model_state_dict[name].size() == param.size():
                curr_model_state_dict[name].copy_(param)
            else:
                dims = len(curr_model_state_dict[name].size())
                if dims == 4:
                    curr_model_state_dict[name][:param.size(0), :param.size(1), ...].copy_(param)
                elif dims == 2:
                    curr_model_state_dict[name][:param.size(0), :param.size(1)].copy_(param)
                elif dims == 1:
                    curr_model_state_dict[name][:param.size(0)].copy_(param)
                else:
                    # 필요한 경우 추가 차원에 대해 처리
                    curr_model_state_dict[name].copy_(param)
    else:
        print(f"[WARNING] {name} is not found in the current model state dict.")

    model.to(device)
    return model

def save_perceiver_checkpoint(model, original_ckpt, save_path):
    """
    기존 checkpoint 메타데이터를 그대로 복사하고, 수정된 모델 state_dict를 덮어쓴 후 저장합니다.
    """
    new_ckpt = original_ckpt.copy()
    new_ckpt['state_dict'] = model.state_dict()
    torch.save(new_ckpt, save_path)
    print(f"Saved new checkpoint to {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description="oxford 데이터셋의 pickaback 결과를 읽어 task 모델에 target 모델의 kv 모듈을 transfer한 후 checkpoint로 저장합니다."
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default='pickaback_oxford_result.csv',
        help="target_id, task_id가 기록된 CSV 파일 경로 (헤더는 첫 행)"
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/home/Minju/MM-Pick-a-back/checkpoints_perceiver/CPG_single_scratch_woexp/perceiver/oxford',
        help="체크포인트가 저장된 기본 디렉토리"
    )
    parser.add_argument(
        '--checkpoint_name',
        type=str,
        default='checkpoint-20.pth.tar',
        help="불러올 체크포인트 파일명"
    )
    args = parser.parse_args()

    with open(args.csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # 헤더 스킵
        for row in reader:
            if len(row) < 2:
                continue
            csv_target_id, csv_task_id = row[0].strip(), row[1].strip()
            # 따옴표나 공백 제거
            target_id = csv_target_id.strip('"').strip()
            task_id = csv_task_id.strip('"').strip()
            print(f"\n[INFO] Processing transfer: task_id={task_id}, target_id={target_id}")
            
            # 각 모델의 modality 결정 (oxford: group 1~6=image, 7~12=text)
            task_modality = get_modality(task_id)
            target_modality = get_modality(target_id)

            # 체크포인트 경로 구성
            task_ckpt_path = os.path.join(args.base_dir, f'group{task_id}', 'gradual_prune', args.checkpoint_name)
            target_ckpt_path = os.path.join(args.base_dir, f'group{target_id}', 'gradual_prune', args.checkpoint_name)
            
            if not os.path.isfile(task_ckpt_path):
                print(f"[WARN] Task checkpoint not found: {task_ckpt_path}")
                continue
            if not os.path.isfile(target_ckpt_path):
                print(f"[WARN] Target checkpoint not found: {target_ckpt_path}")
                continue

            print("[INFO] Loading task model...")
            task_model = load_perceiver_checkpoint(task_ckpt_path, modality=task_modality)
            print("[INFO] Loading target model...")
            target_model = load_perceiver_checkpoint(target_ckpt_path, modality=target_modality)

            print(f"[INFO] Transferring kv modules from target ({target_modality}) to task ({task_modality}) model...")
            target_kv_modules = get_kv_modules(target_model, target_modality)
            set_kv_modules(task_model, target_kv_modules, target_modality)
            task_model.set_modality(target_modality)

            # 수정된 task 모델을 새로운 checkpoint로 저장
            new_ckpt_path = os.path.join(args.base_dir, f'group{task_id}', 'gradual_prune', f'checkpoint-20_kv_transfer_{target_id}.pth.tar')
            original_ckpt = torch.load(task_ckpt_path, map_location='cpu')
            save_perceiver_checkpoint(task_model, original_ckpt, new_ckpt_path)

if __name__ == '__main__':
    main()
