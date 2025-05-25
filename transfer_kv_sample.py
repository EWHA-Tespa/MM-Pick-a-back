#!/usr/bin/env python
import torch
import argparse
import csv
import os

from models import perceiver_io


def get_kv_modules(perceiver_model, modality):
    kv_list = []
    for modulelist in perceiver_model.layers:
        for submodule in modulelist:
            if hasattr(submodule, 'fn') and hasattr(submodule.fn, 'to_kv_image'):
                if modality == 'image':
                    kv_list.append(submodule.fn.to_kv_image)
                elif modality == 'text':
                    kv_list.append(submodule.fn.to_kv_text)
                else:
                    raise ValueError("modality must be 'image' or 'text'")
    return kv_list


def set_kv_modules(perceiver_model, new_kv_list, modality):
    idx = 0
    for modulelist in perceiver_model.layers:
        for submodule in modulelist:
            if hasattr(submodule, 'fn') and hasattr(submodule.fn, 'to_kv_image'):
                if modality == 'image':
                    submodule.fn.to_kv_image = new_kv_list[idx]
                elif modality == 'text':
                    submodule.fn.to_kv_text = new_kv_list[idx]
                else:
                    raise ValueError("modality must be 'image' or 'text'")
                idx += 1


def get_modality(group_id):
    gid = int(group_id)
    return 'image' if gid <= 28 else 'text'


def load_perceiver_checkpoint(ckpt_path, modality, group_id=None):
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt_path, map_location=device)
    # print("[DEBUG] checkpoint keys:", checkpoint.keys())
    # print("[DEBUG] shared_layer_info keys:", checkpoint['shared_layer_info'].keys())
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    model = perceiver_io(
        num_freq_bands=6,
        depth=4,
        max_freq=10,
        image_input_channels=3,
        image_input_axis=2,
        text_input_channels=768,
        text_input_axis=1,
        max_text_length=512,
        queries_dim=512,
        dataset_history=['cub'],
        dataset2num_classes={'cub': 56},
        num_latents=256,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        fourier_encode_data=True,
        decoder_ff=False,
        final_classifier_head=True,
        attn_dropout=0.1,
        ff_dropout=0.1
    )

    model.to(device)

    # First restore piggymask before state_dict load
    if 'shared_layer_info' in checkpoint:
        group_name = f'group{group_id}'
        piggymask_dict = checkpoint['shared_layer_info'].get(group_name, {}).get('piggymask', {})

        for name, module in model.named_modules():
            if name in piggymask_dict:
                setattr(module, 'piggymask', torch.nn.Parameter(piggymask_dict[name]))
                print(f"[INFO] Registered piggymask to: {name}")

        if hasattr(model, 'classifiers'):
            for idx, classifier in enumerate(model.classifiers):
                name = f'classifiers.{idx}'
                if name in piggymask_dict:
                    setattr(classifier, 'piggymask', torch.nn.Parameter(piggymask_dict[name]))
                    print(f"[INFO] Registered piggymask to classifier: {name}")

    # Now restore state_dict
    curr_model_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if 'piggymask' in name:
            continue
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
                    curr_model_state_dict[name].copy_(param)
        else:
            print(f"[WARNING] {name} is not found in the current model state dict.")

    return model


def save_perceiver_checkpoint(model, original_ckpt, save_path):
    new_ckpt = original_ckpt.copy()
    new_ckpt['state_dict'] = model.state_dict()
    torch.save(new_ckpt, save_path)
    print(f"Saved new checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='pickaback_cub_result.csv')
    parser.add_argument('--base_dir', type=str, default='/home/aix22404/MM-Pick-a-back/checkpoints_perceiver_io/CPG_single_scratch_woexp/perceiver_io/cub')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint-20.pth.tar')
    args = parser.parse_args()

    with open(args.csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            target_id, task_id = row[0].strip('"'), row[1].strip('"')

            print(f"\n[INFO] Processing transfer: task_id={task_id}, target_id={target_id}")
            task_modality = get_modality(task_id)
            target_modality = get_modality(target_id)

            task_ckpt_path = os.path.join(args.base_dir, f'group{task_id}', 'gradual_prune', args.checkpoint_name)
            target_ckpt_path = os.path.join(args.base_dir, f'group{target_id}', 'gradual_prune', args.checkpoint_name)

            if not os.path.isfile(task_ckpt_path):
                print(f"[WARN] Task checkpoint not found: {task_ckpt_path}")
                continue
            if not os.path.isfile(target_ckpt_path):
                print(f"[WARN] Target checkpoint not found: {target_ckpt_path}")
                continue

            print("[INFO] Loading task model...")
            task_model = load_perceiver_checkpoint(task_ckpt_path, task_modality, task_id)
            print("[INFO] Loading target model...")
            target_model = load_perceiver_checkpoint(target_ckpt_path, target_modality, target_id)

            print(f"[INFO] Transferring kv modules from target ({target_modality}) to task ({task_modality}) model...")
            target_kv_modules = get_kv_modules(target_model, target_modality)
            set_kv_modules(task_model, target_kv_modules, target_modality)
            task_model.set_modality(target_modality)

            new_ckpt_path = os.path.join(args.base_dir, f'group{task_id}', 'gradual_prune', f'checkpoint-20_kv_transfer_{target_id}.pth.tar')
            original_ckpt = torch.load(task_ckpt_path, map_location='cpu')
            save_perceiver_checkpoint(task_model, original_ckpt, new_ckpt_path)


if __name__ == '__main__':
    main()