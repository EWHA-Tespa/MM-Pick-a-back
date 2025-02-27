import os
import random
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import BertTokenizer


config_path = os.path.join(os.path.dirname(__file__), 'dataset_config.yaml')
with open(config_path, 'r') as f:
    dataset_config = yaml.safe_load(f)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_caption(caption, max_length=40):
    """
    캡션 텍스트를 토크나이즈하여 (input_ids, attention_mask) 튜플로 반환합니다.
    """
    encoding = tokenizer(
        caption,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)


class PairedImageCaptionDataset(Dataset):
    """
    이미지와 캡션 파일명이 동일한 경우를 가정합니다.
    image_dir와 caption_dir는 클래스별 폴더를 가진 디렉토리이며,
    각 클래스 폴더 내에서 동일한 파일명(확장자 제외)의 이미지(.jpg, .png 등)와
    텍스트(.txt)를 쌍으로 구성합니다.
    """
    def __init__(self, image_dir, caption_dir, image_transform=None, max_text_len=40):
        self.image_dir = image_dir
        self.caption_dir = caption_dir
        self.image_transform = image_transform
        self.max_text_len = max_text_len
        
        self.samples = []  # 각 샘플은 (image_path, caption_path, label) 튜플
        classes = sorted(os.listdir(image_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        for cls in classes:
            cls_img_dir = os.path.join(image_dir, cls)
            cls_cap_dir = os.path.join(caption_dir, cls)
            if not os.path.isdir(cls_img_dir) or not os.path.isdir(cls_cap_dir):
                continue
            # 이미지 파일 목록 (확장자 필터링)
            image_files = [f for f in os.listdir(cls_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in image_files:
                img_path = os.path.join(cls_img_dir, img_file)
                base_name = os.path.splitext(img_file)[0]
                cap_file = base_name + '.txt'
                cap_path = os.path.join(cls_cap_dir, cap_file)
                if os.path.exists(cap_path):
                    self.samples.append((img_path, cap_path, self.class_to_idx[cls]))
                else:
                    print(f"Warning: 캡션 파일이 존재하지 않습니다: {cap_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, cap_path, label = self.samples[idx]
        # 이미지 로드 및 전처리
        image = Image.open(img_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        # 캡션 로드
        with open(cap_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        # 캡션 토큰화 (input_ids, attention_mask)
        caption_tokens = tokenize_caption(caption, max_length=self.max_text_len)
        return image, caption_tokens, label


def get_transforms(cfg, is_train=True):
    """
    cfg: YAML 설정(예, resize, crop, mean, std 등)
    is_train: 학습용이면 랜덤 crop, flip 등을 적용합니다.
    """
    transform_list = []
    if is_train:
        if 'crop' in cfg:
            crop_size = tuple(cfg['crop'].get('size', [224, 224]))
            padding = cfg['crop'].get('padding', 4)
            transform_list.append(transforms.RandomCrop(crop_size, padding=padding))
        elif 'resize' in cfg:
            resize_size = tuple(cfg['resize'])
            transform_list.append(transforms.Resize(resize_size))
        if cfg.get('random_horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip())
    else:
        if 'resize' in cfg:
            resize_size = tuple(cfg['resize'])
            transform_list.append(transforms.Resize(resize_size))
    transform_list.append(transforms.ToTensor())
    if 'mean' in cfg and 'std' in cfg:
        # mean, std가 dict 형태일 경우 default 키 사용
        if isinstance(cfg['mean'], dict):
            mean = cfg['mean'].get('default', [0.485, 0.456, 0.406])
            std = cfg['std'].get('default', [0.229, 0.224, 0.225])
        else:
            mean = cfg['mean']
            std = cfg['std']
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)


def vilt_collate_fn(batch):
    """
    배치 샘플: (image, (input_ids, attention_mask), label) 튜플 리스트
    반환 딕셔너리:
      - "image": (B, 3, H, W)
      - "text_ids": (B, L_text)
      - "text_masks": (B, L_text)
      - "itm_labels": (B,)
    """
    images, caption_tokens, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    text_ids = torch.stack([tokens[0] for tokens in caption_tokens], dim=0)
    text_masks = torch.stack([tokens[1] for tokens in caption_tokens], dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return {
        "image": images,
        "text_ids": text_ids,
        "text_masks": text_masks,
        "itm_labels": labels
    }


def train_loader(config_name, batch_size, num_workers=4, pin_memory=True, max_text_len=40):
    """
    train_loader: dataset_config.yaml에서 config_name에 해당하는 설정을 읽어
                  이미지와 캡션 데이터를 로드합니다.
    """
    cfg = dataset_config[config_name]
    train_image_dir = cfg['train_image_path']
    train_caption_dir = cfg['train_caption_path']
    image_transform = get_transforms(cfg, is_train=True)
    
    dataset = PairedImageCaptionDataset(train_image_dir, train_caption_dir, image_transform=image_transform, max_text_len=max_text_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=vilt_collate_fn
    )

def val_loader(config_name, batch_size, num_workers=4, pin_memory=True, max_text_len=40):
    """
    val_loader: 평가용 데이터셋 로드 (test_image_path, test_caption_path 사용)
    """
    cfg = dataset_config[config_name]
    test_image_dir = cfg['test_image_path']
    test_caption_dir = cfg['test_caption_path']
    image_transform = get_transforms(cfg, is_train=False)
    
    dataset = PairedImageCaptionDataset(test_image_dir, test_caption_dir, image_transform=image_transform, max_text_len=max_text_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=vilt_collate_fn
    )