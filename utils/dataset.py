import os
import yaml
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .textfolder import TextFolder


config_path = os.path.join(os.path.dirname(__file__), 'dataset_config.yaml')
with open(config_path, 'r') as f:
    dataset_config = yaml.safe_load(f)


def get_transforms(cfg, dataset_name=None, is_train=True):
    """
    cfg: dataset_config.yaml에서 읽어온 'cifar100' 혹은 'n24news' 섹션 (dict)
    dataset_name: subfolder 이름 (예: 'fish', 'flowers' 등), 없는 경우 None
    is_train: True면 train transform, False면 val transform
    """
    transform_list = []

    if is_train:
        if 'crop' in cfg:
            crop_size = tuple(cfg['crop'].get('size', [32, 32]))
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
        mean = None
        std = None

        if isinstance(cfg['mean'], dict):
            if dataset_name and dataset_name in cfg['mean']:
                mean = cfg['mean'][dataset_name]
                std = cfg['std'][dataset_name]
            else:
                if 'cifar100' in cfg['mean']:
                    mean = cfg['mean']['cifar100']
                    std = cfg['std']['cifar100']
                else:
                    
                    raise ValueError
        else:
            mean = cfg['mean']
            std = cfg['std']

        transform_list.append(transforms.Normalize(mean=mean, std=std))
        
    return transforms.Compose(transform_list)

def train_loader(config_name, batch_size, dataset_name=None, num_workers=4, pin_memory=True, tokenizer=None):
    """
    config_name: 'cifar100', 'n24news', 등
    dataset_name: 서브폴더 이름 (예: 'group7_text' - 텍스트 데이터의 경우 '_text' 접미사 포함) 또는 None
    tokenizer: 텍스트 데이터 전처리에 사용할 토크나이저 함수 (예: lambda text: bert_tokenizer(text)['input_ids'])
    """
    cfg = dataset_config[config_name]
    
    # cifar100은 기존의 ImageFolder 사용
    if config_name == 'cifar100':
        if dataset_name is not None and cfg.get("subfolder", False):
            train_path = os.path.join(cfg['train_path'], dataset_name)
        else:
            train_path = cfg['train_path']
        train_transform = get_transforms(cfg, dataset_name=dataset_name, is_train=True)
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    else:
        # n24news와 같이 image와 text가 분리된 경우
        if dataset_name is not None and dataset_name.endswith("_text"):
            # 텍스트 데이터인 경우, dataset_name에서 "_text" 접미사를 제거하고 text 경로 사용
            subfolder_name = dataset_name[:-5]
            train_path = os.path.join(cfg['text_train_path'], subfolder_name)
            # tokenizer가 제공되면 transform을 tokenizer 함수로 정의
            if tokenizer is not None:
                transform = lambda x: tokenizer(x)['input_ids']
            else:
                transform = None
            train_dataset = TextFolder(train_path, transform=transform)
        else:
            # 기본은 이미지 데이터
            if dataset_name is not None and cfg.get("subfolder", False):
                train_path = os.path.join(cfg['image_train_path'], dataset_name)
            else:
                train_path = cfg['image_train_path']
            train_transform = get_transforms(cfg, dataset_name=dataset_name, is_train=True)
            train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def val_loader(config_name, batch_size, dataset_name=None, num_workers=4, pin_memory=True, tokenizer=None):
    """
    config_name: 'cifar100', 'n24news', 등
    dataset_name: 서브폴더 이름 (예: 'group7_text' - 텍스트 데이터의 경우 '_text' 접미사 포함) 또는 None
    tokenizer: 텍스트 데이터 전처리에 사용할 토크나이저 함수
    """
    cfg = dataset_config[config_name]
    
    if config_name == 'cifar100':
        if dataset_name is not None and cfg.get("subfolder", False):
            test_path = os.path.join(cfg['test_path'], dataset_name)
        else:
            test_path = cfg['test_path']
        val_transform = get_transforms(cfg, dataset_name=dataset_name, is_train=False)
        val_dataset = datasets.ImageFolder(test_path, transform=val_transform)
    else:
        if dataset_name is not None and dataset_name.endswith("_text"):
            subfolder_name = dataset_name[:-5]
            test_path = os.path.join(cfg['text_test_path'], subfolder_name)
            if tokenizer is not None:
                transform = lambda x: tokenizer(x)['input_ids']
            else:
                transform = None
            val_dataset = TextFolder(test_path, transform=transform)
        else:
            if dataset_name is not None and cfg.get("subfolder", False):
                test_path = os.path.join(cfg['image_test_path'], dataset_name)
            else:
                test_path = cfg['image_test_path']
            val_transform = get_transforms(cfg, dataset_name=dataset_name, is_train=False)
            val_dataset = datasets.ImageFolder(test_path, transform=val_transform)
    
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )