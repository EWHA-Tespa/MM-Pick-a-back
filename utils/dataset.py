import os
import yaml
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


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


# utils/dataset.py
import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, root_dir, tokenizer_name='bert-base-uncased', max_length=128, is_train=True):
        self.root_dir = root_dir
        self.samples = []  # (filepath, label) 튜플 리스트
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # 각 서브폴더를 클래스(label)로 간주
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.endswith('.txt'):  # 텍스트 파일만 로드한다고 가정
                        self.samples.append((os.path.join(class_dir, fname), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        # BERT 토크나이저로 토큰화
        encoding = self.tokenizer(text,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_length,
                                  return_tensors='pt')
        # squeeze()를 통해 batch dimension 제거
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding, label

def train_loader(config_name, batch_size, dataset_name=None, num_workers=4, pin_memory=True):
    """
    config_name: 'cifar100' or 'n24news' 등
    dataset_name: subfolder명 (ex: 'fish', 'flowers'), 없으면 None
    """
    cfg = dataset_config[config_name]
    
    # modality 판단: 만약 현재 dataset_name이 text modality에 해당하면?
    is_text = False
    if dataset_name is not None and 'modality' in cfg:
        if dataset_name in cfg['modality'].get('text_groups', []):
            is_text = True

    if is_text:
        train_path = os.path.join(cfg.get('text_train_path', cfg['train_path']), dataset_name)
        max_length = cfg.get('text_max_length', 128)
        train_dataset = TextDataset(train_path, max_length=max_length, is_train=True)
    else:
        if dataset_name is not None and cfg.get("subfolder", False):
            train_path = os.path.join(cfg['train_path'], dataset_name)
        else:
            train_path = cfg['train_path']
        train_transform = get_transforms(cfg, dataset_name=dataset_name, is_train=True)
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def val_loader(config_name, batch_size, dataset_name=None, num_workers=4, pin_memory=True):
    cfg = dataset_config[config_name]
    
    is_text = False
    if dataset_name is not None and 'modality' in cfg:
        if dataset_name in cfg['modality'].get('text_groups', []):
            is_text = True

    if is_text:
        test_path = os.path.join(cfg.get('text_test_path', cfg['test_path']), dataset_name)
        max_length = cfg.get('text_max_length', 128)
        val_dataset = TextDataset(test_path, max_length=max_length, is_train=False)
    else:
        if dataset_name is not None and cfg.get("subfolder", False):
            test_path = os.path.join(cfg['test_path'], dataset_name)
        else:
            test_path = cfg['test_path']
        val_transform = get_transforms(cfg, dataset_name=dataset_name, is_train=False)
        val_dataset = datasets.ImageFolder(test_path, transform=val_transform)
    
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
