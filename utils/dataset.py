import os
import yaml
import torch
import clip

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer, DistilBertTokenizer, DistilBertModel

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

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

class CLIPImageDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        # ImageFolder를 사용하고, CLIP 전처리(preprocess)를 transform으로 지정합니다.
        self.dataset = datasets.ImageFolder(root=root_dir, transform=clip_preprocess)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # image: 이미 CLIP 전처리(preprocess) 적용된 tensor ([3, H, W])
        with torch.no_grad():
            image_embedding = clip_model.encode_image(image.unsqueeze(0).to(device))
        image_embedding = image_embedding.cpu().float()  # 최종 shape: [512]
        return image_embedding, label


class CLIPTextDataset(Dataset):
    def __init__(self, root_dir, max_length=77, is_train=True):
        """
        - root_dir: 텍스트 파일들이 클래스별 서브폴더에 저장되어 있음.
        - CLIP의 tokenize는 기본적으로 max_length=77을 사용합니다.
        """
        self.samples = []  # (filepath, label) 튜플 리스트
        self.max_length = max_length
        # 각 서브폴더를 클래스(label)로 간주 (알파벳 순서 기준)
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.endswith('.txt'):
                        self.samples.append((os.path.join(class_dir, fname), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        # CLIP의 tokenize를 사용하여 텍스트 토큰 생성 (반환 tensor shape: [1, token_length])
        tokens = clip.tokenize([text], truncate=True).squeeze(0)
        # CLIP 텍스트 인코더에 입력하기 위해 device로 이동시키고, 배치 차원 추가
        with torch.no_grad():
            text_embedding = clip_model.encode_text(tokens.unsqueeze(0).to(device))
        text_embedding = text_embedding.cpu().float()  # [1, 512]
        return text_embedding, label
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
        train_path = os.path.join(cfg.get('text_train_path', cfg['text_train_path']), dataset_name)
        max_length = cfg.get('text_max_length', 128)
        train_dataset = CLIPTextDataset(train_path, max_length=max_length, is_train=True)
    else:
        if dataset_name is not None and cfg.get("subfolder", False):
            train_path = os.path.join(cfg['train_path'], dataset_name)
        else:
            train_path = cfg['train_path']
        
        train_dataset = CLIPImageDataset(train_path, is_train=True)
    
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
        test_path = os.path.join(cfg.get('text_test_path', cfg['text_test_path']), dataset_name)
        max_length = cfg.get('text_max_length', 128)
        val_dataset = CLIPTextDataset(test_path, max_length=max_length, is_train=False)
    else:
        if dataset_name is not None and cfg.get("subfolder", False):
            test_path = os.path.join(cfg['test_path'], dataset_name)
        else:
            test_path = cfg['test_path']
        val_dataset = CLIPImageDataset(test_path, is_train=False)
    
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
