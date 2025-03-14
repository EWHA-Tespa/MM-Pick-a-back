import os
from torch.utils.data import Dataset

class TextFolder(Dataset):
    """
    폴더 내의 텍스트 파일을 읽어오는 데이터셋.
    폴더 구조는 ImageFolder와 유사하게, 각 클래스별로 하위 폴더에 파일들이 저장됨.
    """
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []  # (파일 경로, label)
        # 클래스는 폴더 이름으로 결정 (정렬하여 고정 순서 확보)
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        for cls in classes:
            cls_folder = os.path.join(root, cls)
            for fname in os.listdir(cls_folder):
                path = os.path.join(cls_folder, fname)
                if os.path.isfile(path):
                    self.samples.append((path, self.class_to_idx[cls]))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        # 텍스트 파일 읽기
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        if self.transform is not None:
            text = self.transform(text)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return text, target
