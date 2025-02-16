import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LENGTH = 128

class CaptionFolder(Dataset):
    def __init__(self, root):
     
        self.root = root
        self.samples = []
        self.classes = []

        for class_name in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root, cls)
            for fname in sorted(os.listdir(cls_path)):
                if fname.endswith('.txt'):
                    file_path = os.path.join(cls_path, fname)
                    self.samples.append((file_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        tokenized = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )

        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
        return {"input": tokenized, "label": label}


def n24news_train_loader(train_batch_size, num_workers=4, pin_memory=True):
    dataset = CaptionFolder('/data_library/n24news/caption/train')
    return DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def n24news_val_loader(val_batch_size, num_workers=4, pin_memory=True):
    dataset = CaptionFolder('/data_library/n24news/caption/test')
    return DataLoader(
        dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def train_loader(train_batch_size, num_workers=4, pin_memory=True):
    dataset = CaptionFolder('/data_library/n24news/caption/train')
    return DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def val_loader(val_batch_size, num_workers=4, pin_memory=True):
    dataset = CaptionFolder('/data_library/n24news/caption/test')
    return DataLoader(
        dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )