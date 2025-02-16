import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def train_loader(train_batch_size, num_workers=4, pin_memory=True):
    train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(
        '/data_library/n24news/image/train',
        transform=train_transform
    )

    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def val_loader(val_batch_size, num_workers=4, pin_memory=True):
    val_transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(
        '/data_library/n24news/image/test',
        transform=val_transform
    )

    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )