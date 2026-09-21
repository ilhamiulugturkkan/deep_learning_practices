import torchvision
from torch.utils.data import random_split, DataLoader

def get_mnist(batch_size, shuffle):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = torchvision.datasets.MNIST(root='./week2/mnsit/data', train=True, download=True, transform=transform)
    train_size = int(len(train_data) * 0.8)
    val_size = int(len(train_data)) - train_size

    train, val = random_split(train_data, [train_size, val_size])

    train_dataload = DataLoader(train, batch_size=batch_size, shuffle=shuffle,  num_workers=4)
    val_dataload = DataLoader(val, batch_size=batch_size, shuffle=shuffle,  num_workers=4)

    return train_dataload, val_dataload

def get_cifar(batch_size=64, shuffle=True):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    train_data = torchvision.datasets.CIFAR10(root='./src/data', train=True, download=True, transform=transform)

    train_size = int(len(train_data) * 0.8)
    val_size = int(len(train_data)) - train_size

    train, val = random_split(train_data, [train_size, val_size])

    train_dataload = DataLoader(train, batch_size=batch_size, shuffle=shuffle,  num_workers=4)
    val_dataload = DataLoader(val, batch_size=batch_size, shuffle=shuffle,  num_workers=4)

    return train_dataload, val_dataload



DATASETS = {"mnist": get_mnist, "cifar10": get_cifar}