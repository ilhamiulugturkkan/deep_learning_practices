import torchvision
from torch.utils.data import random_split, DataLoader

def get_mnist(batch_size):
    train_data = torchvision.datasets.MNIST(root='./week2/mnsit/data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_size = int(len(train_data) * 0.8)
    val_size = int(len(train_data)) - train_size

    train, val = random_split(train_data, [train_size, val_size])

    train_dataload = DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_dataload = DataLoader(val, batch_size=batch_size, shuffle=True,  num_workers=4)

    return train_dataload, val_dataload




DATASETS = {"mnist": get_mnist}