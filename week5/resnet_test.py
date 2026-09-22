import torch
import time
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
from torch.utils.data import random_split, DataLoader
import torch.nn as nn

def main():
    device = (torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
    print(device)
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()  # returns the exact transform pipeline used in training

    model = resnet50(weights=weights)
    print('Model is loaded')

    # Freeze the backbone
    for p in model.parameters():
        p.requires_grad_(True)

    print('Parameters are not frozen')

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 10)

    )

    model = model.to(device=device)

    model.load_state_dict(torch.load("./best_model_resnet.pt", weights_only=True))

    print('Head is replaced')

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        preprocess
    ])

    test_data = torchvision.datasets.CIFAR10(root='./src/data', train=False, download=True, transform=test_transform)

    test_dataload = DataLoader(test_data, batch_size=64, shuffle=True,  num_workers=2)

    print('CIFAT data is loaded to dataloaders')

    EPOCH = 10

    loss_fn = nn.CrossEntropyLoss()

    start_time = time.perf_counter()


    test_losses = []
    accuracies = []
    model.eval()
    for images, labels in test_dataload:
        with torch.no_grad():
            images = images.to(device=device)
            labels = labels.to(device=device)

            logits = model(images)
            loss = loss_fn(logits, labels)
            test_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            accuracies.append(torch.sum(preds==labels).item() / labels.shape[0] * 100.0)

    test_loss = sum(test_losses) / len(test_losses)
    accuracy = sum(accuracies) / len(accuracies)
    print(f'Test loss is {test_loss} and accuracy is {accuracy}')

    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

if __name__ == '__main__':
    torch.manual_seed(7)
    main()