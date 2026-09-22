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
        p.requires_grad_(False)

    print('Parameters are frozen')

    model.fc = nn.Sequential(
        nn.Linear(2048, 10),
    )

    model = model.to(device=device)

    print('Head is replaced')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        preprocess
    ])
    train_data = torchvision.datasets.CIFAR10(root='./src/data', train=True, download=True, transform=transform)

    train_size = int(len(train_data) * 0.8)
    val_size = int(len(train_data)) - train_size

    train, val = random_split(train_data, [train_size, val_size])


    train_dataload = DataLoader(train, batch_size=256, shuffle=True,  num_workers=2)
    val_dataload = DataLoader(val, batch_size=256, shuffle=True,  num_workers=2)

    print('CIFAT data is loaded to dataloaders')

    EPOCH = 10

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay=5e-4)
    t_max = EPOCH
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    start_time = time.perf_counter()

    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(EPOCH):
        print(f'Epoch {epoch}')
        train_losses = []
        model.train()
        for images, labels in train_dataload:
            images = images.to(device=device)
            labels = labels.to(device=device)
            logits = model(images)

            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()

            train_losses.append(loss.item())
            optimizer.step()

        train_loss = sum(train_losses) / len(train_losses)
        print(f'Train loss in epoch {epoch} is {train_loss}')

        scheduler.step()

        val_losses = []
        accuracies = []
        model.eval()
        for images, labels in val_dataload:
            with torch.no_grad():
                images = images.to(device=device)
                labels = labels.to(device=device)

                logits = model(images)
                loss = loss_fn(logits, labels)
                val_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1)
                accuracies.append(torch.sum(preds==labels).item() / labels.shape[0] * 100.0)

        val_loss = sum(val_losses) / len(val_losses)
        accuracy = sum(accuracies) / len(accuracies)
        print(f"Epoch {epoch} val_loss {val_loss}, accuracy: {accuracy}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_resnet.pt")   # checkpoint
        else:
            patience_counter += 1
            if patience_counter >= 30:
                print(f"Early stopping at epoch {epoch}")
                break

    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

if __name__ == '__main__':
    torch.manual_seed(7)
    main()