import kagglehub
import pathlib
import pandas as pd
import torchvision
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import wandb

DOG_SPECIES = 2


class PetDataset(Dataset):
    def __init__(self, df, images_path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_path = pathlib.Path(images_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.images_path / f"{row['image_id']}.jpg").convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row["label"]


def _load_labels(label_path):
    df = pd.read_csv(
        label_path,
        sep=r"\s+",
        header=None,
        names=["image_id", "class_id", "species", "breed_id"],
    )
    return df[df["species"] == DOG_SPECIES].reset_index(drop=True)


def _remap_breed_ids(*dfs):
    breed_ids = sorted(pd.concat([df["breed_id"] for df in dfs]).unique())
    mapping = {breed_id: idx for idx, breed_id in enumerate(breed_ids)}
    for df in dfs:
        df["label"] = df["breed_id"].map(mapping)
    return mapping


def prepare_data(train_transform=None, eval_transform=None, val_size=0.15, random_state=42):
    path = kagglehub.dataset_download(
        "devdgohil/the-oxfordiiit-pet-dataset",
        output_dir="/Users/ilhamiulugturkkan/Desktop/deep_learning_practices/src/data/dog_dataset",
    )
    print("Path to dataset files:", path)

    label_path = f"{path}/annotations/annotations/"
    trainval_labels = _load_labels(f"{label_path}trainval.txt")
    test_labels = _load_labels(f"{label_path}test.txt")

    _remap_breed_ids(trainval_labels, test_labels)
    print('TEST LABELS:',set(test_labels))

    train_labels, val_labels = train_test_split(
        trainval_labels,
        test_size=val_size,
        stratify=trainval_labels["breed_id"],
        random_state=random_state,
    )
    print('TEST LABELS:',set(train_labels))
    images_path = f"{path}/images/images"

    train_dataset = PetDataset(train_labels, images_path, train_transform)
    val_dataset = PetDataset(val_labels, images_path, eval_transform)
    test_dataset = PetDataset(test_labels, images_path, eval_transform)

    return train_dataset, val_dataset, test_dataset


def main():
    device = (torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        preprocess,
    ])
    eval_transform = preprocess

    train_dataset, val_dataset, test_dataset = prepare_data(
        train_transform=train_transform, eval_transform=eval_transform
    )

    train_dataload = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_dataload = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_dataload = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    model = resnet50(weights=weights)
    print('Model is loaded')

    print('Parameters are not frozen')

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 25)
    )

    model = model.to(device=device)

    wandb.watch(model, log_freq=300, log="all")

    TOTAL_EPOCHS = 15
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith('fc.')]
    head_params = model.fc.parameters()

    head_lr = 0.1
    backbone_lr = 0.1 * head_lr 

    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr},
    ], momentum=0.9, weight_decay=5e-4)

    loss_fn = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=TOTAL_EPOCHS)

    best_val_loss = float("inf")

    
    for epoch in range(TOTAL_EPOCHS):
        print(f'Epoch {epoch}')
        train_losses = []
        model.train()
        for x, y in train_dataload:
            x = x.to(device=device)
            y = y.to(device=device)

            # forward pass
            logits = model(x)

            # calc loss
            loss = loss_fn(logits, y)
            train_losses.append(loss.item())

            # reset grads
            optimizer.zero_grad()
            # backprop
            loss.backward()
            # update
            with torch.no_grad():
                optimizer.step()

            wandb.log({"train/loss": loss.item()})

        train_loss = sum(train_losses) / len(train_losses)
        print(train_loss)
        scheduler.step()

        # val
        val_losses = []
        val_accuracies = []
        model.eval()
        for x, y in val_dataload:
            x = x.to(device=device)
            y = y.to(device=device)

            logits = model(x)

            loss = loss_fn(logits, y)

            val_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds==y).item() / y.shape[0] * 100.0
            val_accuracies.append(acc)
            wandb.log({"val/loss": loss.item()})
            wandb.log({"val/accuraacy": acc})

        val_accuracy = sum(val_accuracies) / len(val_accuracies)
        val_loss = sum(val_losses) / len(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_dog_breed.pt")
        print(f'Epoch {epoch}, val accuracy is {val_accuracy} and val loss is {val_loss}')
    wandb.finish()
    # test it!!
    test_losses = []
    test_accuracies = []
    model.eval()
    for x, y in test_dataload:
        x = x.to(device=device)
        y = y.to(device=device)

        logits = model(x)

        loss = loss_fn(logits, y)

        test_losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        test_accuracies.append(torch.sum(preds==y).item() / y.shape[0] * 100.0)

        cm = confusion_matrix(y_true=y, y_pred=preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    test_accuracy = sum(test_accuracies) / len(test_accuracies)
    test_loss = sum(test_losses) / len(test_losses)
    print(f'Test results, test accuracy is {test_accuracy} and test loss is {test_loss}')        

    return train_dataload, val_dataload, test_dataload


if __name__ == "__main__":
    wandb.login()
    wandb.init(project="Dog_dataset", name="ResNet")
    main()
