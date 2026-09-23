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
import torch.nn.functional as F

DOG_SPECIES = 2
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def unnormalize(img_tensor):
    """Reverses ImageNet normalization for display. img_tensor: (3, H, W)."""
    img = img_tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()

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
    test_dataload = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = resnet50(weights=weights)
    print('Model is loaded')

    print('Parameters are not frozen')

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 25)
    )

    model.load_state_dict(torch.load("./best_model_dog_breed.pt", weights_only=True))

    model = model.to(device=device)

    print('Lets test this')

    # test it!!
    test_losses = []
    test_accuracies = []
    all_preds = []
    all_ys = []
    worst = []
    model.eval()
    for x, y in test_dataload:
        x = x.to(device=device)
        y = y.to(device=device)

        logits = model(x)

        losses =  F.cross_entropy(logits, y, reduction='none')
        loss  =  F.cross_entropy(logits, y)

        test_losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)

        # Now I'll get the worst predicted 

        all_preds = all_preds + preds.tolist()
        all_ys = all_ys + y.flatten().tolist()

        test_accuracies.append(torch.sum(preds==y).item() / y.shape[0] * 100.0)


        for i in range(x.size(0)):
            if preds[i] != y[i]:  # only keep mistakes
                worst.append((
                    losses[i].item(),
                    x[i].cpu(),
                    y[i].item(),
                    preds[i].item(),
                ))



    print(len(all_preds), all_preds)
    print(len(all_ys), all_ys)
    
    test_accuracy = sum(test_accuracies) / len(test_accuracies)
    test_loss = sum(test_losses) / len(test_losses)
    print(f'Test results, test accuracy is {test_accuracy} and test loss is {test_loss}')    
    labels = list(range(25))
    cm = confusion_matrix(y_true=all_ys, y_pred=all_preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation="vertical")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    worst.sort(key=lambda x: x[0], reverse=True)
    top10 = worst[:10]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, (loss, img, true_idx, pred_idx) in zip(axes.flat, top10):
        ax.imshow(unnormalize(img))
        ax.set_title(
            f"True: {true_idx}\nPred: {pred_idx}",
            fontsize=9,
        )
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('worst10_misclassified.png', dpi=150)
    plt.show()

    return train_dataload, val_dataload, test_dataload


if __name__ == "__main__":
    main()
