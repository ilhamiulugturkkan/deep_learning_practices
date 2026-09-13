import torchvision
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#load train data
train_data = torchvision.datasets.MNIST(root='./week2/mnsit/data', train=True, download=True)

xs = train_data.data.to(dtype=torch.float, device=device)
xs = xs / 255.0
mean = xs.mean()
std = xs.std()
xs = (xs - mean) / std
ys = train_data.targets.to(device=device)

train_dataset = TensorDataset(xs, ys)

dataload = DataLoader(train_dataset, batch_size=64, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.seq = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.seq(x)
        return logits

model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
lr = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr)

def train(model, dataloader, epoch,  optimizer, loss_fn):
    for i in range(epoch):
        for x, y in (dataloader):
            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    logits = model(xs)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, ys)
    print(loss.item())

def test(model):
    test_data = torchvision.datasets.MNIST(root='./week2/mnsit/data', train=False, download=True)

    test_xs = test_data.data.to(dtype=torch.float, device=device)
    test_xs = test_xs / 255.0
    test_xs = (test_xs - mean) / std
    test_ys = test_data.targets.to(device=device)


    logits = model(test_xs)
    accuracy = (torch.argmax(logits, dim=1) == test_ys).float().mean() * 100
    print(accuracy.item())

train(model, dataloader=dataload, epoch=10,optimizer=optimizer,loss_fn=loss_fn )
test(model)


