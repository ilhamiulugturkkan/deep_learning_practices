import torch.nn as nn


class MLP(nn.Module):
        def __init__(self, **args):
            super().__init__()
            self.flatten = nn.Flatten()
            self.seq = nn.Sequential(
                nn.Linear(args['fan_in'], 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(128,args['fan_out']),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.seq(x)
            return logits

class CONV(nn.Module):
        def __init__(self, **args): # input size: torch.Size([64, 1, 28, 28])
            super().__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1), # (W-F+2P)/S + 1 = W so the keep the size the same WHILE s=1 P = (F - 1) / 2 out:torch.Size([64, 32, 32, 32])
                #nn.BatchNorm2d(32),
                nn.ReLU(), # out: torch.Size([64, 32, 32, 32])
                nn.MaxPool2d(2,2), # out: torch.Size([64, 32, 16, 16])

                nn.Conv2d(32, 64, 3, 1, 1), # out: torch.Size([64, 64, 16, 16])
                #nn.BatchNorm2d(64),
                nn.ReLU(), # out: torch.Size([64, 64, 16, 16])
                nn.MaxPool2d(2, 2), # out: torch.Size([64, 64, 8, 8])

                nn.Conv2d(64, 128, 3, 1, 1), # out: torch.Size([64, 128, 8, 8])
                #nn.BatchNorm2d(128),
                nn.ReLU(), # out: torch.Size([64, 128, 8, 8])
                nn.MaxPool2d(2,2), # out: torch.Size([64, 128, 4, 4])

                nn.Flatten(),
                nn.Linear(2048, 10)
            )

        def forward(self, x):
            logits = self.seq(x)
            return logits

MODELS = {"mlp": MLP, "cnn": CONV}