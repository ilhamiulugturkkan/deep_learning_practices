import torch.nn as nn


class MLP(nn.Module):
        def __init__(self, **args):
            super().__init__()
            self.flatten = nn.Flatten()
            self.seq = nn.Sequential(
                nn.Linear(args['fan_in'], 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128,args['fan_out']),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.seq(x)
            return logits

MODELS = {"mlp": MLP}