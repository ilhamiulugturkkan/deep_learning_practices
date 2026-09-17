import argparse
import torch
from torch import nn
import yaml
import torch.nn.functional as T
import matplotlib.pyplot as plt
import numpy as np
import wandb
import time
from src.models import MODELS
from src.data import DATASETS

torch.manual_seed(0)
w0 = torch.randn(5, 5)
grad = torch.randn(5, 5)

#adam gibi adam
p3 = w0.clone().requires_grad_(True)
p3.grad = grad.clone()
v = torch.zeros_like(p3)
s = torch.zeros_like(p3)
v = 0.9 * v + 0.1 * p3.grad
s = 0.999 * s + 0.001 * p3.grad**2
v_hat = v / (1 - 0.9**1)
s_hat = s / (1 - 0.999**1) #t is 1 on step 1

with torch.no_grad():
    p3 -= 0.01 * v_hat / (torch.sqrt(s_hat) + 1e-8)

# yours
p1 = w0.clone().requires_grad_(True)
p1.grad = grad.clone()
v = torch.zeros_like(p1)
v = 0.9 * v + p1.grad
with torch.no_grad():
    p1 -= 0.01 * v

# torch's
p2 = w0.clone().requires_grad_(True)
p2.grad = grad.clone()
opt = torch.optim.SGD([p2], lr=0.01, momentum=0.9)
opt.step()

print((p1 - p2).abs().max())   # should be ~0, e.g. 1e-7
print((p1 - p3).abs().max())   # should be ~0, e.g. 1e-7