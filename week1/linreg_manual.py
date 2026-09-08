import torch
import random

x = torch.rand(20000)
error = torch.rand_like(x)
y = 3 * x + 2 + error

w = random.uniform(-10,10)
b = random.uniform(-10,10)

# Training loop
for i in range(10001):
    y_pred = x * w + b
    loss = ((y - y_pred)**2).mean()
    w -= 0.001 * (-2*x*(y-w*x-b)).mean()
    b -= 0.001 * (-2*(y-w*x-b)).mean()
    if i % 1000 == 0:
        print(loss.data)

print(w, b)