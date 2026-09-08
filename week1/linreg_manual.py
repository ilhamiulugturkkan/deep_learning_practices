import torch
import random

x = torch.rand(200)
y = 3 * x + 2 + random.normalvariate(0, 15)


# MSE  = sum((y - (wx + b))^2) = sum((y - wx - b)^2)
# dMSE/dw = -2w(y - wx - b)
# dMSE/db = -2(y - wx - b)


# Randomly initialize initial w and b
w = random.uniform(-10,10)
b = random.uniform(-10,10)

# Training loop
for i in range(10000):
    y_pred = x * w + b
    loss = (y - y_pred).mean()
    w -= 0.001 * (-2*w*(y-w*x-b)).mean()
    b -= 0.001 * (-2*(y-w*x-b)).mean()
    if i % 1000 == 0:
        print(w.data,b.data)
        print(loss.data)

print(w, b)