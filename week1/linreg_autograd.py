import torch
import random
import matplotlib.pyplot as plt

x = torch.rand(20000)
error = torch.rand(20000)
y = 3 * x + 2 + error


w = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

print(w,b)

losses = []
for i in range(100001):
    y_pred = x * w + b
    loss = ((y - y_pred)**2).mean()
    losses.append(loss)
    loss.backward()
    with torch.no_grad():
        w.data -= 0.001 * w.grad
        b.data -= 0.001 * b.grad
    w.grad.zero_()
    b.grad.zero_()
    if i % 1000 == 0:
        print(loss.data)
print(w,b)




