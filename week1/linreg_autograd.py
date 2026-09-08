import torch
import random
import matplotlib.pyplot as plt

torch.manual_seed(7)

x = torch.rand(20000)
error = torch.rand(20000)
y = 3 * x + 2 + error

w0 = torch.rand(1); b0 = torch.rand(1)
w = w0.clone().requires_grad_(True)
b = b0.clone().requires_grad_(True)


for i in range(50000):
    y_pred = x * w + b
    loss = ((y - y_pred)**2).mean()
    loss.backward()
    with torch.no_grad():
        w.data -= 0.001 * w.grad
        b.data -= 0.001 * b.grad
    w.grad.zero_()
    b.grad.zero_()

print(w,b)

x_man, y_man = x, y
w_man = w0.item()
b_man = b0.item()

# Training loop
for i in range(50000):
    y_pred_man = x_man * w_man + b_man
    loss_man = ((y_man - y_pred_man)**2).mean()
    w_temp = w_man
    w_man -= 0.001 * (-2*x_man*(y_man-w_man*x_man-b_man)).mean()
    b_man -= 0.001 * (-2*(y_man-w_temp*x_man-b_man)).mean()

print(w_man, b_man)

print(f'Diff for w is:{w.item()-w_man} and for b is:{b.item()-b_man}')