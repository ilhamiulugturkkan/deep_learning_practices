import torch
import random
import matplotlib.pyplot as plt

torch.manual_seed(7)

x = torch.rand(20000)
error = torch.rand(20000)
y = 3 * x + 2 + error


w = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)


for i in range(100001):
    y_pred = x * w + b
    loss = ((y - y_pred)**2).mean()
    loss.backward()
    with torch.no_grad():
        w.data -= 0.001 * w.grad
        b.data -= 0.001 * b.grad
    w.grad.zero_()
    b.grad.zero_()

print(w,b)

# COMPARE WITH MANUAL
x_man = torch.rand(20000)
error_man = torch.rand(20000)
y_man = 3 * x_man + 2 + error_man
w_man = random.uniform(-10,10)
b_man = random.uniform(-10,10)

# Training loop
for i in range(10001):
    y_pred_man = x_man * w_man + b_man
    loss_man = ((y_man - y_pred_man)**2).mean()
    w_man -= 0.001 * (-2*x_man*(y_man-w_man*x_man-b_man)).mean()
    b_man -= 0.001 * (-2*(y_man-w_man*x_man-b_man)).mean()

print(w_man, b_man)

print(f'Diff for w is:{w-w_man} and for b is:{b-b_man}')