from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torch

x, y = make_moons(n_samples=500, noise=0.2)

print(f'{x.shape}')
print(f'{y.shape}')

x_tensor = torch.tensor(x)
y_tensor = torch.tensor(y)

w = torch.rand(2, requires_grad=True, dtype=x_tensor.dtype)
b = torch.rand(1, requires_grad=True, dtype=x_tensor.dtype)


lr = 0.001
for i in range(5000):
    y_pred = torch.sigmoid((x_tensor @ w + b)) #FORWARD PASS. I had some problems with the matmul especially because of dtypes of elements

    cross_entropy = (-(y_tensor * torch.log(y_pred)+ (1-y_tensor) * torch.log(1-y_pred))).sum() # LOSS FUNCTION (couldnt remember it so checked google)


    cross_entropy.backward() # BACKPROPOGATION


    with torch.no_grad(): # UPDATE PARAMETERS
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()
    if i % 1000 == 0:
        print(cross_entropy.data/y_tensor.size(0))




# This, I borrowed from Claude. Everything else is handwritten
with torch.no_grad():
    w0, w1, b0 = w[0].item(), w[1].item(), b.item()

xs = torch.tensor([x_tensor[:, 0].min(), x_tensor[:, 0].max()])   # left and right edge of the data
ys = -(w0 * xs + b0) / w1

plt.scatter(x_tensor[:, 0], x_tensor[:, 1], c=y_tensor, s=10)
plt.plot(xs, ys, "k--")
plt.ylim(x_tensor[:, 1].min() - 0.5, x_tensor[:, 1].max() + 0.5)   # so a steep line doesn't blow up the axes
plt.savefig("week1/logreg_boundary.png")