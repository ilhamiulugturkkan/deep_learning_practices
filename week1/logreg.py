from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torch

x, y = make_moons(n_samples=500, noise=0.2)

print(f'{x.shape}')
print(f'{y.shape}')

x_tensor = torch.tensor(x)
y_tensor = torch.tensor(y)

w = torch.rand(2, requires_grad=True)
b = torch.rand(1, requires_grad=True)

x_tensor = x_tensor.float()
w = w.float()
b = b.float()


lr = 0.001
for i in range(5000):
    y_pred = torch.sigmoid((x_tensor @ w + b)) #FORWARD PASS. I had some problems with the matmul especially because of dtypes of elements

    cross_entropy = sum(-(y_tensor * torch.log(y_pred)+ (1-y_tensor) * torch.log(1-y_pred))) # LOSS FUNCTION (couldnt remember it so checked google)


    cross_entropy.backward() # BACKPROPOGATION


    with torch.no_grad(): # UPDATE PARAMETERS
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()
    if i % 1000 == 0:
        print(cross_entropy.data/y_tensor.size(0))




# This, I borrowed from Claude. Everything else is handwritten
plt.scatter(x_tensor[:, 0], x_tensor[:, 1], c=y_tensor.ravel(), cmap="viridis")
plt.xlabel("x1"); plt.ylabel("x2"); plt.colorbar(label="y")
plt.show()