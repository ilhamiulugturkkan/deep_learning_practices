import torchvision # just to load dataset
import numpy as np


#load train data
train_data = torchvision.datasets.MNIST(root='./week2/mnsit/data', train=True, download=True)

xs = train_data.data.numpy()
ys = train_data.targets.numpy()

# first we have to flatten the data
xs = xs.reshape(60000, 784)

B = xs.shape[0]

# normalize
xs = xs / 255.0
# standardize
mean = np.mean(xs)
std = np.std(xs)
xs = (xs - mean) / std


# init weights 
W1 = np.random.randn(784, 128) * 0.01
B1 = np.random.randn(128) * 0.01

W2 = np.random.randn(128, 10) * 0.01
B2 = np.random.randn(10) * 0.01

lr = 0.1

batch = 64
for epoch in range(10):
    perm = np.random.permutation(B)
    for s in range(0, B, batch):
        idx = perm[s:s+batch]
        xb, yb = xs[idx], ys[idx]

        
        L1 = (xb @ W1 + B1) # (B, 128)
        # apply ReLU
        Z1 = np.maximum(0, L1) # (B, 128)

        # the output of the first layer is out1 with shape (1,128) for each input
        L2 = (Z1 @ W2 ) + B2 # (B, 10)

        # apply softmax
        L2s = L2 - L2.max(axis=1, keepdims=True)
        Z2 = np.exp(L2s) / np.sum(np.exp(L2s), axis=1, keepdims=True)

        L = - np.mean(np.log(Z2[range(xb.shape[0]), yb]))

        print(L)

        Y = np.zeros_like(Z2); Y[np.arange(xb.shape[0]), yb] = 1
        dL2 = (Z2 - Y) / xb.shape[0] # (B, 10)

        dW2 = Z1.T @ dL2 # (B, 10)
        dB2 = dL2.sum(axis=0) # (10)

        dL1 = dL2 @ W2.T * (0 < L1) # b, 128

        dW1 = xb.T @ dL1 # 784, 128
        dB1 = dL1.sum(axis=0)

        W2 -= lr * dW2
        B2 -= lr * dB2
        W1 -= lr * dW1
        B1 -= lr * dB1



# load test data
test_data = torchvision.datasets.MNIST(root='./week2/mnsit/data', train=False, download=True)

test_xs = test_data.data.numpy()
test_xs = test_xs.reshape(test_xs.shape[0], 784)
# normalize
test_xs = test_xs / 255.0
# standardize
test_xs = (test_xs - mean) / std
test_ys = test_data.targets.numpy()

L1 = (test_xs @ W1 + B1) # (B, 128)
# apply ReLU
Z1 = np.maximum(0, L1) # (B, 128)

# the output of the first layer is out1 with shape (1,128) for each input
L2 = (Z1 @ W2 ) + B2 # (B, 10)

# apply softmax
L2s = L2 - L2.max(axis=1, keepdims=True)
Z2 = np.exp(L2s) / np.sum(np.exp(L2s), axis=1, keepdims=True)

acc = (Z2.argmax(axis=1) == test_ys).mean() * 100

print(acc)