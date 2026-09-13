import numpy as np, torch, torch.nn.functional as F
np.random.seed(0)

# one tiny batch — random data is fine, this checks the math not the model
xb = np.random.randn(8, 784).astype(np.float32)
yb = np.random.randint(0, 10, size=8)

W1 = np.random.randn(784, 128).astype(np.float32) * 0.01; B1 = np.zeros(128, np.float32)
W2 = np.random.randn(128, 10).astype(np.float32) * 0.01;  B2 = np.zeros(10, np.float32)

# --- NumPy: paste your forward + 6 backward lines here -> dW1, dB1, dW2, dB2 ---
L1 = (xb @ W1 + B1) # (B, 128)
# apply ReLU
Z1 = np.maximum(0, L1) # (B, 128)

# the output of the first layer is out1 with shape (1,128) for each input
L2 = (Z1 @ W2 ) + B2 # (B, 10)

# apply softmax
L2s = L2 - L2.max(axis=1, keepdims=True)
Z2 = np.exp(L2s) / np.sum(np.exp(L2s), axis=1, keepdims=True)

L = - np.mean(np.log(Z2[range(xb.shape[0]), yb]))
Y = np.zeros_like(Z2); Y[np.arange(xb.shape[0]), yb] = 1
dL2 = (Z2 - Y) / xb.shape[0] # (B, 10)

dW2 = Z1.T @ dL2 # (B, 10)
dB2 = dL2.sum(axis=0) # (10)

dL1 = dL2 @ W2.T * (0 < L1) # b, 128

dW1 = xb.T @ dL1 # 784, 128
dB1 = dL1.sum(axis=0)

# --- PyTorch: same weights, autograd ---
tW1 = torch.tensor(W1, requires_grad=True); tB1 = torch.tensor(B1, requires_grad=True)
tW2 = torch.tensor(W2, requires_grad=True); tB2 = torch.tensor(B2, requires_grad=True)
tx, ty = torch.tensor(xb), torch.tensor(yb)
logits = torch.relu(tx @ tW1 + tB1) @ tW2 + tB2
F.cross_entropy(logits, ty).backward()

for name, mine, theirs in [("W1", dW1, tW1.grad), ("B1", dB1, tB1.grad),
                           ("W2", dW2, tW2.grad), ("B2", dB2, tB2.grad)]:
    print(name, np.allclose(mine, theirs.numpy(), atol=1e-5),
          np.abs(mine - theirs.numpy()).max())