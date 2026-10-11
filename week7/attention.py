import torch
from torch.nn import functional as F

torch.manual_seed(7)
B, T, C = 4, 8, 32   # batch, sequence length, embedding size
head_size = 16

x = torch.randn(B, T, C)

# weights for the projections: (C, head_size), so x @ W gives (B, T, head_size)
W_q = torch.randn(C, head_size)
W_k = torch.randn(C, head_size)
W_v = torch.randn(C, head_size)

def attention(x, W_q, W_k, W_v, causal=True): # W's shape are (emb_size, head_size)
    B, T, C = x.shape

    q = x @ W_q # T, C @ C, head size -> T, head_size
    k = x @ W_k
    wei = q @ k.transpose(-2, -1) / q.shape[-1]**0.5 # T, T
    if causal:
        bum = torch.ones(T, T)
        wei = wei.masked_fill(torch.tril(bum) == 0, float('-inf'))

    wei = F.softmax(wei, -1)

    v = x @ W_v # T, head_size

    out = wei @ v # T, head_size
    
    return out


out = attention(x, W_q, W_k, W_v, True)

print(out.shape)

x2 = x.clone()
x2[:, -1, :] = torch.randn(B, C)   

out2 = attention(x2, W_q, W_k, W_v, True)

assert out.shape == (B, T, head_size)

# 1. mask works: positions 0..T-2 can't see the changed position T-1
assert torch.allclose(out[:, :-1], out2[:, :-1])

# 2. control: the last position should change, or test 1 proves nothing
assert not torch.allclose(out[:, -1], out2[:, -1])

# 3. causal=False should leak backwards
out3 = attention(x,  W_q, W_k, W_v, causal=False)
out4 = attention(x2, W_q, W_k, W_v, causal=False)
assert not torch.allclose(out3[:, :-1], out4[:, :-1])

print("all attention tests passed")