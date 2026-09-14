import torch
import torch.nn.functional as F

words = open('week2/names.txt').read().splitlines()

x = sorted(set(''.join(words)))

x = ['.',] + x

stoi = {}
itos = {}
for i, s in enumerate(x):
    stoi[s] = i
    itos[i] = s

g = torch.Generator().manual_seed(7)
block_size = 3


X, y = [], []
for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        y.append(ix)
        context = context[1:] + [ix,]

X = torch.tensor(X)
y = torch.tensor(y)


# let's create the table that we'll index nums and get the embeddings of the chars (we got 27 chars)
C = torch.randn((27, 2), generator=g, requires_grad=True)


W1 = torch.randn((block_size * 2, 100), requires_grad=True, generator=g)
b1 = torch.randn((100), requires_grad=True, generator=g)

W2 = torch.randn((100, 27), requires_grad=True, generator=g)
b2 = torch.randn((27), requires_grad=True, generator=g)

parameters = [C, W1, b1, W2, b2] # optimizer will save us from for loops very soon!

lr = 0.1

for i in range(1000):

    idx = torch.randint(0, X.shape[0], (64,)) # get batch elements' indexes

    emb = C[X[idx]] # (64, 3, 2)

    h1 = torch.tanh((emb.view(-1, block_size * 2) @ W1 + b1)) # -1: pytorch calculates that dim based on the other inputs. h1.shape=(64, 100)
    logits = h1 @ W2 + b2 # (64, 27)

    for p in parameters:
        p.grad = None

    loss = F.cross_entropy(logits, y[idx])

    loss.backward()

    with torch.no_grad():
        for p in parameters:
            p -= lr * p.grad

emb = C[X]
h1 = torch.tanh((emb.view(-1, block_size*2) @ W1 + b1))
logits = h1 @ W2 + b2

loss = F.cross_entropy(logits, y)
print(loss)






