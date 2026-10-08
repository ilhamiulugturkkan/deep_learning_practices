import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 8
batch_size = 256
max_iter = 2000
eval_iter = 200
n_embed = 32
head_size = 16


corpus = open('/Users/ilhamiulugturkkan/Desktop/deep_learning_practices/src/data/_chat.txt', 'r').read()
chars = sorted(list(set(corpus)))
print(f'Total number of unique chars used:{len(chars)}')
print(f'Total number of chars in corpus {len(corpus)}')

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda a: ''.join([itos[i] for i in a])
print(encode('sude'))
print(decode([69, 71, 54, 55]))


enc_corpus = torch.tensor(encode(corpus), dtype=torch.long)
print(f'Now I\'ve tokenized the corpus. Tokenized corpus has this shape {enc_corpus.shape} and type {enc_corpus.dtype}')

train_len = int(enc_corpus.shape[0] * 0.9)

train = enc_corpus[:train_len]
val = enc_corpus[train_len:]

def getBatch(split):
    data = train if split == 'train' else val
    idx = torch.randint(low=0, high=len(data)-block_size, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx ])
    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    losses = {}
    model.eval()
    for split in ['train', 'val']:
        zeros = torch.zeros(eval_iter)
        for k in range(eval_iter):
            xb, yb = getBatch(split)
            logits, loss = model(xb, yb)
            zeros[k] = loss.item()
        losses[split] = zeros.mean()
    model.train()
    return losses

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

    def forward(self, X):
        B, T, C = X.shape
        q = self.query(X)
        key = self.key(X)

        wei = q @ key.transpose(-2,-1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(X)
        out = wei @ v

        return out

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
    def forward(self, X):
        out = torch.cat([h(X) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.seq = nn.Sequential(
        nn.Linear(n_emb, n_emb * 4),
        nn.ReLU()
        )
        self.proj = nn.Linear(n_emb * 4, n_emb)

    def forward(self, x):
        out = self.seq(x)
        out = self.proj(out)
        return out

class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.multihead = MultiHead(n_head, head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.multihead(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Simple(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embed)
        self.position_emb = nn.Embedding(block_size, n_embed)
        #self.head = Head(n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            nn.LayerNorm(n_embed)
        )
        self.lin = nn.Linear(n_embed, vocab_size)

    def forward(self, X, Y=None): # X and Y (B, T)
        B, T = X.shape
        tok_embs = self.emb(X) # Logits is (B, T, C)
        pos_embs = self.position_emb(torch.arange(T, device=device))
        embs = tok_embs + pos_embs
        outs = self.blocks(embs)
        logits = self.lin(outs)
        if Y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            Y = Y.view(B*T)
            loss = nn.functional.cross_entropy(logits, Y)

        return logits, loss

    def sample(self, X, sample_length):

        for i in range(sample_length):
            idxs = X[:, -block_size:]
            logits, loss = self.forward(idxs)
            what_we_care = logits[:,-1, :]
            softed = nn.functional.softmax(what_we_care, dim=-1)
            pick = torch.multinomial(softed, 1)
            X = torch.concat([X, pick], dim=1)
        return X
            


model = Simple(len(chars))
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), 0.001)


for i in range(max_iter):

    if i % eval_iter == 0:
        losses = estimate_loss(model)
        print(f'Iter {i}: Train loss is {losses["train"].item()} and val loss is {losses["val"].item()}')

    x, y = getBatch('train')

    logits, loss = model(x, y) # Logits is (B, T, C)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



print(loss.item())

idx = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.sample(idx, 500)[0].tolist()))