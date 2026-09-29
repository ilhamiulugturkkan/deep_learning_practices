import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

torch.manual_seed(7)

def buildStoi():
    data = open('/Users/ilhamiulugturkkan/Desktop/deep_learning_practices/src/data/corpus.txt', 'r').read()
    all_chars = sorted(list(set(data)))
    itoc = {}
    stoi = {}
    for i, c in enumerate(all_chars):
        itoc[i] = c
        stoi[c] = i
    return stoi, itoc

stoi, itoc = buildStoi()

class TextSet(Dataset):
    def __init__(self, path, seq_length):
        data = open(path, 'r').read()
        all_chars = sorted(list(set(data)))
        self.vocab_size = len(all_chars)
        self.itoc = {}
        self.stoi = {}
        for i, c in enumerate(all_chars):
            self.itoc[i] = c
            self.stoi[c] = i

        real_data = torch.tensor([self.stoi[c] for c in data], dtype=torch.long)

        self.data = real_data
        self.seq_length = seq_length

    def __len__(self):
        return (len(self.data) - seq_len - 1) // seq_len

    def __getitem__(self, index):
        s = index * seq_len
        return self.data[s:s + self.seq_length], self.data[s + 1:s + self.seq_length + 1]

class RNNCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.randn(hidden_size,) * 0.01)

    def forward(self, X, prev_out=None):
        if prev_out is None:
            prev_out = torch.zeros(X.shape[0], self.W_hh.shape[1])
        x_out = (torch.matmul(X, self.W_xh))
        h_out = (torch.matmul(prev_out, self.W_hh) + self.b_h)
        out = torch.tanh(x_out + h_out)
        return out

class RNNScratch(nn.Module):

    def __init__(self, vocab_len, hidden_size, embedding_dim=16):
        super().__init__()
        self.emb_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_len, embedding_dim=embedding_dim)
        self.rnn = RNNCell(embedding_dim, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, vocab_len)


    def forward(self, x): # x_shape: (batch_size, seq_length). after embeddings should be (batch_size, seq_length, emb_dim)
        x_emb = self.embeddings(x)
        outputs = []
        prev = None
        for i in range(x_emb.shape[1]):
            prev = self.rnn(x_emb[:, i, :], prev)
            y_preds = self.out(prev)
            outputs.append(y_preds)
        return torch.stack(outputs, dim=1)

    def sample(self, seed, sample_size ,temp=1):
        self.eval()
        seed_idx = torch.tensor([stoi[c] for c in seed], dtype=torch.long)
        seed_idx = seed_idx.reshape(1, seed_idx.shape[0])
        with torch.no_grad():
            for i in range(sample_size):
                preds = self.forward(seed_idx[:, -64:])
                logits = torch.softmax(preds[0,-1,:] / temp, dim=0)
                taken = torch.multinomial(logits, 1, False).reshape(1,-1)
                seed_idx = torch.concat([seed_idx, taken], dim=1)
                seed = seed + itoc[taken.item()]

        self.train()
        return seed
                
total_epoch = 5
seq_len = 64
datas = TextSet('/Users/ilhamiulugturkkan/Desktop/deep_learning_practices/src/data/corpus.txt', seq_length=seq_len)
dl = DataLoader(dataset=datas,batch_size=64, shuffle=True)
model = RNNScratch(datas.vocab_size, 1024, 256)
optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.9)

for e in range(total_epoch):
    model.train()
    losses = []
    for x, y in dl:
        logits = model(x)                                        # (B, T, V)
        loss = F.cross_entropy(logits.reshape(-1, datas.vocab_size), y.reshape(-1))   # (B*T, V) vs (B*T,)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        with torch.no_grad():
            optimizer.step()

    if e % 20 == 0:
        avg_loss = sum(losses) / len(losses)
        print(avg_loss)

# For sampling I need to feed a (natch_size, seq_length) sequence. 
sample = model.sample('The ', 500, 1)
print(sample)