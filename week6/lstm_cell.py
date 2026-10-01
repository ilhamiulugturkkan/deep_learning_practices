import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

torch.manual_seed(7)
device = torch.accelerator.current_accelerator()
print(device)

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

        real_data = torch.tensor([self.stoi[c] for c in data], dtype=torch.long, device=device)

        self.data = real_data
        self.seq_length = seq_length

    def __len__(self):
        return (len(self.data) - seq_len - 1) // seq_len

    def __getitem__(self, index):
        s = index * seq_len
        return self.data[s:s + self.seq_length], self.data[s + 1:s + self.seq_length + 1]

class LSTMCell(nn.Module):

    def __init__(self, input_size, out_size):
        super().__init__()
        self.W_fg = nn.Parameter(torch.randn(input_size+out_size, out_size) * 0.01)
        self.B_fg = nn.Parameter(torch.ones(out_size, ))
        self.W_sigu = nn.Parameter(torch.randn(input_size+out_size, out_size) * 0.01)
        self.B_sigu = nn.Parameter(torch.randn(out_size,) * 0.01)
        self.W_tanu = nn.Parameter(torch.randn(input_size+out_size, out_size) * 0.01)
        self.B_tanu = nn.Parameter(torch.randn(out_size,) * 0.01)
        self.W_sigo = nn.Parameter(torch.randn(input_size+out_size, out_size) * 0.01)
        self.B_sigo = nn.Parameter(torch.randn(out_size,) * 0.01)

    def forward(self, X, C=None, h=None):
        if C is None:
            C = torch.zeros(X.shape[0], self.W_fg.shape[1], device=device)
        if h is None:
            h = torch.zeros(X.shape[0], self.W_fg.shape[1], device=device)

        concatted = torch.concat((X, h), dim=1) # concat old result and input. size: (BATCH, SEQ, IN+OUT)

        x_sigmo_fg = torch.matmul(concatted, self.W_fg) + self.B_fg # size (B, SEQ, OUT)
        sigmo = torch.sigmoid(x_sigmo_fg) # we'll use this a lot. size (B, SEQ, OUT

        forget_gate = C * sigmo # Forget some of the old cell state # size (B, SEQ, OUT) WAS DOT

        x_sigmo_ug = torch.matmul(concatted, self.W_sigu) + self.B_sigu
        x_tanh_ug = torch.matmul(concatted, self.W_tanu) + self.B_tanu
        sigmo_ug = torch.sigmoid(x_sigmo_ug)
        tanh_ug = torch.tanh(x_tanh_ug)
        update_gate = sigmo_ug * tanh_ug # WAS DOT

        new_C = forget_gate + update_gate # update the cell state

        x_sigout = torch.matmul(concatted, self.W_sigo) + self.B_sigo
        sigmout = torch.sigmoid(x_sigout)

        new_H = sigmout * torch.tanh(new_C)

        # print(f'Shape of C is {new_C.shape}, and new h is {new_H.shape}')
        return new_C, new_H

class LSTMScratch(nn.Module):

    def __init__(self, vocab_len, hidden_size, embedding_dim=16):
        super().__init__()
        self.emb_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_len, embedding_dim=embedding_dim, device=device)
        self.lstm = LSTMCell(embedding_dim, hidden_size).to(device=device)
        self.out_layer = nn.Linear(hidden_size, vocab_len, device=device)


    def forward(self, x): # x_shape: (batch_size, seq_length). after embeddings should be (batch_size, seq_length, emb_dim)
        x_emb = self.embeddings(x)
        outputs = []
        prev_C = None
        prev_H = None
        for i in range(x_emb.shape[1]):
            prev_C, prev_H = self.lstm(x_emb[:, i, :], prev_C, prev_H)
            logits = self.out_layer(prev_H)
            # print(f'Shape of prev_C is {prev_C.shape}')
            # print(f'Shape of prev_h is {logits.shape}')
            outputs.append(logits)
        return torch.stack(outputs, dim=1)

    def sample(self, seed, sample_size ,temp=1):
        self.eval()
        seed_idx = torch.tensor([stoi[c] for c in seed], dtype=torch.long, device=device)
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
model = LSTMScratch(datas.vocab_size, 256, 256)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for e in range(total_epoch):
    model.train()
    losses = []
    for x, y in dl:
        logits = model(x) # (B, T, V)

        loss = F.cross_entropy(logits.reshape(-1, datas.vocab_size), y.reshape(-1))   # (B*T, V) vs (B*T,)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        with torch.no_grad():
            optimizer.step()

    if e % 1 == 0:
        avg_loss = sum(losses) / len(losses)
        print(avg_loss)

# For sampling I need to feed a (natch_size, seq_length) sequence. 
sample = model.sample('The ', 500, 1)
print(sample)