import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
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
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return self.data[index:index + self.seq_length], self.data[index + 1:index + self.seq_length + 1]

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
                

seq_len = 8
datas = TextSet('/Users/ilhamiulugturkkan/Desktop/deep_learning_practices/corpus.txt', seq_length=seq_len)
dl = DataLoader(dataset=datas,batch_size=64, shuffle=False)
rnn_scratch = RNNScratch(datas.vocab_size, 64, 16)
for x, y in dl:
    y_preds = rnn_scratch(x)
    print(y_preds.shape) # list with len 8 and and element shape are (64, vocab_len)
    print(y.shape)
    break