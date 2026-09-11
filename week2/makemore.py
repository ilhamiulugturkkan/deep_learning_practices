import torch
import torch.nn.functional as T

words = open('week2/names.txt').read().splitlines()

x = sorted(set(''.join(words)))

x = ['.',] + x

stoi = {}
itos = {}
for i, s in enumerate(x):
    stoi[s] = i
    itos[i] = s

N = torch.zeros(len(stoi), len(stoi), dtype=torch.int32)

for w in words:
    w = '.' + w + '.'
    for c1, c2 in zip(w, w[1:]):
        N[stoi[c1],stoi[c2]] += 1

P = (N+1).float() # to eliminate 0s (which leads to inf while calculating the likelihood etc.)
P /= P.sum(dim=1, keepdim=True) # SUM PART IS IMPORTANT. WE USED BROADCASTING, BUT WITH KEEP DIM. 
# BECAUSE KEEP_DIM IS FALSE BY DEFAULT AND WOULD RETURN [27] WHICH WOULD BROADCASTED LIKE [1, 27]
# GOTTA KEEP BROADCASTING RULES IN HEAD EVERYTIME, IT CAN LEAD TO DISASTERS

# Let's sample some
g = torch.Generator().manual_seed(7)
word = ''

while True:
    lastchar = ''
    if (word == ''):
        lastchar = '.'
    else:
        lastchar = word[len(word)-1]
    index = stoi[lastchar]
    nextChar = torch.multinomial(P[index],1,replacement=True,generator=g).item()
    nextChar = itos[nextChar]
    if nextChar == '.':
        break
    word = word + nextChar

print(word)


likelihood = 0.0
loglikelihood = 0
n = 0
for w in words:
    w = '.' + w + '.'
    for c1, c2 in zip(w, w[1:]):
        likelihood *= P[stoi[c1],stoi[c2]]
        loglikelihood += torch.log(P[stoi[c1],stoi[c2]])
        n += 1

print(f'{likelihood=}')
print(f'{loglikelihood=}')
print(f'{-loglikelihood.item()/n}')

# MINIMIZE -loglikelihood.item()/n!!!
g = torch.Generator().manual_seed(7)
# Now let's do a neural network type of approach
xs, ys = [], []
for w in words[:]:
    w = '.' + w + '.'
    for c1, c2 in zip(w, w[1:]):
        xs.append(stoi[c1])
        ys.append(stoi[c2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(xs)
print(ys)

ones = T.one_hot(xs, num_classes=27).float() # INPUTS (FIXED)
ys_enc = T.one_hot(ys, num_classes=27).float() # INPUTS (FIXED)

W = torch.randn((27,27), generator=g, requires_grad=True) # WEIGHTS (MUTABLE)

# Input will be like 1, 27 and w.shape is 27,27 so the output.shape is 1,27
# if we feed the nn with all the input data we can have like 1mil, 27 @ 27,27 -> (1mil, 27) kinda thin
lr = 50
for i in range(200):
    logits = ones @ W
    counts = logits.exp()
    prob = counts / counts.sum(dim=1, keepdim=True)
    loss = -prob[torch.arange(prob.shape[0]), ys].log().mean()+ 0.01*(W**2).mean()

    W.grad = None

    loss.backward()

    with torch.no_grad():
        W.data -= lr * W.grad

g = torch.Generator().manual_seed(7)
def genName():
    f = T.one_hot(torch.tensor([0]), 27).float()
    word = ''
    while True:
        logits = f @ W
        counts = logits.exp()
        prob = counts / counts.sum(dim=1, keepdim=True)
        ix = torch.multinomial(prob, 1, replacement=True, generator=g).item()
        char = itos[ix]
        if (char == '.'):
            break
        word += char
        f = T.one_hot(torch.tensor([stoi[char]]), 27).float()
    print(word)

for i in range(5):
    genName()



    



    

    
