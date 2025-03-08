# autoregressive text generation using embedding table (or something)

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
# import tiktoken # Not used, smallest embedding table is too large for potato laptop compute

#################
# hyperparameters
batch_size = 32
context_window = 128 # this is what karpathy calls block size in his code
max_iters = 6000 
elval_intervals = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

#################

torch.manual_seed(42)

os.chdir("../")
def get_sample(filepath, size=1024 *1024*64):
  with open(filepath, 'r') as f:
    sample = f.read(size)
  return sample

# wikihow corpus
txt = get_sample('data/wikihow.txt')

# character-wise tokenization
char_set = sorted(set(txt))
vocab_size = len(char_set)
# encoder and decoder mappings and functions
encode_mapping = {t: i for i, t in enumerate(sorted(set(txt)))}
decode_mapping = {v: k for k, v in encode_mapping.items()}
encode = lambda s: [encode_mapping[t] for t in s]
decode = lambda tok: ''.join([decode_mapping[t] for t in tok])

tokens = encode(txt)
data = torch.tensor(data=tokens,dtype=torch.long)

train_test_split_index = int(0.9*(len(data)))
train = data[:train_test_split_index]
test = data[train_test_split_index:]

# data loading
def get_batch(split):
  """Return a batch of x and y tensors of shape (batch_size, context_window)"""
  data = train if split == "train" else test
  ix = torch.randint(high=(len(data) - context_window), size=(batch_size,)) # get random indices for context sequences
  x = torch.stack([data[i:i+context_window] for i in ix])
  y = torch.stack([data[i+1:i+context_window+1] for i in ix])
  x, y = x.to(device), y.to(device) # move data to device
  return x, y

@torch.no_grad()
def estimate_loss():
   out = {}
   model.eval()
   for split in ['train', 'test']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
        X, Y = get_batch(split)
        _, loss = model(X, Y)
        losses[k] = loss.item()
      out[split] = losses.mean() # mean loss over a bunch of evaluation iterations
   model.train()
   return out

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token simply picks the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # it is a parameter tensor

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx) # x tokens are used as indices of embedding table
        B, T, C = logits.shape

        if targets is None:
            loss = None
        
        else:
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self.forward(idx)

            # last time step element contains the prediction
            logits = logits[:,-1, :] # becomes (B, C) shape

            # convert to probabilities
            probs = F.softmax(logits, dim=1) # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sample to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)
        return idx

model = BigramModel(vocab_size=vocab_size)
model = model.to(device) # move model parameters to device

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iters):

    if iter % elval_intervals == 0:
       losses = estimate_loss()
       print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")

    # get a batch of x and y
    xb, yb = get_batch("train")

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # reset gradients from previous step to zero
    loss.backward() # calc gradients for all the parameters
    optimizer.step()

initial_context = torch.zeros((1, 1), dtype=torch.long, device=device) # start with token 0 (i.e. \u character)
gen = model.generate(initial_context, max_new_tokens=500)
prediction = decode(gen[0].tolist())

print(prediction)
