import os
import torch
import torch.nn as nn
from torch.nn import functional as F
# import tiktoken # Not used, smallest embedding table is too large for potato laptop compute

#################
# hyperparameters
batch_size = 16
context_window = 8 # this is what karpathy calls block size in his code # same as N (Bishop, Deep Learning)
max_iters = 6000 
elval_intervals = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32 # aka d_model (vaswani 2017)
# head_size = 16 # aka d_k = d_q (vaswani) # not used in multi-head attention
h = 4 # head count

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


class Head(nn.Module):
  """One head of scaled dot-product attention"""
  def __init__(self, head_size):
    super().__init__()
    # key: what do i contain
    self.key = nn.Linear(n_embd, head_size, bias=False) # W_k (Bishop, Deep Learning)
    # query: what am i looking for
    self.query = nn.Linear(n_embd, head_size, bias=False) # W_q
    # value: aggregate the information
    self.value = nn.Linear(n_embd, head_size, bias=False) # W_v
    # buffer: not a parameter
    self.register_buffer('tril', torch.tril(torch.ones(context_window, context_window)))
  
  def forward(self, x):
    B, T, C = x.shape # T: N; C: d_model
    q = self.query(x) # Q (B, T, head_size = d_q)
    k = self.key(x) # (B, T, head_size = d_k)

    mat_mul = q @ k.transpose(-2, -1) # (B, T, head_size) @ (B, head_size, T) --> (B, T, T) Same as N * N (Bishop, Deep Learning)    

    scale = mat_mul * C**-0.5
    mask = scale.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # optional: used only in decoder 
    
    softmax = F.softmax(mask, dim=-1) # (B, T, T)

    v = self.value(x) # (B, T, head_size)

    attention = softmax @ v # (B, T, n_embd)
    return attention


class MultiHeadAttention(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(h)])

  def forward(self, x):
    return torch.cat([sa_head(x) for sa_head in self.heads], dim=-1)

class BigramModel(nn.Module):
  def __init__(self):
    super().__init__()
    # each token simply picks the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # aka input embeddings (vaswani 2017) 
    self.positional_encoding_table = nn.Embedding(context_window, n_embd)
    
    # multi-attention heads
    self.sa_heads = MultiHeadAttention(n_embd // h) # d_v = d_model / h # number of computations is same as single, larger self-attention head
    # conversion from token embeddings to logits
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_embd = self.token_embedding_table(idx) # (B, T, C=n_embd=d_model) # x tokens are used as indices of embedding table
    pos_enc = self.positional_encoding_table(torch.arange(T, device=device)) # (T, C)
    x = token_embd + pos_enc # (B, T, C)
    attention = self.sa_heads(x)
    logits = self.lm_head(attention) # (B, T, vocab_size)

    if targets is None:
      loss = None

    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      # crop idx to limit to context window, for positional encoding to work
      idx_cropped = idx[:,-context_window:]
      # get predictions
      logits, loss = self(idx_cropped)

      # last time step element contains the prediction
      logits = logits[:,-1, :] # becomes (B, C) shape

      # convert to probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)

      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

      # append sample to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)
    return idx


model = BigramModel()
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
