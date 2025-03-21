# implementation of a decoder-only transformer for text generation task
# GPT family of models is decoder-only architecture
# Vaswani's encoder-decoder architecture was purpose built for machine translation

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
# import tiktoken # Not used, smallest embedding table is too large for potato laptop compute

#################
# hyperparameters
batch_size = 64
context_window = 256 # this is what karpathy calls block size in his code # same as N (Bishop, Deep Learning)
max_iters = 6000
elval_intervals = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # aka d_model (vaswani 2017)
# head_size = 16 # aka d_k = d_q (vaswani) # not used in multi-head attention
n_heads = 6 # aka h (vaswani) # head count
n_layers = 6
dropout = 0.2

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
    self.dropout = nn.Dropout(p=dropout)
  
  def forward(self, x):
    B, T, C = x.shape # T: N; C: d_model
    q = self.query(x) # Q (B, T, head_size = d_q)
    k = self.key(x) # (B, T, head_size = d_k)

    mat_mul = q @ k.transpose(-2, -1) # (B, T, head_size) @ (B, head_size, T) --> (B, T, T) Same as N * N (Bishop, Deep Learning)    

    scale = mat_mul * C**-0.5
    mask = scale.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # optional: used only in decoder 
    
    softmax = F.softmax(mask, dim=-1) # (B, T, T)
    softmax = nn.Dropout(p=dropout)

    v = self.value(x) # (B, T, head_size)

    attention = softmax @ v # (B, T, n_embd)
    return attention


class MultiHeadAttention(nn.Module):
  """Multiple heads of scaled-dot product attention in parallel"""

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd) # projection W_o => dim(n_heads * d_v, d_model) (Vaswani et.al) # (n_heads * d_v) = d_model = n_embd
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
    out = torch.cat([sa_head(x) for sa_head in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class LayerNorm(nn.Module):
  """NOTE: UDF LayerNorm is not used for the model
  A neuron must have a unit Gaussian distribution across feature dimension of each token"""
  def __init(self, dim, eps=1e-5, momentum=0.1):
    super().__init__()
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate forward pass
    xmean = x.mean(1, keepdim=True) # layer mean
    xvar = x.var(1, keepdim=True) # layer variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = xhat * self.gamma + self.beta


class Feedforward(nn.Module):
  """ A simple linear layer followed by a non-linearity"""
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd), # in Vaswani et. al. d_model = 512 & d_ff = 2048. So FF layer dimension has a multiplier of 4 in the inner layer
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd), # same as initializing a self.proj attribute
      nn.Dropout(p=dropout)
    )
  
  def forward(self, x):
    return self.net(x)

# Block: called "layer" in Vaswani et.al.
class Block(nn.Module):
  """Transformer block: communication then computation"""

  def __init__(self, n_embd, n_heads):
    super().__init__()
    head_size = n_embd // n_heads
    self.sa = MultiHeadAttention(num_heads=n_heads, head_size=head_size)
    self.ffwd = Feedforward(n_embd=n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    # communication
    # addition denotes residual connection
    # layernorm is applied before the transformation
    #  This pre-norm implementation is different from the original paper
    x = x + self.sa(self.ln1(x))
    # computation
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramModel(nn.Module):
  def __init__(self):
    super().__init__()
    # each token simply picks the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # aka input embeddings (vaswani 2017) 
    self.positional_encoding_table = nn.Embedding(context_window, n_embd)
    
    self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_heads=n_heads) for _ in range(n_layers)])
    
    # multi-attention heads
    self.sa_heads = MultiHeadAttention(num_heads=n_heads, head_size=n_embd // n_heads) # d_v = d_model / n_heads # number of computations is same as single, larger self-attention head
    
    self.ffwd = Feedforward(n_embd=n_embd)
    
    # conversion from token embeddings to logits
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_embd = self.token_embedding_table(idx) # (B, T, C=n_embd=d_model) # x tokens are used as indices of embedding table
    pos_enc = self.positional_encoding_table(torch.arange(T, device=device)) # (T, C)
    x = token_embd + pos_enc # (B, T, C)
    attention = self.blocks(x)
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
