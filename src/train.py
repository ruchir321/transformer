import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ModelConfig
from src.model import Transformer


def get_tokenizer(dataset):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
    text_iterator = (item["text"] for item in dataset if item["text"])
    tokenizer.train_from_iterator(text_iterator, trainer)
    return tokenizer


def stream_data(dataset, tokenizer, block_size):
    """
    Create a token iterator from a streaming dataset.
    """
    buffer = []
    for example in dataset:
        if "text" in example and example["text"]:
            tokens = tokenizer.encode(example["text"]).ids
            buffer.extend(tokens)
            while len(buffer) >= block_size + 1:
                x = torch.tensor(buffer[:block_size], dtype=torch.long)
                y = torch.tensor(buffer[1 : block_size + 1], dtype=torch.long)
                yield x, y
                buffer = buffer[block_size:]


def batch_iterator(iterator, batch_size, device):
    """
    Create a batch iterator from a token iterator.
    """
    batch_x, batch_y = [], []
    for x, y in iterator:
        batch_x.append(x)
        batch_y.append(y)
        if len(batch_x) == batch_size:
            yield torch.stack(batch_x).to(device), torch.stack(batch_y).to(device)
            batch_x, batch_y = [], []


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def train():
    config = ModelConfig()

    # Load the full training data as a streaming dataset
    train_dataset_stream = load_dataset(
        "ajibawa-2023/WikiHow", split="train", streaming=True
    )

    # Create a small, in-memory validation set
    val_dataset_list = list(train_dataset_stream.take(2000))
    # The remainder of the stream is used for training
    train_dataset_stream = train_dataset_stream.skip(2000)

    # Train the tokenizer on the in-memory validation set
    tokenizer = get_tokenizer(val_dataset_list)
    vocab_size = tokenizer.get_vocab_size()

    # Shuffle the training dataset for randomness
    train_dataset_shuffled = train_dataset_stream.shuffle(seed=42, buffer_size=10000)

    model = Transformer(config, vocab_size)
    model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_token_iter = stream_data(train_dataset_shuffled, tokenizer, config.block_size)
    train_batch_iter = batch_iterator(
        train_token_iter, config.batch_size, config.device
    )

    train_losses = []
    for iter_num, (xb, yb) in enumerate(train_batch_iter):
        if iter_num >= config.max_iters:
            break

        logits, loss = model(xb, yb)
        train_losses.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_num > 0 and iter_num % config.eval_interval == 0:
            val_loss = estimate_loss(model, val_dataset_list, tokenizer, config)
            avg_train_loss = sum(train_losses) / len(train_losses)
            print(
                f"step {iter_num}: train loss {avg_train_loss:.4f}, val loss {val_loss:.4f}"
            )
            train_losses = []

    torch.save(model.state_dict(), "models/transformer.pt")
    tokenizer.save("models/bpe_tokenizer.json")


@torch.no_grad()
def estimate_loss(model, val_dataset, tokenizer, config):
    model.eval()
    val_text = [item["text"] for item in val_dataset if item["text"]]
    val_encodings = tokenizer.encode_batch(val_text)
    val_tokens = [token for enc in val_encodings for token in enc.ids]
    val_data = torch.tensor(val_tokens, dtype=torch.long)

    losses = torch.zeros(config.eval_iters)
    for k in range(config.eval_iters):
        X, Y = get_batch(val_data, config.block_size, config.batch_size, config.device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    val_loss = losses.mean()
    model.train()
    return val_loss


if __name__ == "__main__":
    train()
