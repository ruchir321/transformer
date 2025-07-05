import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from tokenizers import Tokenizer
from transformer_101.modeling import TransformerModel
from transformer_101.configuration import TransformerConfig

def convert_weights():
    """
    Converts the original PyTorch model weights to the Hugging Face format.
    """
    # Load the tokenizer to get the vocabulary size
    tokenizer = Tokenizer.from_file("models/bpe_tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    # Create a Hugging Face configuration object with the correct parameters
    config = TransformerConfig(
        vocab_size=vocab_size,
        n_embd=384,
        n_layer=6,
        n_head=6,
        block_size=256,
        dropout=0.2,
    )

    # Load the original model's state dictionary
    original_state_dict = torch.load("models/transformer.pt", map_location="cpu")

    # Create an instance of the Hugging Face model
    model = TransformerModel(config)

    # Create a new state dictionary with the keys prefixed for the Hugging Face model
    new_state_dict = {}
    for key, value in original_state_dict.items():
        new_key = "transformer." + key
        new_state_dict[new_key] = value

    # Load the new state dictionary into the Hugging Face model
    model.load_state_dict(new_state_dict)

    # Save the Hugging Face model and configuration
    model.save_pretrained("transformer_101")

if __name__ == "__main__":
    convert_weights()