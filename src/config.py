from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Configuration for the Transformer model.
    """

    batch_size: int = 64  # How many independent sequences will we process in parallel?
    block_size: int = 256  # What is the maximum context length for predictions?
    n_embd: int = 384  # The number of embedding dimensions.
    n_head: int = 6  # The number of heads in the multi-head attention.
    n_layer: int = 6  # The number of transformer blocks.
    dropout: float = 0.2  # The dropout rate.
    learning_rate: float = 3e-4  # The learning rate for the optimizer.
    max_iters: int = 50000  # The total number of training iterations.
    eval_interval: int = 500  # How often to evaluate the model.
    eval_iters: int = 200  # The number of iterations to run for evaluation.
    device: str = "cuda"  # The device to run the model on, e.g., 'cpu' or 'cuda'.
