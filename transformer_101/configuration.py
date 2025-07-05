from transformers import PretrainedConfig

class TransformerConfig(PretrainedConfig):
    model_type = "transformer"

    def __init__(
        self,
        vocab_size=50257,
        n_embd=768,
        n_layer=12,
        n_head=12,
        block_size=256,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        dropout=0.2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.dropout = dropout

        super().__init__(**kwargs)