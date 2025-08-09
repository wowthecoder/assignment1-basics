import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import repeat

from .rope import RotaryPositionalEmbedding
from .multihead_attention import MultiheadSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .embedding import Embedding
from .linear import Linear

class TransformerBlock(nn.Module):
    def __init__(self,
        d_model: int, # Dimensionality of the Transformer block inputs.
        num_heads: int, # Number of heads to use in multi-head self-attention.
        d_ff: int, # Dimensionality of the position-wise feed-forward inner layer.
        positionEncoder: RotaryPositionalEmbedding | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.positionEncoder = positionEncoder

        # Sub-layer 1: Multi-head self-attention
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, positionEncoder)

        # Sub-layer 2: Position-wise feed-forward network
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def generate_token_positions(self, x: Float[Tensor, " batch sequence_length d_model"]) -> Int[Tensor, " batch sequence_length"]:
        batch_size, seq_len = x.shape[:2]
        # Create base positions [0, 1, 2, ..., seq_len - 1]
        base_positions = torch.arange(seq_len, device=x.device)
        # Repeat for each batch: (seq_len,) -> (batch_size, seq_len)
        token_positions = repeat(base_positions, 'seq -> batch seq', batch=batch_size)
        return token_positions

    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]) -> Float[Tensor, " batch sequence_length d_model"]:
        # Causal Multihead Self Attention with RoPE
        token_positions = self.generate_token_positions(x)
        attention = self.attn(self.ln1(x), token_positions)
        x = x + attention

        # Position-wise feedforward network
        ffn_output = self.ffn(self.ln2(x))
        ffn_output = x + ffn_output

        return ffn_output

class TransformerLM(nn.Module):
    def __init__(self,
        vocab_size: int, # The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
        context_length: int, # The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
        d_model: int,
        num_layers: int, # The number of Transformer blocks to use.
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        d_k = d_model // num_heads
        # Use the same positional embedding because it doesn't have any learnable parameters
        self.positionEncoder = RotaryPositionalEmbedding(rope_theta, d_k, max_seq_len=context_length)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, self.positionEncoder)
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        # PART 1: Convert each token to embeddings
        x = self.token_embeddings(x)

        # PART 2: Pass the embeddings through all the transformer blocks
        for layer in self.layers:
            x = layer(x)

        # PART 3: Normalise output of transformers
        x = self.ln_final(x)

        # PART 4: Generate output embeddings
        x = self.lm_head(x)

        return x

