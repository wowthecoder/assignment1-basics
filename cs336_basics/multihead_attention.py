import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int
from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .softmax_attention import scaled_dot_product_attention

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, positionEncoder: RotaryPositionalEmbedding | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        self.positionEncoder = positionEncoder
        self.init_weights()

    def init_weights(self):
        h_dk = self.num_heads * self.d_k
        h_dv = self.num_heads * self.d_v
        self.q_proj = Linear(self.d_model, h_dk)
        self.k_proj = Linear(self.d_model, h_dk)
        self.v_proj = Linear(self.d_model, h_dv)
        self.output_proj = Linear(h_dv, self.d_model)

    def forward(self,
        x: Float[torch.Tensor, '... seq_len d_model'],
        token_positions: Int[torch.Tensor, " ... seq_len"] | None = None
    ) -> Float[torch.Tensor, '... seq_len d_model']:
        # All of the below has dimensions [... seq_len (h_dk OR h_dv)]
        wq_x, wk_x, wv_x = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        wq_x = rearrange(wq_x, '... seq_len (h dk) -> ... h seq_len dk', h=self.num_heads, dk=self.d_k)
        wk_x = rearrange(wk_x, '... seq_len (h dk) -> ... h seq_len dk', h=self.num_heads, dk=self.d_k)
        wv_x = rearrange(wv_x, '... seq_len (h dv) -> ... h seq_len dv', h=self.num_heads, dv=self.d_v)

        # construct the mask
        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))

        # apply positional encoding
        if self.positionEncoder is not None and token_positions is not None:
            wq_x = self.positionEncoder(wq_x, token_positions)
            wk_x = self.positionEncoder(wk_x, token_positions)

        attention = scaled_dot_product_attention(wq_x, wk_x, wv_x, mask=causal_mask)
        attention = rearrange(attention, '... h seq_len dv -> ... seq_len (h dv)', h=self.num_heads, dv=self.d_v)

        output = self.output_proj(attention)
        return output


