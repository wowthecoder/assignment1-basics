import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.compute_buffer()

    def compute_buffer(self):
        rows, cols = self.max_seq_len, self.d_k // 2
        ii, jj = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')
        # At this point (example):
        # ii = [[0, 0, 0, 0, 0],
        #       [1, 1, 1, 1, 1],
        #       [2, 2, 2, 2, 2],
        #       [3, 3, 3, 3, 3]]
        #
        # jj = [[0, 1, 2, 3, 4],
        #       [0, 1, 2, 3, 4],
        #       [0, 1, 2, 3, 4],
        #       [0, 1, 2, 3, 4]]
        angles = self.compute_angles(ii, jj)
        rope_buffer = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        self.register_buffer("rope_buffer", rope_buffer, persistent=False)

    def compute_angles(self, ii: Tensor, kk: Tensor):
        # No need to -1 because kk is zero-indexed, so we need to +1 to offset then since k starts from 1 in the docs
        exp = (2 * kk) / self.d_k
        # ii is a position that starts from 0 so no need to +1
        angle = ii / (self.theta ** exp)
        return angle

    def forward(self,
        x: Float[Tensor, '... seq_len d_k'],
        token_positions: Int[Tensor, '... seq_len']
    ) -> Float[Tensor, '... seq_len d_k']:
        # The pattern splits the last dimension 'd' into pairs
        # This is to process even and odd indices in the embeddings separately
        x_pairs = rearrange(x, '... seq_len (d_half two) -> ... seq_len d_half two', two=2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        # Fetch the sin and cos values from the buffer that we need
        cache = self.rope_buffer[token_positions]
        cos = cache[..., 0]
        sin = cache[..., 1]

        x_rotated_even = cos * x_even - sin * x_odd
        x_rotated_odd = sin * x_even + cos * x_odd

        # Combine the results
        y = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).reshape(x.shape)
        return y
