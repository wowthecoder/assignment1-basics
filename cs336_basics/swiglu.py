import torch
import torch.nn as nn
from einops import einsum
from .linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        self.d_model = d_model
        if d_ff and d_ff != 0:
            self.d_ff = d_ff
        else:
            self.calculate_dff()
        self.init_weights()

    def calculate_dff(self):
        target_dff = 8 / 3 * self.d_model
        d_ff = round(target_dff / 64) * 64
        self.d_ff = int(d_ff)

    def init_weights(self):
        self.w1 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, self.d_model)
        self.w3 = Linear(self.d_model, self.d_ff)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        expr = self.silu(self.w1(x)) * self.w3(x) # element wise multiplication
        res = self.w2(expr)
        return res
