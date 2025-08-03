import torch 
import torch.nn as nn 
from einops import einsum

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
        self.weights1 = nn.Parameter(torch.randn(self.d_ff, self.d_model))
        self.weights2 = nn.Parameter(torch.randn(self.d_model, self.d_ff))
        self.weights3 = nn.Parameter(torch.randn(self.d_ff, self.d_model))

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        w1x = einsum(x, self.weights1, '... d_model, d_ff d_model -> ... d_ff')
        w3x = einsum(x, self.weights3, '... d_model, d_ff d_model -> ... d_ff')
        expr = self.silu(w1x) * w3x # element wise multiplication
        res = einsum(expr, self.weights2, '... d_ff, d_model d_ff -> ... d_model')
        return res