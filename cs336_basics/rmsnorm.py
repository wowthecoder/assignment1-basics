import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from einops import reduce, einsum

class RMSNorm(nn.Module):
    def __init__(self,
        d_model: int, # Hidden dimension of the model
        eps: float = 1e-5, # Epsilon value for numerical stability
        device: torch.device | None = None, # Device to store the parameters on
        dtype: torch.dtype | None = None # Data type of the parameters
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        # Upcast type
        in_dtype = x.dtype
        x = x.to(torch.float32)

        squared_sum = reduce(x ** 2, '... d_model -> ...', 'sum')
        expr = (squared_sum / self.d_model) + self.eps
        rms = torch.sqrt(expr)
        prod = einsum(x, self.weight, '... d_model, d_model -> ... d_model')
        # unsqueeze rms to dimension (..., 1)
        res = prod / rms.unsqueeze(-1)
        return res.to(in_dtype)
