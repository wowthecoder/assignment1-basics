import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self,  
        in_features: int, # final dimension of the input
        out_features: int, # final dimension of the output
        device: torch.device | None = None, # Device to store the parameters on
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_in = in_features 
        self.d_out = out_features 
        self.device = device 
        self.dtype = dtype
        self.weights = nn.Parameter(torch.empty(out_features, in_features))
        self.init_weights()

    def init_weights(self):
        sigma = math.sqrt(2 / (self.d_in + self.d_out))
        nn.init.trunc_normal_(self.weights, mean=0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T