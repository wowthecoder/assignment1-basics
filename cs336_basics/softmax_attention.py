from jaxtyping import Float
from torch import Tensor
import torch

def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    norm_x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(norm_x)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp
