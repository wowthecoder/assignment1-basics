from jaxtyping import Float
from torch import Tensor
import torch
from einops import einsum
import math

def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    norm_x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(norm_x)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp

# pre-condition: keys == values
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    qt_k = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys')
    expr = qt_k / math.sqrt(d_k)
    if mask is not None:
        expr = expr.masked_fill(mask == 0, float('-inf'))

    softmaxxed = softmax(expr, dim=-1)
    res = einsum(softmaxxed, V, '... queries keys, ... keys d_v -> ... queries d_v')
    return res

