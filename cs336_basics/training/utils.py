import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
import math

def log_softmax(x: Float[Tensor, " ..."], dim=-1) -> Float[Tensor, " ..."]:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    # check equations 16 and 17, the log from 16 and the exp in the numerator from 17 cancels out
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))

# The equation: mean of neg log of the probabilities of the correct words
# The probabilities are obtained by softmax (logits) [correct word]
def cross_entropy_loss(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    neg_log_softmax_logits = -log_softmax(inputs)

    # 2. Create the row indices. This will just be [0, 1, 2, ..., batch_size-1]
    # This selects every row.
    row_indices = torch.arange(inputs.shape[0], device=inputs.device)

    # 3. Use the row_indices and the targets (as column indices) to
    #    gather the log probability of the correct class for each item in the batch.
    # Shape: (batch_size,)
    loss_per_item = neg_log_softmax_logits[row_indices, targets]

    return torch.mean(loss_per_item)

def cosine_lr_schedule(
    t: int, # The current iteration number
    max_lr: float, # Max learning rate
    min_lr: float, # Min learning rate
    t_warmup: int, # Number of warm-up iterations
    t_cosine: int # Number of cosine-annealing iterations
) -> float:
    if t < t_warmup:
        return t / t_warmup * max_lr
    if t_warmup <= t <= t_cosine:
        expr = (t - t_warmup) / (t_cosine - t_warmup) * math.pi
        big_expr = 0.5 * (1 + math.cos(expr)) * (max_lr - min_lr)
        return min_lr + big_expr
    # t > t_cosine
    return min_lr
