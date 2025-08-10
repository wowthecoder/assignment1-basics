import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

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
