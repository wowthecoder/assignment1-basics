import torch
from typing import Optional
from collections.abc import Callable
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # Store all hyperparameters in defaults
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data

                # Update the first moment estimate
                m = (beta1 * m) + (1 - beta1) * grad
                # Update the second moment estimate
                v = (beta2 * v) + (1 - beta2) * grad * grad
                # Compute adjusted lr for iteration t
                lr_t = lr * math.sqrt(1 - (beta2 ** t)) / (1 - (beta1 ** t))
                # Update the parameters
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                # Apply weight decay
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
