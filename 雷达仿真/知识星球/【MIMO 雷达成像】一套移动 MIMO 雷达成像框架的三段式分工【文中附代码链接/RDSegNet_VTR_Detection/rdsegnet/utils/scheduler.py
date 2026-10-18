"""Learning-rate scheduler: linear warm-up then cosine annealing.

Implemented as a per-iteration LambdaLR (stepped once per optimizer update, not
per epoch) for a smoother LR curve on short runs.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 5,
    min_lr_ratio: float = 1e-3,
) -> LambdaLR:
    """Linear warm-up for warmup_epochs then cosine annealing.

    Stepped once per optimizer update. min_lr_ratio is the floor LR as a
    fraction of the base LR (min_lr / base_lr).
    """
    total_steps = max(int(epochs * steps_per_epoch), 1)
    warmup_steps = max(int(warmup_epochs * steps_per_epoch), 0)
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
