"""Tversky / Focal-Tversky loss.

T = TP / (TP + alpha*FP + beta*FN); beta > alpha favors recall of sparse
structure. The focal variant raises (1 - T) to gamma to focus on hard samples.
"""

import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    """(Focal-)Tversky loss.

    alpha: FP weight; beta: FN weight (>alpha favors recall); gamma: focal
    exponent on (1-T), 1.0 = plain Tversky; smooth: Laplace smoothing.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.0,
                 smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred [B, 1, H, W] logits; target [B, 1, H, W] binary {0, 1}
        if self.from_logits:
            pred = torch.sigmoid(pred)

        pred = pred.flatten(1)
        target = target.flatten(1)

        tp = (pred * target).sum(dim=1)
        fp = (pred * (1.0 - target)).sum(dim=1)
        fn = ((1.0 - pred) * target).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = (1.0 - tversky)
        if self.gamma != 1.0:
            loss = loss.clamp_min(0.0).pow(self.gamma)
        return loss.mean()
