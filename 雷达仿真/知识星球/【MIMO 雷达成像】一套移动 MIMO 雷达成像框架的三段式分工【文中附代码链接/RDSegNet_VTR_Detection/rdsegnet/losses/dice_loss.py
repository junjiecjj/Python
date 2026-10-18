"""Soft Dice loss. Consumes raw logits when from_logits=True (default)."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss (smooth = Laplace smoothing; from_logits applies sigmoid)."""

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred [B, 1, H, W] logits; target [B, 1, H, W] binary {0, 1}
        if self.from_logits:
            pred = torch.sigmoid(pred)

        pred = pred.flatten(1)
        target = target.flatten(1)

        intersection = (pred * target).sum(dim=1)
        denominator = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()
