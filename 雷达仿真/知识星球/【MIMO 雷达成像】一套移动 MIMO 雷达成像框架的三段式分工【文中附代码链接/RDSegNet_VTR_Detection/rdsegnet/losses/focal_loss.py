"""Binary focal loss (from logits) for class-imbalanced segmentation.

Lin et al., ICCV 2017. Sparse RDM targets make focal loss down-weight the
abundant easy negatives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss from logits.

    alpha: positive-class weight in [0, 1] (negatives get 1-alpha).
    gamma: focusing strength. reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred [B, 1, H, W] logits; target [B, 1, H, W] binary {0, 1}
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        prob = torch.sigmoid(pred)
        p_t = prob * target + (1.0 - prob) * (1.0 - target)   # p for pos, 1-p for neg
        focal_weight = (1.0 - p_t).pow(self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
