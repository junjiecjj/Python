"""Combined segmentation loss = weighted BCE + Focal + Dice + Tversky + Lovasz.

All terms consume raw logits (sigmoid handled internally). The Lovasz term is
ramped in via set_lovasz_ramp(r) because it is noisy early. Unweighted component
values are stashed in self.components after each forward pass for logging.
"""

from typing import Dict

import torch
import torch.nn as nn

from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .lovasz_loss import lovasz_hinge
from .tversky_loss import TverskyLoss


class CombinedLoss(nn.Module):
    """Weighted sum of BCE, focal, dice, Tversky and Lovasz-hinge losses."""

    def __init__(self, bce_weight: float = 0.0, focal_weight: float = 1.0,
                 dice_weight: float = 1.0, tversky_weight: float = 0.0,
                 lovasz_weight: float = 0.0,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 tversky_alpha: float = 0.3, tversky_beta: float = 0.7,
                 tversky_gamma: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.lovasz_weight = lovasz_weight
        self.lovasz_ramp = 1.0   # multiplied into lovasz_weight; set per epoch

        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss(from_logits=True)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta,
                                   gamma=tversky_gamma, from_logits=True)

        self.components: Dict[str, float] = {}   # unweighted values, filled per forward

    def set_lovasz_ramp(self, ramp: float) -> None:
        """Set the Lovasz ramp factor, clamped to [0, 1]."""
        self.lovasz_ramp = float(max(0.0, min(1.0, ramp)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred [B, 1, H, W] logits; target [B, 1, H, W] binary {0, 1}
        zero = pred.new_zeros(())
        bce_loss = self.bce(pred, target) if self.bce_weight else zero
        focal_loss = self.focal(pred, target) if self.focal_weight else zero
        dice_loss = self.dice(pred, target) if self.dice_weight else zero
        tversky_loss = self.tversky(pred, target) if self.tversky_weight else zero
        lov_eff = self.lovasz_weight * self.lovasz_ramp
        lovasz_loss = lovasz_hinge(pred, target) if self.lovasz_weight else zero

        total = (self.bce_weight * bce_loss
                 + self.focal_weight * focal_loss
                 + self.dice_weight * dice_loss
                 + self.tversky_weight * tversky_loss
                 + lov_eff * lovasz_loss)

        self.components = {
            "bce": float(bce_loss.detach()),
            "focal": float(focal_loss.detach()),
            "dice": float(dice_loss.detach()),
            "tversky": float(tversky_loss.detach()),
            "lovasz": float(lovasz_loss.detach()),
            "total": float(total.detach()),
        }
        return total
