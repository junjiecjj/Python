"""Lovasz-hinge loss: differentiable surrogate for IoU (Berman et al., CVPR 2018).

Consumes RAW signed logits (not probabilities), computed per-image. Relies on a
sort + cumsum, so keep it in fp32 (disable autocast around callers under AMP).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension of the Jaccard loss for sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovasz hinge on flattened logits and {0, 1} labels."""
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0          # {0,1} -> {-1,+1}
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor,
                 per_image: bool = True) -> torch.Tensor:
    """Lovasz hinge over a batch of [B, 1, H, W] logits / {0, 1} labels."""
    logits = logits.float()
    if per_image:
        losses = [lovasz_hinge_flat(lg.reshape(-1), lb.reshape(-1))
                  for lg, lb in zip(logits, labels)]
        return torch.stack(losses).mean()
    return lovasz_hinge_flat(logits.reshape(-1), labels.reshape(-1))


class LovaszHingeLoss(nn.Module):
    """Module wrapper around lovasz_hinge."""

    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return lovasz_hinge(pred, target, per_image=self.per_image)
