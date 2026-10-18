"""Tests for the Tversky / Lovasz losses and the extended CombinedLoss."""

import torch

from rdsegnet.losses import CombinedLoss, TverskyLoss, lovasz_hinge


def _sample():
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = (torch.rand(2, 1, 64, 64) > 0.9).float()   # ~10% sparse foreground
    return logits, target


def test_tversky_finite_and_backward():
    logits, target = _sample()
    loss = TverskyLoss(alpha=0.3, beta=0.7)(logits, target)
    assert torch.isfinite(loss) and loss.item() >= 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_lovasz_finite_and_backward():
    logits, target = _sample()
    loss = lovasz_hinge(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_combined_loss_all_terms_and_ramp():
    logits, target = _sample()
    crit = CombinedLoss(bce_weight=0.5, focal_weight=0.5, dice_weight=0.0,
                        tversky_weight=1.0, lovasz_weight=0.5, focal_alpha=0.75,
                        tversky_alpha=0.3, tversky_beta=0.7)

    crit.set_lovasz_ramp(0.0)
    loss_off = crit(logits, target)
    comp = dict(crit.components)
    for key in ("bce", "focal", "dice", "tversky", "lovasz", "total"):
        assert key in comp
    assert comp["dice"] == 0.0          # dice disabled

    crit.set_lovasz_ramp(1.0)
    loss_on = crit(logits, target)
    # ramping Lovasz in changes the total (same inputs, different effective weight)
    assert not torch.allclose(loss_off, loss_on)
    assert torch.isfinite(loss_on)
    loss_on.backward()
    assert torch.isfinite(logits.grad).all()
