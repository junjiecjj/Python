"""Tests for the RDSegNet model forward/backward."""

import torch

from rdsegnet.losses import CombinedLoss
from rdsegnet.models import RDSegNet


def test_model_forward_shape():
    model = RDSegNet(in_complex_channels=3, base_complex_channels=8,
                     depth=3, use_cross_attention=True,
                     use_self_attention=True)
    x = torch.randn(2, 6, 64, 64)
    a = torch.randn(2, 3, 64, 64)
    y = model(x, a)
    assert y.shape == (2, 1, 64, 64)


def test_model_backward_finite_grads():
    model = RDSegNet(in_complex_channels=3, base_complex_channels=8,
                     depth=3, use_cross_attention=True,
                     use_self_attention=True)
    x = torch.randn(2, 6, 64, 64)
    a = torch.randn(2, 3, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.9).float()

    loss = CombinedLoss()(model(x, a), target)
    loss.backward()

    assert torch.isfinite(loss)
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_model_without_attention():
    model = RDSegNet(in_complex_channels=3, base_complex_channels=8,
                     depth=3, use_cross_attention=False,
                     use_self_attention=False)
    x = torch.randn(1, 6, 64, 64)
    a = torch.randn(1, 3, 64, 64)
    y = model(x, a)
    assert y.shape == (1, 1, 64, 64)
