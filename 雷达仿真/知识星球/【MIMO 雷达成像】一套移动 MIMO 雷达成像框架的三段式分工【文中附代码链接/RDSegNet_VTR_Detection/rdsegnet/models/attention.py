"""Attention modules.

ChannelCrossAttention: log-amplitude-guided channel attention over complex
features (SE-style). SpatialSelfAttention: SAGAN-style spatial self-attention
used at the bottleneck.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelCrossAttention(nn.Module):
    """Log-amplitude guided channel attention for complex features.

    The guide is pooled into one weight per complex channel; the same weight
    scales that channel's real and imag parts, so phase is preserved.

    features: [B, 2C, H, W] interleaved; log_amp: [B, G, H0, W0] -> [B, 2C, H, W]
    """

    def __init__(self, num_complex_channels: int, guide_channels: int = 3, hidden: int = 32,
                 weight: float = 1.0):
        super().__init__()
        self.num_complex_channels = num_complex_channels
        # Forward-only blend scalar (not a Parameter, so the state_dict is
        # unchanged): 1.0 = original SE gating, 0.0 = identity, between = blend.
        self.weight = float(weight)
        self.guide_encoder = nn.Sequential(
            nn.Conv2d(guide_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, hidden, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_complex_channels),
        )

    def forward(self, features: torch.Tensor, log_amp: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.mlp(self.guide_encoder(log_amp)))   # [B, C]
        weights = weights.repeat_interleave(2, dim=1)                    # [B, 2C], pair (re, im)
        if self.weight != 1.0:
            weights = (1.0 - self.weight) + self.weight * weights
        return features * weights.unsqueeze(-1).unsqueeze(-1)


class SpatialSelfAttention(nn.Module):
    """SAGAN-style spatial self-attention; x: [B, C, H, W] -> residual [B, C, H, W].

    Treats the complex tensor as real concatenated channels. Used at the
    bottleneck where H*W is small.
    """

    def __init__(self, real_channels: int, reduction: int = 8):
        super().__init__()
        inner = max(real_channels // reduction, 1)
        self.query = nn.Conv2d(real_channels, inner, kernel_size=1)
        self.key = nn.Conv2d(real_channels, inner, kernel_size=1)
        self.value = nn.Conv2d(real_channels, real_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))   # start as identity
        self.scale = inner ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x).flatten(2).transpose(1, 2)          # [B, HW, C']
        k = self.key(x).flatten(2)                            # [B, C', HW]
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)  # [B, HW, HW]
        v = self.value(x).flatten(2)                          # [B, C, HW]
        out = torch.bmm(v, attn.transpose(1, 2))              # [B, C, HW]
        out = out.reshape(b, c, h, w)
        return x + self.gamma * out
