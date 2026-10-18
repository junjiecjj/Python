"""RDSegNet: U-shaped complex-valued segmentation network for RDM VTR detection.

Encoder (complex channels base, 2x per stage) with per-stage log-amplitude
channel cross-attention, a bottleneck with spatial self-attention, and a
mirrored decoder with CV-UpConv + skip concat. Complex channel count c maps to
2c real channels (interleaved). Output: raw logits [B, 1, H, W] (sigmoid only at
inference).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ChannelCrossAttention, SpatialSelfAttention
from .complex_layers import (
    ComplexConv2d,
    ComplexConvBlock,
    ComplexConvTranspose2d,
    ComplexDownsample,
)


class RDSegNet(nn.Module):
    def __init__(
        self,
        in_complex_channels: int = 3,
        base_complex_channels: int = 32,
        depth: int = 3,
        use_cross_attention: bool = True,
        use_self_attention: bool = True,
        attention_hidden: int = 32,
        cross_attention_weight: float = 1.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.in_complex_channels = in_complex_channels
        self.depth = depth

        enc_channels = [base_complex_channels * (2 ** i) for i in range(depth)]   # e.g. [32, 64, 128]
        bottleneck_channels = base_complex_channels * (2 ** depth)                # e.g. 256

        # Encoder
        encoders = []
        prev = in_complex_channels
        for c in enc_channels:
            encoders.append(ComplexConvBlock(prev, c))
            prev = c
        self.encoders = nn.ModuleList(encoders)

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            # One cross-attention module after every encoder stage.
            self.cross_attn = nn.ModuleList(
                ChannelCrossAttention(c, guide_channels=in_complex_channels,
                                      hidden=attention_hidden,
                                      weight=cross_attention_weight)
                for c in enc_channels
            )
        else:
            self.cross_attn = None

        self.down = ComplexDownsample(2)

        # Bottleneck
        self.bottleneck = ComplexConvBlock(enc_channels[-1], bottleneck_channels)
        self.use_self_attention = use_self_attention
        self.self_attn = (
            SpatialSelfAttention(2 * bottleneck_channels) if use_self_attention else None
        )

        # Decoder (skip concat doubles channels)
        ups, decoders = [], []
        prev = bottleneck_channels
        for c in reversed(enc_channels):
            ups.append(ComplexConvTranspose2d(prev, c, kernel_size=2, stride=2))
            decoders.append(ComplexConvBlock(2 * c, c))
            prev = c
        self.ups = nn.ModuleList(ups)
        self.decoders = nn.ModuleList(decoders)

        # Segmentation head: CV-Conv to 1 complex channel, then real 1x1 -> logit.
        self.head_complex = ComplexConv2d(enc_channels[0], 1, kernel_size=1, padding=0)
        self.head_real = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, complex_input: torch.Tensor, log_amp: torch.Tensor) -> torch.Tensor:
        """complex_input [B, 2*in, H, W] interleaved, log_amp [B, in, H, W] -> logits [B, 1, H, W]."""
        if complex_input.shape[1] != 2 * self.in_complex_channels:
            raise ValueError(
                f"complex_input has {complex_input.shape[1]} channels, expected "
                f"{2 * self.in_complex_channels} (= 2 x {self.in_complex_channels} complex channels)."
            )

        skips = []
        x = complex_input
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if self.cross_attn is not None:
                x = self.cross_attn[i](x, log_amp)
            skips.append(x)
            x = self.down(x)

        x = self.bottleneck(x)
        if self.self_attn is not None:
            x = self.self_attn(x)

        for j in range(self.depth):
            skip = skips[self.depth - 1 - j]
            x = self.ups[j](x)
            if x.shape[-2:] != skip.shape[-2:]:               # guard for odd sizes
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)                   # interleaving preserved
            x = self.decoders[j](x)

        x = self.head_complex(x)                              # [B, 2, H, W]
        logits = self.head_real(x)                            # [B, 1, H, W]
        return logits


def build_model(cfg: dict) -> RDSegNet:
    """Instantiate RDSegNet from cfg['model']."""
    m = dict(cfg.get("model", {}))
    return RDSegNet(
        in_complex_channels=int(m.get("in_complex_channels", 3)),
        base_complex_channels=int(m.get("base_complex_channels", 32)),
        depth=int(m.get("depth", 3)),
        use_cross_attention=bool(m.get("use_cross_attention", True)),
        use_self_attention=bool(m.get("use_self_attention", True)),
        cross_attention_weight=float(m.get("cross_attention_weight", 1.0)),
    )
