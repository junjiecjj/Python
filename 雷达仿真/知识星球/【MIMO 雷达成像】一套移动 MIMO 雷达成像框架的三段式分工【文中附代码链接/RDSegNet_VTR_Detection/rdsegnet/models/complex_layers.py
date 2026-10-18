"""Complex-valued building blocks.

A tensor with C complex channels is stored as a real tensor of 2C channels in
interleaved order [re_0, im_0, re_1, im_1, ...]. This matches the dataset output
and is preserved by torch.cat, so U-Net skip connections stay valid complex
tensors. split_complex / merge_complex convert to/from (real, imag).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def split_complex(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved [B, 2C, H, W] -> (real, imag), each [B, C, H, W]."""
    if x.shape[1] % 2 != 0:
        raise ValueError(
            f"Expected an even number of channels (interleaved real/imag), got {x.shape[1]}."
        )
    return x[:, 0::2], x[:, 1::2]


def merge_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    """(real, imag) [B, C, H, W] -> interleaved [B, 2C, H, W]."""
    b, c, h, w = real.shape
    return torch.stack((real, imag), dim=2).reshape(b, 2 * c, h, w)


class ComplexConv2d(nn.Module):
    """Complex 2-D convolution; channel counts are COMPLEX (real tensors hold 2x).

    (W_r + jW_i)(x_r + jx_i) = (W_r*x_r - W_i*x_i) + j(W_r*x_i + W_i*x_r)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = split_complex(x)
        yr = self.conv_real(xr) - self.conv_imag(xi)
        yi = self.conv_real(xi) + self.conv_imag(xr)
        return merge_complex(yr, yi)


class ComplexConvTranspose2d(nn.Module):
    """Complex transposed convolution (CV-UpConv 2x2); defaults double the size."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deconv_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        self.deconv_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = split_complex(x)
        yr = self.deconv_real(xr) - self.deconv_imag(xi)
        yi = self.deconv_real(xi) + self.deconv_imag(xr)
        return merge_complex(yr, yi)


class ComplexBatchNorm2d(nn.Module):
    """Paired BatchNorm: real/imag normalized together as 2C real channels."""

    def __init__(self, num_complex_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(2 * num_complex_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class CVReLU(nn.Module):
    """Complex ReLU: ReLU on real and imag separately (elementwise here)."""

    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x, inplace=self.inplace)


class ComplexDownsample(nn.Module):
    """2x average pooling: a linear op valid on real/imag alike (unlike max pool)."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.pool = nn.AvgPool2d(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class ComplexConvBlock(nn.Module):
    """[CV-Conv -> paired BN -> CV-ReLU] x2 (complex channel counts)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ComplexConv2d(in_channels, out_channels, kernel_size=3, bias=False),
            ComplexBatchNorm2d(out_channels),
            CVReLU(),
            ComplexConv2d(out_channels, out_channels, kernel_size=3, bias=False),
            ComplexBatchNorm2d(out_channels),
            CVReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
