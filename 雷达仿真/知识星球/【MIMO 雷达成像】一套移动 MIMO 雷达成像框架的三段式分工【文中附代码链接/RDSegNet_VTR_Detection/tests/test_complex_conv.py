"""Tests for complex-valued layers."""

import torch

from rdsegnet.models.complex_layers import (
    ComplexConv2d,
    ComplexConvTranspose2d,
    merge_complex,
    split_complex,
)


def test_complex_conv_output_shape():
    conv = ComplexConv2d(3, 8, kernel_size=3, padding=1)
    x = torch.randn(2, 6, 32, 32)  # 3 complex channels -> 6 real channels
    y = conv(x)
    assert y.shape == (2, 16, 32, 32)  # 8 complex channels -> 16 real channels


def test_split_merge_roundtrip():
    x = torch.randn(2, 6, 8, 8)
    real, imag = split_complex(x)
    assert real.shape == (2, 3, 8, 8) and imag.shape == (2, 3, 8, 8)
    y = merge_complex(real, imag)
    assert torch.equal(x, y)


def test_complex_conv_matches_native_complex_math():
    """A 1x1 ComplexConv2d must agree with torch native complex matmul."""
    torch.manual_seed(0)
    conv = ComplexConv2d(2, 3, kernel_size=1, bias=False)

    x = torch.randn(1, 4, 5, 5)  # 2 complex channels
    real, imag = split_complex(x)
    x_c = torch.complex(real, imag)  # [1, 2, 5, 5] complex

    # Complex weight from the two real kernels: [out_c, in_c] for k=1.
    w_c = torch.complex(conv.conv_real.weight, conv.conv_imag.weight)
    w_c = w_c.squeeze(-1).squeeze(-1)  # [3, 2]

    expected = torch.einsum("oi,bihw->bohw", w_c, x_c)

    y = conv(x)
    y_real, y_imag = split_complex(y)
    got = torch.complex(y_real, y_imag)

    assert torch.allclose(got, expected, atol=1e-5), \
        "ComplexConv2d disagrees with native complex arithmetic"


def test_complex_conv_transpose_doubles_spatial():
    up = ComplexConvTranspose2d(4, 2, kernel_size=2, stride=2)
    x = torch.randn(2, 8, 16, 16)  # 4 complex channels
    y = up(x)
    assert y.shape == (2, 4, 32, 32)  # 2 complex channels, 2x spatial
