import os
import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch.nn.modules import activation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# used for DJSCC
# calculate the last conv2d layer channels according to the compress ratio
def Calculate_filters(comp_ratio, F=5, n=3072):
    K = (comp_ratio * n) / F**2
    return int(K)


def crop(inputs, len):
    (b, c, h, w) = inputs.shape
    inputs = inputs[:, :, len:h - len, len:w - len]
    return inputs


# Normalization + AWGN  1dim
class Real_AWGN_Channel(nn.Module):
    def __init__(self):
        super(Real_AWGN_Channel, self).__init__()

    # default power = 1.0
    def forward(self, x, snr):
        # power normalization
        # flatten
        y = x.view(x.shape[0], -1)
        k = y.shape[1]
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True).repeat([1, y.shape[-1]])
        x_norm = torch.div(y, y_norm) * math.sqrt(k)
        x_norm = x_norm.view(x.shape)
        assert compute_power(x_norm) - 1.0 <= 1e-5
        # print("mean power : ", compute_power(x_norm))
        # generate noise
        var = np.power(10.0, -0.1 * snr)
        sigma = np.sqrt(var)
        # print("snr :", snr, ", sigma:", sigma)
        noise = (torch.randn(size=x.shape) * sigma).to(device)
        return x_norm + noise



def AWGN(Tx_sig, n_var):
    Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
    return Rx_sig


def PowerNormalize(x):

    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)

    return x


def SNR_to_noise(snr):
    snr = 10**(snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std


def awgn(x, snr):
    noise_std = SNR_to_noise(snr)
    # print("snr :", snr, ", sigma:", noise_std)
    x_norm = PowerNormalize(x)
    x_output = AWGN(x_norm, noise_std)
    return x_output


def compute_power(x):
    abs_x = torch.abs(x)
    summation = torch.square(abs_x).mean().sqrt()
    return summation


# dense layer is a full connection layer and used to gather information
def dense(input_size, output_size):
    return torch.nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())


def conv2d_prelu(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=True,
        ),
        nn.PReLU(),
    )


def convTrans2d_prelu(in_channels, out_channels, kernel_size, stride, pad=0, out_pad=0):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=True,
        ),
        nn.PReLU(),
    )


def default_conv(in_channels, out_channels, kernel_size, pad=None, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2) if pad is None else pad, bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


########################### end ############################################