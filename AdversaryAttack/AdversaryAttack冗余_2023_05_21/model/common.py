
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""

# 系统库
import math
import sys
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





# # 自己的库
# sys.path.append("..")
# from  Option import args

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

# 根据压缩比和输出输入的图像大小计算压缩层的输出通道数。
def calculate_channel(comp_ratio, F=5, n=3*48*48):
    K = (comp_ratio * n) / F**2
    return int(K)



def conv2d_prelu(in_channels, out_channels, kernel_size, stride, pad=0):
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


"""
归一化处理,先是归一化,再是去归一化,变量sign控制
"""
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        # torch.nn.Conv2d函数调用后会自动初始化weight和bias，涉及自定义weight和bias为需要的数均分布类型:
        # .weight.data是卷积核参数, .bias.data是偏置项参数
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

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
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                # # Pixelshuffle会将shape为 (*, r^2C, H, W) 的Tensor给reshape成 (*, C, rH,rW) 的Tensor
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

# device = torch.device(args.device if torch.cuda.is_available() else "cpu")

def Awgn(Tx_sig, n_var, device):
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


# 先将信号功率归一化，再计算噪声功率，再计算加躁的信号。
def AWGN(x, snr):
    noise_std = SNR_to_noise(snr)
    # print("snr :", snr, ", sigma:", noise_std)
    x_norm = PowerNormalize(x)
    x_output = Awgn(x_norm, noise_std)
    return x_output

# 以实际信号功率计算噪声功率，再将信号加上噪声。
def awgn(x, snr):
    SNR = 10.0**(snr/10.0)
    signal_power = ((x**2)*1.0).mean()
    noise_power = signal_power/SNR
    noise_std = torch.sqrt(noise_power)
    #print(f"x.shape = {x.shape}, signal_power = {signal_power}, noise_power={noise_power}, noise_std={noise_std}")

    noise = torch.normal(mean=0, std = float(noise_std), size=x.shape)
    return x+noise




#============================================================================================================================

















