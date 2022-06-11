#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch,sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)



# 方法一：类继承nn.Module，必须实现forward函数
class ResBlock1(nn.Module):
     def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock1, self).__init__()
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


#方法2：类继承nn.Sequential,但是这么做失去部分灵活性
class ResBlock2(nn.Sequential):
     def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock2, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        
     # def forward(self, x):
     #       res = self.body(x).mul(self.res_scale)
     #       res += x
     #       return res


conv = default_conv
input1 = torch.randn(16, 64, 20, 20)


resb = ResBlock1(conv, 64, 5)
a1 = resb(input1)





resb = ResBlock2(conv, 64, 5)
a2 = resb(input1)































