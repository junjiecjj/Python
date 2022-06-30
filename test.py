#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://www.jianshu.com/p/12a8207149b0

import torch
from torch.nn import functional as f

batchszie = 1
C = 3
H = 5
W = 5

x = torch.arange(0, batchszie*C*H*W).float()
x = x.view(batchszie,C,H,W)

# print(f"x = \n{x}")

print(f"x.shape = {x.shape}")

kernelsize = 2
stride = 1
x1 = f.unfold(x, kernel_size=kernelsize, dilation=1, stride=stride)
print(f"x1.shape = {x1.shape}")

x2 = x1.transpose(0,2).contiguous()
print(f"x2.shape = {x2.shape}")


x3 = x2.view(x2.size(0),-1,kernelsize,kernelsize)
print(f"x3.shape = {x3.shape}")


B, C_kh_kw, L = x1.size()
x4 = x1.permute(0, 2, 1)
print(f"x4.shape = {x4.shape}")

x5 = x4.view(B, L, -1, kernelsize, kernelsize)
print(f"x5.shape = {x5.shape}")



