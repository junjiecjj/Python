#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:16:20 2022

@author: jack
"""


#  https://blog.csdn.net/disanda/article/details/105762054


import torch
import torch.nn as nn

x = torch.randn(1,1,2,2)
l = nn.ConvTranspose2d(1,1,3)#Conv2d(1, 1, kernel_size=3,stride=1,padding=0)
y = l(x) # y.shape:[1,1,4,4]
print(f"y.shape = {y.shape}\n")




import torch
import torch.nn as nn

x = torch.randn(1,1,6,6)
l = nn.ConvTranspose2d(1,1,4,padding=2)#Conv2d(1, 1, kernel_size=4,stride=1,padding=2)
y = l(x) # y.shape:[1,1,5,5]
print(f"y.shape = {y.shape}\n")




import torch
import torch.nn as nn

x = torch.randn(1,1,7,7)
l = nn.ConvTranspose2d(1,1,3,padding=2)#Conv2d(1, 1, kernel_size=3,stride=1,padding=2)
y = l(x) # y.shape:[1,1,5,5]
print(f"y.shape = {y.shape}\n")



import torch
import torch.nn as nn

x = torch.randn(1,1,2,2)
l = nn.ConvTranspose2d(1,1,3,stride=2,padding=0)#Conv2d(1, 1, kernel_size=3,stride=2,padding=0)
y = l(x) # y.shape:[1,1,5,5]
print(f"y.shape = {y.shape}\n")


import torch
import torch.nn as nn

x = torch.randn(1,1,3,3)
l = nn.ConvTranspose2d(1,1,3,stride=2,padding=1)#Conv2d(1, 1, kernel_size=3,stride=2,padding=1)
y = l(x) # y.shape:[1,1,5,5]
print(f"y.shape = {y.shape}\n")



# 根据压缩比和输出输入的图像大小计算压缩层的输出通道数。
def calculate_channel(comp_ratio, F=5, n=3072):
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


x = torch.randn(1,64,48,48)
print(f"x.shape = {x.shape}\n")
n_feats=64
img_dim=48

H_hat = int(1+int( img_dim+2*2-12)//4)
n = 3*img_dim**2
        
C =  conv2d_prelu(64, calculate_channel(0.17,F=H_hat, n=n), 12, 4, 2) 

x1 = C(x)
print(f"x1.shape = {x1.shape}\n")



D =nn.ConvTranspose2d(
            9,
            64,
            kernel_size=10,
            stride=4,
            padding=1,
        )


x2 = D(x1)
print(f"x2.shape = {x2.shape}\n")
