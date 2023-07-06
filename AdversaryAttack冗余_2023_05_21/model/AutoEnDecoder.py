#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:18:44 2023

@author: jack

https://zhuanlan.zhihu.com/p/116769890

https://blog.csdn.net/weixin_38739735/article/details/119013420


https://zhuanlan.zhihu.com/p/137571225


https://zhuanlan.zhihu.com/p/625085766


https://blog.csdn.net/Cy_coding/article/details/113840883

https://blog.csdn.net/winycg/article/details/90318371

https://www.bilibili.com/read/cv12946597

https://zhuanlan.zhihu.com/p/133207206

https://zhuanlan.zhihu.com/p/80377698

https://zhuanlan.zhihu.com/p/628604566
"""

import pandas as pd
import numpy as np
import torch, torchvision 
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import os , sys
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lrs
import collections
import matplotlib.pyplot as plt

import argparse


sys.path.append("../")
# from model  import common



# https://zhuanlan.zhihu.com/p/116769890
class AutoEncoderMnist(nn.Module):
    def __init__(self):
        super(AutoEncoderMnist,self).__init__()
        self.encoder  =  nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3,16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()

        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return  encoded, decoded







# https://blog.csdn.net/weixin_38739735/article/details/119013420
class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        ### Convolutional p
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear p
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        print(f"1 x.shape = {x.shape}")
        # torch.Size([25, 1, 28, 28])
        x = self.encoder_cnn(x)
        print(f"2 x.shape = {x.shape}")
        # torch.Size([25, 32, 3, 3])
        x = self.flatten(x)
        print(f"3 x.shape = {x.shape}")
        # torch.Size([25, 288])
        x = self.encoder_lin(x)
        print(f"4 x.shape = {x.shape}")
        # torch.Size([25, 4])
        return x


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
 
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )

    def forward(self, x):
        print(f"1 x.shape = {x.shape}")
        # 1 torch.Size([25, 4])

        x = self.decoder_lin(x)
        print(f"2 x.shape = {x.shape}")
        # 2 x.shape = torch.Size([25, 288])

        x = self.unflatten(x)
        print(f"3 x.shape = {x.shape}")
        # 3 x.shape = torch.Size([25, 32, 3, 3])

        x = self.decoder_conv(x)
        print(f"4 x.shape = {x.shape}")
        # 4 x.shape = torch.Size([25, 1, 28, 28])

        x = torch.sigmoid(x)
        print(f"5 x.shape = {x.shape}")
        # 5 x.shape = torch.Size([25, 1, 28, 28])

        return x









































































































































































































































































































































































































































