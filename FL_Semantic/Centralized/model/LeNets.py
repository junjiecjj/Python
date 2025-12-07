#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 20:49:23 2022

@author: jack

https://lulaoshi.info/machine-learning/convolutional/lenet


"""



# import pandas as pd
# import numpy as np
# import torch, torchvision
from torch import nn
import torch.nn.functional as F
# from torch.autograd import Variable
import os , sys
# from torch.utils.tensorboard import SummaryWriter
# import torch.optim.lr_scheduler as lrs
# import collections
# import matplotlib.pyplot as plt

# import argparse


# sys.path.append("..")
# from model  import common


#==============================================================================================
#                            定义LeNet模型
#==============================================================================================

class LeNet_1(nn.Module):
    def __init__(self, ):
        super(LeNet_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# x = torch.randn(size = (1, 1, 28, 28))

# mo = LeNet_1()
# # mo.train()
# y = mo(x)

class LeNet_2(nn.Module):
    def __init__(self, ):
        super(LeNet_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.maxpool2D1 = nn.MaxPool2d(kernel_size = 2, )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.maxpool2D2 = nn.MaxPool2d(kernel_size = 2, )
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.droup = nn.Dropout( )
        self.fc2 = nn.Linear(50, 10)
        self.logsoft = nn.LogSoftmax(dim = 1)


    def forward(self, x):
        # print(f"0: x.shape = {x.shape}") # 0: x.shape = torch.Size([1, 1, 28, 28])
        x = self.conv1(x)
        # print(f"1: x.shape = {x.shape}") # 1: x.shape = torch.Size([1, 10, 24, 24])
        x = self.maxpool2D1(x)
        # print(f"2: x.shape = {x.shape}") # 2: x.shape = torch.Size([1, 10, 12, 12])
        x = self.relu1(x)
        # print(f"3: x.shape = {x.shape}") # 3: x.shape = torch.Size([1, 10, 12, 12])
        x = self.conv2(x)
        # print(f"4: x.shape = {x.shape}") #  4: x.shape = torch.Size([1, 20, 8, 8])
        x = self.conv2_drop(x)
        # print(f"5: x.shape = {x.shape}") # 5: x.shape = torch.Size([1, 20, 8, 8])
        x = self.maxpool2D2(x)
        # print(f"6: x.shape = {x.shape}") # 6: x.shape = torch.Size([1, 20, 4, 4])
        x = self.relu2(x)
        # print(f"7: x.shape = {x.shape}")  # 7: x.shape = torch.Size([1, 20, 4, 4])
        x = x.view(1, -1)
        # print(f"mid: x.shape = {x.shape}") # mid: x.shape = torch.Size([1, 320])
        x = self.fc1(x)
        # print(f"8: x.shape = {x.shape}") # 8: x.shape = torch.Size([1, 50])
        x = self.relu3(x)
        # print(f"9: x.shape = {x.shape}") # 9: x.shape = torch.Size([1, 50])
        x = self.droup(x)
        # print(f"10: x.shape = {x.shape}") # 10: x.shape = torch.Size([1, 50])
        x = self.fc2(x)
        # print(f"11: x.shape = {x.shape}") # 11: x.shape = torch.Size([1, 10])
        x = self.logsoft(x)
        # print(f"12: x.shape = {x.shape}") # 12: x.shape = torch.Size([1, 10])

        return  x

# mo1 = LeNet_2()
# # mo.train()
# y1 = mo1(x)


class LeNet_3(nn.Module):
    def __init__(self, ):
        super(LeNet_3, self).__init__()
        # input shape: 1 * 1 * 28 * 28
        self.conv = nn.Sequential(
            ## conv layer 1
            ## conv: 1, 28, 28 -> 10, 24, 24
            nn.Conv2d(1, 10, kernel_size = 5),
            ## 10, 24, 24 -> 10, 12, 12
            nn.MaxPool2d(kernel_size = 2, ),
            nn.ReLU(),

            ## 10, 12, 12 -> 20, 8, 8
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            ## 20, 8, 8 -> 20, 4, 4
            nn.MaxPool2d(kernel_size = 2, ),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            ## full connect layer 1
            nn.Linear(320, 50), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.LogSoftmax(dim = 1)
        )

    def forward(self, img):
        feature = self.conv(img)

        # aver = torch.mean(feature, )
        # print(f"3 aver = {aver}")
        # snr = 0.2
        # aver_noise = aver * (1 / 10 **(snr/10))
        # # print(f"{aver}, {aver_noise}")
        # noise = torch.randn(size = feature.shape) * aver_noise.to('cpu')
        # feature = feature + noise.to(feature.device)

        output = self.fc(feature.view(img.shape[0], -1))
        return output

# x = torch.randn(size = (1, 1, 28, 28))
# mo2 = LeNet_3()
# # mo.train()
# y2 = mo2(x)


#==============================================================================================
#                            定义LeNet模型
#==============================================================================================

#  nn.Conv2d的输入必须为4维的(N,C,H,W)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # input shape: 1 * 28 * 28
        self.conv = nn.Sequential(
            # conv layer 1
            # add padding: 28 * 28 -> 32 * 32
            # conv: 1 * 32 * 32 -> 6 * 28 * 28
            nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=5, padding=2), nn.Sigmoid(),
            # 6 * 28 * 28 -> 6 * 14 * 14
            nn.MaxPool2d(kernel_size=2, stride=2), # kernel_size, stride
            # conv layer 2
            # 6 * 14 * 14 -> 16 * 10 * 10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.Sigmoid(),
            # 16 * 10 * 10 -> 16 * 5 * 5
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            # full connect layer 1
            nn.Linear(in_features=16 * 5 * 5, out_features=120), nn.Sigmoid(),
            # full connect layer 2
            nn.Linear(in_features=120, out_features=84), nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        """
        img.shape = torch.Size([256, 1, 28, 28])
        feature.shape = torch.Size([256, 16, 5, 5])
        output.shape = torch.Size([256, 10])
        feature.view(img.shape[0], -1).shape = torch.Size([256, 400])
        """
        return output












































































































































































































































































































































































































































































































































































