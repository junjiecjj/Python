#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Created on:  2024/08/24

@author: Junjie Chen

# nn.CrossEntropyLoss() 接受的输入是 logits，这说明分类的输出不需要提前经过 log_softmax. 如果提前经过 log_softmax, 则需要使用 nn.NLLLoss()（负对数似然损失）。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys,os

import matplotlib
# matplotlib.use('TkAgg')
import torch.optim as optim

class Mnist_1MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, inputs):
        inputs = inputs.view(-1, 28*28)
        tensor = self.fc1(inputs)
        return tensor
# ### Data volume = 7850 (floating point number)
# net = Mnist_1MLP()
# data_valum1 = np.sum([param.numel() for param in net.state_dict().values()])
# print(f"Data volume = {data_valum1} (floating point number) ")

class Mnist_2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, inputs):
        inputs = inputs.view(-1, 28*28)
        tensor = F.relu(self.fc1(inputs))
        tensor = self.fc2(tensor)
        return tensor
# ### Data volume = 39760 (floating point number)
# net = Mnist_2MLP()
# data_valum1 = np.sum([param.numel() for param in net.state_dict().values()])
# print(f"Data volume = {data_valum1} (floating point number) ")

# for key, var in net.state_dict().items():
#     print(f"{key}, {var.is_leaf}, {var.shape},  " )


class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, inputs):
        inputs = inputs.view(-1, 28*28)
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor
# ### Data volume = 178110 (floating point number)
# net = Mnist_2NN()
# data_valum = np.sum([param.numel() for param in net.state_dict().values()])
# print(f"Data volume = {data_valum} (floating point number) ")

# for key, var in net.state_dict().items():
#     print(f"{key}, {var.is_leaf}, {var.shape},  " )

class Mnist_CNN(nn.Module):
    def __init__(self, input_channels = 1, output_channels = 10):
        super(Mnist_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, output_channels)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) # [, 10, 12, 12]
        x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2) # [, 20, 4, 4]
        x = x.contiguous().view(-1, 320) # [, 320]
        x = F.relu(self.fc1(x)) # [, 50]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x) # [, 10]
        return x

# ## Data volume = 21840 (floating point number)
# net = Mnist_CNN()
# data_valum = np.sum([param.numel() for param in net.state_dict().values()])
# print(f"Data volume = {data_valum} (floating point number) ")



class CNNMnist(nn.Module):
    def __init__(self, num_channels, num_classes, batch_norm=False):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        if batch_norm:
            self.conv2_norm=nn.BatchNorm2d(20)
        else:
            self.conv2_norm = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_norm(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# # # ## Data volume = 21840 (floating point number)
# model = CNNMnist(1, 10, batch_norm = True)
# data_valum = np.sum([param.numel() for param in model.state_dict().values()])
# print(f"Data volume = {data_valum} (floating point number) ")

class CNNCifar(nn.Module):
    def __init__(self, input_channels = 3, output_channels = 10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_channels)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# # # # ## Data volume = 61706 (floating point number)
# model = CNNCifar(1, 10, )
# data_valum = np.sum([param.numel() for param in model.state_dict().values()])
# print(f"Data volume = {data_valum} (floating point number) ")


class CNNCifar1(nn.Module):
     def __init__(self, input_channels = 3, output_channels = 10):
         super(CNNCifar1, self).__init__()
         self.conv1 = nn.Conv2d(3, 6, 5)
         self.conv2 = nn.Conv2d(6, 16, 5)
         self.fc1 = nn.Linear(16*5*5, 120)
         self.fc2 = nn.Linear(120, 84)
         self.fc3 = nn.Linear(84, 10)
     def forward(self,x):
         x = F.max_pool2d(F.relu(self.conv1(x)),(2, 2))
         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
         x = x.view(x.size()[0],-1)
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
         x = self.fc3(x)
         return  x

# # # # ## Data volume = 62006 (floating point number)
# model = CNNCifar1( )
# data_valum = np.sum([param.numel() for param in model.state_dict().values()])
# print(f"Data volume = {data_valum} (floating point number) ")
class Block(nn.Module):
    def __init__(self, inchannel, outchannel, res=True, stride=2):
        super(Block, self).__init__()
        self.res = res     # 是否带残差连接
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),
                nn.BatchNorm2d(outchannel),
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out


class myModel(nn.Module):
    def __init__(self, cfg=[64, 'M', 128,  'M', 256, 'M', 512, 'M'], res=True):
        super(myModel, self).__init__()
        self.res = res       # 是否带残差连接
        self.cfg = cfg       # 配置列表
        self.inchannel = 3   # 初始输入通道数
        self.futures = self.make_layer()
        # 构建卷积层之后的全连接层以及分类器：
        self.classifier = nn.Sequential(nn.Dropout(0.4),            # 两层fc效果还差一些
                                        nn.Linear(4 * 512, 10), )   # fc，最终Cifar10输出是10类

    def make_layer(self):
        layers = []
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Block(self.inchannel, v, self.res))
                self.inchannel = v    # 输入通道数改为上一层的输出通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.futures(x)
        # view(out.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
# model = myModel( )
# data_valum = np.sum([param.numel() for param in model.state_dict().values()])
# print(f"Data volume = {data_valum} (floating point number) ")
# 4887702










