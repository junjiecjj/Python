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
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 10)

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

# for key, var in net.state_dict().items():
#     print(f"{key}, {var.is_leaf}, {var.shape},  " )

# param_W = net.state_dict()

### torch
# params_float = torch.Tensor([], )
# for key, val in param_W.items():
#     params_float = torch.cat((params_float, val.detach().cpu().flatten()))
# std = params_float.std()
# var = params_float.var()
# mean = params_float.mean()


class CNNMnist(nn.Module):
    def __init__(self, num_channels, num_classes,batch_norm=False):
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

# # ## Data volume = 21840 (floating point number)
# net = CNNMnist(1, 10)
# data_valum = np.sum([param.numel() for param in net.state_dict().values()])
# print(f"Data volume = {data_valum} (floating point number) ")













