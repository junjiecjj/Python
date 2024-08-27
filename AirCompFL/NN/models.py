#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Created on:  2024/08/24

@author: Junjie Chen

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class Mnist_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, inputs):
        inputs = inputs.view(-1, 28*28)
        tensor = self.fc1(inputs)
        return tensor

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

# nn.CrossEntropyLoss() 接受的输入是 logits，这说明分类的输出不需要提前经过 log_softmax. 如果提前经过 log_softmax, 则需要使用 nn.NLLLoss()（负对数似然损失）。

# # Data volume = 178110 (floating point number)
# net = Mnist_2NN()
# data_valum = 0
# for key, var in net.state_dict().items():
#     data_valum += var.numel()
# print(f"Data volume = {data_valum} (floating point number) ")



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

# # Data volume = 21840 (floating point number)
# net = Mnist_CNN()
# data_valum = 0
# for key, var in net.state_dict().items():
#     data_valum += var.numel()
# print(f"Data volume = {data_valum} (floating point number) ")


class Mnist_CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

# Data volume = 1663370 (floating point number)
# net = Mnist_CNN()
# data_valum = 0
# for key, var in net.state_dict().items():
#     data_valum += var.numel()
# print(f"Data volume = {data_valum} (floating point number) ")











