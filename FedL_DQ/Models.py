#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Created on:  2024/08/24

@author: Junjie Chen

# nn.CrossEntropyLoss() 接受的输入是 logits，这说明分类的输出不需要提前经过 log_softmax. 如果提前经过 log_softmax, 则需要使用 nn.NLLLoss()（负对数似然损失）。
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

#%%%%%%%%%%%%%%%%%%%%% MNIST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
# a = np.sum([p.numel() for  p in model.state_dict().values()])  # 全部参数
# b = np.sum([p.numel() for p in model.parameters() if p.requires_grad]) # 全部可导参数
# print(f"a = {a}, b = {b}")
# a = 21921, b = 21880

## 全部参数
# for key, var in model.state_dict().items():
#     print(f"{key}, {var.is_leaf}, {var.shape},  " )
## 全部可导参数
# for name, param in  model.named_parameters():
#     print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ")

#%%%%%%%%%%%%%%%%%%%%% CIFAR10 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


class CNNCifar1(nn.Module):
     def __init__(self, input_channels = 3, output_channels = 10, ):
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

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes) )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])





























