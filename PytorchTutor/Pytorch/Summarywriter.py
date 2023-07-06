#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:18:48 2022

@author: jack
"""


# ==================================================================================
# https://blog.csdn.net/u013602059/article/details/107480120
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/home/jack/公共的/Python/PytorchTutor/Pytorch/Summary')
x = range(100)
for i in x:
    writer.add_scalar('y=2x', i * 2, i)
writer.close()


#  tensorboard --logdir=/home/jack/公共的/Python/PytorchTutor/Pytorch/Summary
#  http://0.0.0.0:6006/

# ==================================================================================
#   https://mp.weixin.qq.com/s/UYnBRU2b0InzM9H1xl4b4g
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default
writer = SummaryWriter('/home/jack/公共的/Python/PytorchTutor/Pytorch/model')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(trainloader))

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()

#会发现刚刚的log文件夹里面有文件了。在命令行输入如下，载入刚刚做图的文件（那个./log要写完整的路径）
#  tensorboard --logdir=/home/jack/公共的/Python/PytorchTutor/Pytorch/model
# 在浏览器输入以下任意一个网址，即可查看结果：（训练过程中可以实时更新显示）
#  http://0.0.0.0:6006/
#  http://localhost:6006/
#  http://127.0.0.1:6006/

































































































































































































