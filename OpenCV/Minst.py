#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:10:03 2023

@author: jack


"""

import os
from skimage import io
import torchvision.datasets.mnist as mnist

import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision,sys


root='/home/jack/公共的/MLData/MNIST'
batch_size = 128
trans = [] 
resize = None

if resize:
    trans.append(torchvision.transforms.Resize(size=resize))
trans.append(torchvision.transforms.ToTensor())
transform = torchvision.transforms.Compose(trans)

trainset1 = torchvision.datasets.MNIST(root=root, # 表示 MNIST 数据的加载的目录
                                      train=True,  # 表示是否加载数据库的训练集，false的时候加载测试集
                                      download=True, # 表示是否自动下载 MNIST 数据集
                                      transform=transform) # 表示是否需要对数据进行预处理，none为不进行预处理


testset1 = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
train_iter1 = torch.utils.data.DataLoader(trainset1, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter1 = torch.utils.data.DataLoader(testset1, batch_size=batch_size, shuffle=False, num_workers=num_workers)


for X, y in train_iter1:
     print(f"X.shape = {X.shape}, y.shape = {y.shape}")
     
     

figsize=(10,10)
dim=(5,5)




plt.figure(figsize=figsize)
for i in range(25):
    plt.subplot(dim[0], dim[1], i+1)
    plt.imshow(X[i,0], interpolation='none', cmap='Greys')
    plt.title("Ground Truth: {}".format(y[i])) 
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.savefig('Origin.png')