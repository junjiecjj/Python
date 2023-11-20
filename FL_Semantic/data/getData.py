# -*- coding: utf-8 -*-
"""
Created on 2023/06/30

@author: Junjie Chen

"""


import numpy as np

import os
import torchvision
from torchvision import transforms as transforms
import torch


class GetDataSet(object):
    def __init__(self, dataSetName, isIID = False, data_root = "/home/jack/公共的/MLData/"):
        self.data_root       = data_root
        self.name            = dataSetName
        self.train_data      = None  # 训练集
        self.train_label     = None  # 标签
        self.train_data_size = None  # 训练数据的大小

        self.test_data       = None  # 测试数据集
        self.test_label      = None  # 测试的标签
        self.test_data_size  = None   # 测试集数据Size

        if self.name.lower() == 'mnist':
            self.load_MNIST_torch(isIID)
        elif self.name.lower() == 'cifar10':
            self.load_cifar10(isIID)
        else:
            pass
        return


    def load_MNIST_torch(self, isIID = False):
        train_tf = torchvision.transforms.Compose([transforms.ToTensor()])
        test_tf  = torchvision.transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(root = self.data_root, train = True,  download = True, transform = train_tf)
        test_set  = torchvision.datasets.MNIST(root = self.data_root, train = False, download = True, transform = test_tf)

        ## 训练数据Size
        self.train_data_size = train_set.data.shape[0]  # 60000
        self.test_data_size  = test_set.data.shape[0]   # 10000

        ## 训练集
        train_data   =  train_set.data                # 训练数据 torch.Size([60000, 28, 28]), 0-255
        train_labels =  train_set.targets             # (60000)
        # 将训练集转化为（60000，28*28）矩阵
        train_images = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])

        ## 测试集
        test_data    =  test_set.data         # 测试数据 torch.Size([10000, 28, 28]), 0-255
        test_labels  =  test_set.targets    # 10000
        # 将测试集转化为（10000，28*28）矩阵
        test_images  = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])

        ## ---------------------------归一化处理------------------------------
        train_images = train_images.type(torch.float32)
        train_images = torch.mul(train_images, 1.0 / 255.0)

        test_images = test_images.type(torch.float32)
        test_images = torch.mul(test_images, 1.0 / 255.0)
        ## -------------------------------------------------------------------------

        if isIID:
            ## 这里将 60000 个训练数据随机打乱
            order = np.random.permutation(self.train_data_size)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            ## 对数据标签进行排序
            order = np.argsort(train_labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels
        return

    # 加载cifar10 的数据
    def load_cifar10(self, isIID = False):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root = "/home/jack/公共的/MLData/CIFAR10", train = True, download = True, transform = train_transform)
        test_set = torchvision.datasets.CIFAR10(root = "/home/jack/公共的/MLData/CIFAR10", train = False, download = True, transform = test_transform)

        train_data = train_set.data  # (50000, 32, 32, 3)
        train_labels = train_set.targets
        train_labels = np.array(train_labels)  # 将标签转化为

        test_data = test_set.data  # 测试数据
        test_labels = test_set.targets
        test_labels = np.array(test_labels)

        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]

        # 将训练集转化为（50000，32*32*3）矩阵
        train_images = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2] * train_data.shape[3])
        # 将测试集转化为（10000，32*32*3）矩阵
        test_images = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2] * test_data.shape[3])

        ## ---------------------------归一化处理------------------------------#
        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        ## -------------------------------------------------------------------------#

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            order = np.argsort(train_labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels
        return



# mnistDataSet = GetDataSet('mnist', 0) # test NON-IID
# print(f"{type(mnistDataSet.train_data)} | {type(mnistDataSet.test_data)} | {type(mnistDataSet.train_label)} | {type(mnistDataSet.test_label)}")
# # <class 'torch.Tensor'> | <class 'torch.Tensor'> | <class 'torch.Tensor'> | <class 'torch.Tensor'>

# print(f"{mnistDataSet.train_data.shape}, {mnistDataSet.train_label.shape}, {mnistDataSet.test_data.shape}, {mnistDataSet.test_label.shape}")
# # torch.Size([60000, 784]), torch.Size([60000]), torch.Size([10000, 784]), torch.Size([10000])


# print(f"{mnistDataSet.train_data[0].max()}, {mnistDataSet.train_data[0].min()}")
# # 1.0, 0.0


# if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and  type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
#     print('the type of data is numpy ndarray')
# else:
#     print('the type of data is not numpy ndarray')

# print(mnistDataSet.train_label, mnistDataSet.test_label )
# # tensor([0, 0, 0,  ..., 9, 9, 9]) tensor([7, 2, 1,  ..., 4, 5, 6])













