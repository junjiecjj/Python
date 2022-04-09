#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 21:28:35 2022

@author: jack

https://pytorchbook.cn/chapter2/2.1.4-pytorch-basics-data-loader/


"""

# 首先要引入相关的包
import torch

#引用
from torch.utils.data import Dataset
import pandas as pd
import pretty_errors

#打印一下版本
print(f"torch.__version__ = {torch.__version__}")



#定义一个数据集
class BulldozerDataset(Dataset):
    """ 数据集演示 """
    def __init__(self, csv_file):
        """实现初始化方法，在初始化的时候将数据读载入"""
        self.df=pd.read_csv(csv_file)
    def __len__(self):
        '''
        返回df的长度
        '''
        return len(self.df)
    def __getitem__(self, idx):
        '''
        根据 idx 返回一行数据
        '''
        return self.df.iloc[idx].SalePrice
    def __getitem1__(self, idx):
        '''
        根据 idx 返回一行数据
        '''
        return self.df.iloc[idx]


ds_demo= BulldozerDataset('median_benchmark.csv')

print(f"len(ds_demo) = {len(ds_demo)}")


print(f" ds_demo[2] = {ds_demo[2]}\n")

print(f" ds_demo__getitem1__(2) = {ds_demo.__getitem1__(2)}")







#Dataloader
#DataLoader为我们提供了对Dataset的读取操作，常用参数有：batch_size(每个batch的大小)、 shuffle(是否进行shuffle操作)、 num_workers(加载数据的时候使用几个子进程)。下面做一个简单的操作
print("-"*60)
dl = torch.utils.data.DataLoader(ds_demo, batch_size=10, shuffle=True, num_workers=0)


idata=iter(dl)
print(next(idata))


#常见的用法是使用for循环对其进行遍历
for i, data in enumerate(dl):
    print(f"i = {i}, data.shape = {data.shape} \n data = {data}\n")
    # 为了节约空间，这里只循环一遍
    #break





#torchvision 包
#torchvision 是PyTorch中专门用来处理图像的库，PyTorch官网的安装教程中最后的pip install torchvision 就是安装这个包。

#torchvision.datasets
#torchvision.datasets 可以理解为PyTorch团队自定义的dataset，这些dataset帮我们提前处理好了很多的图片数据集，我们拿来就可以直接使用： - MNIST - COCO - Captions - Detection - LSUN - ImageFolder - Imagenet-12 - CIFAR - STL10 - SVHN - PhotoTour 我们可以直接使用，示例如下：


#FashionMNIST
import torch, torchvision,sys


root='../../MLData/FashionMNIST'
batch_size = 128
trans = [] 
resize = None

if resize:
    trans.append(torchvision.transforms.Resize(size=resize))
trans.append(torchvision.transforms.ToTensor())
transform = torchvision.transforms.Compose(trans)
    
trainset = torchvision.datasets.FashionMNIST(root=root, # 表示 MNIST 数据的加载的目录
                                      train=True,  # 表示是否加载数据库的训练集，false的时候加载测试集
                                      download=True, # 表示是否自动下载 MNIST 数据集
                                      transform=transform) # 表示是否需要对数据进行预处理，none为不进行预处理


testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


for X, y in train_iter:
     print(f"X.shape = {X.shape}, y.shape = {y.shape}")
     
     
     
for X, y in test_iter:
     print(f"X.shape = {X.shape}, y.shape = {y.shape}")     


"""
X.shape = torch.Size([128, 1, 28, 28]), y.shape = torch.Size([128])
X.shape = torch.Size([128, 1, 28, 28]), y.shape = torch.Size([128])
X.shape = torch.Size([16, 1, 28, 28]), y.shape = torch.Size([16])  #last
"""






#MNIST
import torch, torchvision,sys


root='../../MLData/MNIST'
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
     
     
     
for X, y in test_iter1:
     print(f"X.shape = {X.shape}, y.shape = {y.shape}")     


"""
X.shape = torch.Size([128, 1, 28, 28]), y.shape = torch.Size([128])
X.shape = torch.Size([128, 1, 28, 28]), y.shape = torch.Size([128])
X.shape = torch.Size([16, 1, 28, 28]), y.shape = torch.Size([16])  #last
"""


#MNIST
import torch, torchvision,sys


root='../../MLData/MNIST'
batch_size = 128
trans = [] 
resize = 224

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
     
     
     
for X, y in test_iter1:
     print(f"X.shape = {X.shape}, y.shape = {y.shape}")     


"""
X.shape = torch.Size([128, 1, 224, 224]), y.shape = torch.Size([128])
X.shape = torch.Size([128, 1, 224, 224]), y.shape = torch.Size([128])
X.shape = torch.Size([128, 1, 224, 224]), y.shape = torch.Size([128])
X.shape = torch.Size([128, 1, 224, 224]), y.shape = torch.Size([128])
X.shape = torch.Size([128, 1, 224, 224]), y.shape = torch.Size([128])
X.shape = torch.Size([16, 1, 224, 224]), y.shape = torch.Size([16])   #last
"""





#CIFAR10
import torch, torchvision,sys


root='../../MLData/CIFAR10'
batch_size = 128
trans = [] 
resize = None

if resize:
    trans.append(torchvision.transforms.Resize(size=resize))
trans.append(torchvision.transforms.ToTensor())
transform = torchvision.transforms.Compose(trans)
    
trainset1 = torchvision.datasets.CIFAR10(root=root, # 表示 MNIST 数据的加载的目录
                                      train=True,  # 表示是否加载数据库的训练集，false的时候加载测试集
                                      download=True, # 表示是否自动下载 MNIST 数据集
                                      target_transform=None,
                                      transform=transform) # 表示是否需要对数据进行预处理，none为不进行预处理


testset1 = torchvision.datasets.CIFAR10(root=root, train=False, download=True,target_transform=None, transform=transform)
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
train_iter1 = torch.utils.data.DataLoader(trainset1, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter1 = torch.utils.data.DataLoader(testset1, batch_size=batch_size, shuffle=False, num_workers=num_workers)


for X, y in train_iter1:
     print(f"X.shape = {X.shape}, y.shape = {y.shape}")
     
     
     
for X, y in test_iter1:
     print(f"X.shape = {X.shape}, y.shape = {y.shape}")     


"""
X.shape = torch.Size([128, 3, 32, 32]), y.shape = torch.Size([128])
X.shape = torch.Size([128, 3, 32, 32]), y.shape = torch.Size([128])
X.shape = torch.Size([128, 3, 32, 32]), y.shape = torch.Size([128])
X.shape = torch.Size([128, 3, 32, 32]), y.shape = torch.Size([128])
X.shape = torch.Size([128, 3, 32, 32]), y.shape = torch.Size([128])
X.shape = torch.Size([128, 3, 32, 32]), y.shape = torch.Size([128])
X.shape = torch.Size([16, 3, 32, 32]), y.shape = torch.Size([16])  #last
"""

































































































































































































































































































































































































































































































