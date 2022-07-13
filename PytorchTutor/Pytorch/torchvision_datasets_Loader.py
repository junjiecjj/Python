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


import pandas as pd
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
#DataLoader为我们提供了对Dataset的读取操作，常用参数有：batch_size(每个batch的大小)、 shuffle(是否进行shuffle操作)、 
# num_workers(加载数据的时候使用几个子进程)。下面做一个简单的操作
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
#torchvision.datasets 可以理解为PyTorch团队自定义的dataset，这些dataset帮我们提前处理好了很多的图片数据集，我们拿来就可以直接使用：
# - MNIST - COCO - Captions - Detection - LSUN - ImageFolder - Imagenet-12 - CIFAR - STL10 - SVHN - PhotoTour 
# 我们可以直接使用，示例如下：


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
#我们使用CIFAR10数据集，CIFAR10由 10 个类别的 60000 张 32x32 彩色图像组成，每类 6000 张图像。这些类是：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。
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










#MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
# 导入数据
train_dataset = datasets.MNIST(root = '~/公共的/MLData/MNIST', train = True,
                               transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root = 'data/', train = False, 
                               transform = transforms.ToTensor(), download = True)
# 加载数据，打乱顺序并一次给模型100个数据
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
images, labels = next(iter(train_loader))
# 使用images生成宽度为10张图的网格大小
img = torchvision.utils.make_grid(images, nrow=10)
# cv2.imshow()的格式是(size1,size1,channels),而img的格式是(channels,size1,size1),
# 所以需要使用.transpose()转换，将颜色通道数放至第三维
img = img.numpy().transpose(1,2,0)
print(images.shape)
print(labels.reshape(10,10))
print(img.shape)
cv2.imshow('img', img)
cv2.waitKey()



#pytorch 图像分类数据集（Fashion-MNIST）
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")  #导入d2lzh_pytorch
#import d2lzh_pytorch as d2l   #导入所需要的包和模块

mnist_train =torchvision.datasets.FashionMNIST(root='~/公共的/MLData/FashionMNIST',train=True, download=True, transform=transforms.ToTensor())
#用torchvision的torchvision.datasets来下载数据集 通过参数train来指定训练数据集或测试数据集                    
#用transform=transform.ToTensor（）将所有数据转换为Tensor (不进行转换 换回的为PIL图片)
mnist_test =torchvision.datasets.FashionMNIST(root='~/公共的/MLData/FashionMNIST',train=False, download=True, transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test)) #获取数据集的大小


print(f"mnist_train[0][0].shape = {mnist_train[0][0].shape},\n mnist_train[0][1] = {mnist_train[0][1]}")
# mnist_train[0][0].shape = torch.Size([1, 28, 28]),
# mnist_train[0][1] = 9

feature, label = mnist_train[0]  #通过下标来访问任意一个样本
print(feature.shape, label)  # Channel x Height X Width



def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress','coat','sandal', 'shirt', 'sneaker', 'bag', 'ankleboot']
    return [text_labels[int(i)] for i in labels]
#将数值标签转换为相应的文本标签


#定义可以在一行里画出多张图像和对应标签
def show_fashion_mnist(images, labels):
    #d2l.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))  # len(X) = 5
    for f, img, lbl in zip(figs, images, labels):
         print(f"img.shape = {img.shape}, lal = {lbl}")
         f.imshow(img.view((28, 28)).numpy())
         f.set_title(lbl)
         f.axes.get_xaxis().set_visible(False)
         f.axes.get_yaxis().set_visible(False)
    plt.show()

X, y = [], []
for i in range(5):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1]) # [9, 0, 0, 3, 0]
show_fashion_mnist(X, get_fashion_mnist_labels(y))  #['ankleboot', 't-shirt', 't-shirt', 'dress', 't-shirt']



batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0 #0表示不用额外的进程来加速读取数据
else:
    num_workers = 4 #设置4个进程读取数据
train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=False, num_workers=num_workers)
#PyTorch的DataLoader中?个很?便的功能是允许使?多进程来加速数据读取

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))  #查看读取?遍训练数据需要的时间





"""
Imagenet一共有三个文件夹，分别是train、val、test，train里边有1000个文件夹分别代表1000个类,
每个类下边有1300张对应类别的图片，val里边有50000张验证集图片，用ImageFolder进行数据加载的时候，训练集直接按上述办法加载就好，而验证集我们需要对他进行处理，
使用如下脚本把验证集50000张图片分别划分到相应类的文件夹中，再进行加载就可以了。
"""


# ImageNet

import torch
import torchvision
import torchvision.transforms as transforms

import   sys


root='../../MLData/ImageNet'
batch_size = 128
num_workers = 4


data_transform = transforms.Compose([
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
     ])


trainData = torchvision.datasets.ImageFolder(root = root,transform=data_transform )
trainData_iter = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=num_workers)


for X, y in trainData_iter:
     print(f"X.shape = {X.shape}, y.shape = {y.shape}")
     
     

#=====================================================================================
import torch
import torchvision
data_dirs = {"train":"~/公共的/MLData", "val":"~/公共的/MLData"}

phases = ["train", "val"]
data_sets = {}
data_loaders = {}

for p in phases:
	data_sets[p] =  torchvision.datasets.ImageFolder(
	                data_dirs[p],
	                transform=data_transforms[p],
	            )
            
	data_loaders[p] = torch.utils.data.DataLoader(
	            data_sets[p], batch_size=batch_size,
	            shuffle=True,
	            num_workers=num_workers, pin_memory=pin_memory,
	            sampler=None)



#=====================================================================================
"""
torchvision 中有一个常用的数据集类 ImageFolder，它假定了数据集是以如下方式构造的:

root/ants/xxx.png
root/ants/xxy.jpeg
root/ants/xxz.png
.
.
.
root/bees/123.jpg
root/bees/nsdf3.png
root/bees/asd932_.png
这里需要说明一下，root是你的根目录，ants和bees是root文件夹下的两个子文件夹，xxx.png、xxy.jpeg、xxz.png是ants文件夹下的图片，
123.jpg、nsdf3.png、asd932_.png是bees文件夹下的图片，也就是说ants和bees是分类标签，利用这些你可以按如下的方式创建一个数据加载器 (dataloader) ,
在这里我们以Imagenet数据集为例：




"""


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

data_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    	     std=[0.229, 0.224, 0.225])
    ])
train_dataset =torchvision.datasets.ImageFolder(root='ILSVRC2012/train',transform=data_transform)
train_dataset_loader =DataLoader(train_dataset,batch_size=4, shuffle=True,num_workers=4)

train_dataset = torchvision.datasets.ImageFolder(root='ILSVRC2012/val',transform=data_transform)
train_dataset_loader = DataLoader(train_dataset,batch_size=4, shuffle=True,num_workers=4)   


"""
Imagenet一共有三个文件夹，分别是train、val、test，train里边有1000个文件夹分别代表1000个类每个类下边有1300张对应类别的图片，val里边有50000张验证集图片，
用ImageFolder进行数据加载的时候，训练集直接按上述办法加载就好，而验证集我们需要对他进行处理，使用如下脚本把验证集50000张图片分别划分到相应类的文件夹中，再进行加载就可以了。
"""




"""
https://blog.csdn.net/geter_CS/article/details/83378786
首先看torch.utils.data.Dataset这个抽象类。可以使用这个抽象类来构造pytorch数据集。要注意的是以这个类构造的子类，一定要定义两个函数一个是__len__，另一个是__getitem__，前者提供数据集size，而后者通过给定索引获取数据和标签。__getitem__一次只能获取一个数据（不知道是不是强制性的），所以通过torch.utils.data.DataLoader来定义一个新的迭代器，实现batch读取。首先我们来定义一个简单的数据集：

"""


import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np

class TxtDataset(Dataset):#这是一个Dataset子类
    def __init__(self):
        self.Data=np.asarray([[1,2],[3,4],[2,1],[6,4],[4,5]])#特征向量集合,特征是2维表示一段文本
        self.Label=np.asarray([1, 2, 0, 1, 2])#标签是1维,表示文本类别
 
    def __getitem__(self, index):
        txt=  self.Data[index]
        label= self.Label[index]
        return txt, label #返回标签
 
    def __len__(self):
        return len(self.Data)




Txt=TxtDataset()
print(f"Txt[1] = {Txt[1]}")
print(f"Txt.__len__() = {Txt.__len__()}")



test_loader = DataLoader(Txt,batch_size=2,shuffle=False,
                          num_workers=4)
for i,traindata in enumerate(test_loader):
    print('i:',i)
    Data,Label=traindata
    print('data:',Data)
    print('Label:',Label)
#这里的enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中




   
     
   




































































































































































































































































































































































































































