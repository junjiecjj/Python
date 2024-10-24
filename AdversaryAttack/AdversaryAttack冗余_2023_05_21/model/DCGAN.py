#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2023/04/25
@author: Junjie Chen

https://blog.csdn.net/qq_39547794/article/details/125389000

https://zhuanlan.zhihu.com/p/149563859

https://blog.csdn.net/jizhidexiaoming/article/details/96485095

https://blog.csdn.net/weixin_50113231/article/details/122959899


https://zhuanlan.zhihu.com/p/55991450

https://www.cnblogs.com/picassooo/p/12601909.html

https://zhuanlan.zhihu.com/p/72987027

https://blog.csdn.net/qq_39547794/article/details/125409710

https://www.cnblogs.com/picassooo/p/12601909.html

https://github.com/longpeng2008/yousan.ai/tree/master/books/%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/%E7%AC%AC5%E7%AB%A0/DCGAN

https://github.com/eriklindernoren/PyTorch-GAN

https://blog.csdn.net/qq_39547794/article/details/125409710

https://github.com/venkateshtata/GAN_Medium/blob/master/dcgan.py

"""



# 系统库
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
import copy
import datetime
#内存分析工具
from memory_profiler import profile
import objgraph
import sys, os


# 自己的库
#sys.path.append(os.getcwd())
from model  import common
# 或
# from .  import common

sys.path.append("..")
from  Option import args
from  ColorPrint import ColoPrint
color =  ColoPrint()

#================================================= GAN for minst ==========================================================


## ##### 定义判别器 Discriminator ######
## 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
## 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类
class Minst_Discriminator(nn.Module):
    def __init__(self, args ):
        super(Minst_Discriminator, self).__init__()
        self.minst_shape = (args.Minst_channel, args.Minst_heigh, args.Minst_width)
        self.minst_dim = np.prod(self.minst_shape)
        self.model = nn.Sequential(
            nn.Linear(self.minst_dim, 512),             ## 输入特征数为784，输出为512
            nn.LeakyReLU(0.2, inplace=True),            ## 进行非线性映射
            nn.Linear(512, 256),                        ## 输入特征数为512，输出为256
            nn.LeakyReLU(0.2, inplace=True),            ## 进行非线性映射
            nn.Linear(256, 1),                          ## 输入特征数为256，输出为1
            nn.Sigmoid(),                               ## sigmoid是一个激活函数，二分类问题中可将实数映射到[0, 1],作为概率值, 多分类用softmax函数
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)            ## 鉴别器输入是一个被view展开的(784)的一维图像:(64, 784)
        validity = self.model(img_flat)                 ## 通过鉴别器网络
        return validity                                 ## 鉴别器返回的是一个[0, 1]间的概率



## ###### 定义生成器 Generator #####
## 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
## 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
## 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布, 能够在-1～1之间。
class Minst_Generator(nn.Module):
    def __init__(self, args):
        super(Minst_Generator, self).__init__()
        self.minst_shape = (args.Minst_channel, args.Minst_heigh, args.Minst_width)
        self.minst_dim = np.prod(self.minst_shape)
        ## 模型中间块儿
        def block(in_feat, out_feat, normalize=True):           ## block(in， out )
            layers = [nn.Linear(in_feat, out_feat)]             ## 线性变换将输入映射到out维
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))    ## 正则化
            layers.append(nn.LeakyReLU(0.2, inplace=True))      ## 非线性激活函数
            return layers
        ## prod():返回给定轴上的数组元素的乘积:1*28*28=784
        self.model = nn.Sequential(
            *block(args.noise_dim, 128, normalize=False),       ## 线性变化将输入映射 100 to 128, 正则化, LeakyReLU
            *block(128, 256),                                   ## 线性变化将输入映射 128 to 256, 正则化, LeakyReLU
            *block(256, 512),                                   ## 线性变化将输入映射 256 to 512, 正则化, LeakyReLU
            *block(512, 1024),                                  ## 线性变化将输入映射 512 to 1024, 正则化, LeakyReLU
            nn.Linear(1024, self.minst_dim),                          ## 线性变化将输入映射 1024 to 784
            nn.Tanh()                                           ## 将(784)的数据每一个都映射到[-1, 1]之间
        )
    ## view():相当于numpy中的reshape，重新定义矩阵的形状:这里是reshape(64, 1, 28, 28)
    def forward(self, z):                                       ## 输入的是(64， 100)的噪声数据
        #print(f"z.shape = {z.shape}")
        imgs = self.model(z)                                     ## 噪声数据通过生成器模型
        imgs = imgs.view(imgs.size(0), *self.minst_shape)                 ## reshape成(64, 1, 28, 28)
        #print(f"2  imgs.shape = {imgs.shape}")
        return imgs                                              ## 输出为64张大小为(1, 28, 28)的图像



#================================================= GAN for cifer ==========================================================

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CIFER_Discriminator(nn.Module):
    def __init__(self):
        super(CIFER_Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)


class CIFER_Generator(nn.Module):
    def __init__(self):
        super(CIFER_Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output




#==========================================================================================================================
#  GAN for Gauss 
#==========================================================================================================================

"""
https://www.pytorchtutorial.com/pytorch-sample-gan/

https://github.com/rcorbish/pytorch-notebooks/blob/master/gan-basic.ipynb

https://blog.csdn.net/lizzy05/article/details/90611102

https://blog.csdn.net/qq_40994260/article/details/114699755



"""



class Gauss_Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Gauss_Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.xfer = torch.nn.SELU()
    def forward(self, x):
        x = self.xfer( self.map1(x) )
        x = self.xfer( self.map2(x) )
        return self.xfer( self.map3( x ) )


class Gauss_Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Gauss_Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.elu = torch.nn.ELU()
 
    def forward(self, x):
        x = self.elu(self.map1(x))
        x = self.elu(self.map2(x))
        return torch.sigmoid( self.map3(x) )
































































































































































































































































