#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch,sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

# https://zhuanlan.zhihu.com/p/75206669
# 1.  nn.Sequential与nn.ModuleList简介
# nn.Sequential
# nn.Sequential里面的模块按照顺序进行排列的，所以必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。如下面的例子所示：

#首先导入torch相关包
import torch
import torch.nn as nn
import torch.nn.functional as F
class net_seq(nn.Module):
    def __init__(self):
        super(net_seq, self).__init__()
        self.seq = nn.Sequential(
                        nn.Conv2d(1,20,5),
                        nn.ReLU(),
                        nn.Conv2d(20,64,5),
                        nn.ReLU()
                    )
    def forward(self, x):
        return self.seq(x)
net_seq = net_seq()
print(net_seq)
#net_seq(
#  (seq): Sequential(
#    (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#    (1): ReLU()
#    (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#    (3): ReLU()
#  )
#)


#nn.Sequential中可以使用OrderedDict来指定每个module的名字，而不是采用默认的命名方式(按序号 0,1,2,3...)。例子如下：
from collections import OrderedDict

class net_seq(nn.Module):
    def __init__(self):
        super(net_seq, self).__init__()
        self.seq = nn.Sequential(OrderedDict([
                        ('conv1', nn.Conv2d(1,20,5)),
                         ('relu1', nn.ReLU()),
                          ('conv2', nn.Conv2d(20,64,5)),
                       ('relu2', nn.ReLU())
                       ]))
    def forward(self, x):
        return self.seq(x)
net_seq = net_seq()
print(net_seq)
#net_seq(
#  (seq): Sequential(
#    (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#    (relu1): ReLU()
#    (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#    (relu2): ReLU()
#  )
#)



# nn.ModuleList
# nn.ModuleList，它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。你可以把任意 nn.Module 的子类 (比如 nn.Conv2d, nn.Linear 之类的) 加到这个 list 里面，方法和 Python 自带的 list 一样，无非是 extend，append 等操作。但不同于一般的 list，加入到 nn.ModuleList 里面的 module 是会自动注册到整个网络上的，同时 module 的 parameters 也会自动添加到整个网络中。若使用python的list，则会出问题。下面看一个例子：

class net_modlist(nn.Module):
    def __init__(self):
        super(net_modlist, self).__init__()
        self.modlist = nn.ModuleList([
                       nn.Conv2d(1, 20, 5),
                       nn.ReLU(),
                        nn.Conv2d(20, 64, 5),
                        nn.ReLU()
                        ])

    def forward(self, x):
        for m in self.modlist:
            x = m(x)
        return x

net_modlist = net_modlist()
print(net_modlist)
#net_modlist(
#  (modlist): ModuleList(
#    (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#    (1): ReLU()
#    (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#    (3): ReLU()
#  )
#)

for param in net_modlist.parameters():
    print(type(param.data), param.size())
#<class 'torch.Tensor'> torch.Size([20, 1, 5, 5])
#<class 'torch.Tensor'> torch.Size([20])
#<class 'torch.Tensor'> torch.Size([64, 20, 5, 5])
#<class 'torch.Tensor'> torch.Size([64])
# 可以看到，这个网络权重 (weithgs) 和偏置 (bias) 都在这个网络之内。
# 接下来看看另一个作为对比的网络，它使用 Python 自带的 list：

class net_modlist(nn.Module):
    def __init__(self):
        super(net_modlist, self).__init__()
        self.modlist = [
                       nn.Conv2d(1, 20, 5),
                       nn.ReLU(),
                        nn.Conv2d(20, 64, 5),
                        nn.ReLU()
                        ]

    def forward(self, x):
        for m in self.modlist:
            x = m(x)
        return x

net_modlist = net_modlist()
print(net_modlist)
#net_modlist()
for param in net_modlist.parameters():
    print(type(param.data), param.size())
#None
print(list(net_modlist.parameters()))
# []


class net2(nn.Module):
    def __init__(self):
        super(net2, self).__init__()
        self.linears = [nn.Linear(10,10) for i in range(2)]
    def forward(self, x):
        for m in self.linears:
            x = m(x)
        return x

net = net2()
print(net)
# net2()
print(list(net.parameters()))
# []



#2.   nn.Sequential与nn.ModuleList的区别
# 不同点1：
# nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数。

# 对于nn.Sequential：



#例1：这是来自官方文档的例子
seq = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
print(seq)
# Sequential(
#   (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (1): ReLU()
#   (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#   (3): ReLU()
# )

#对上述seq进行输入
input = torch.randn(16, 1, 20, 20)
print(seq(input).shape)
#torch.Size([16, 64, 12, 12])

#例2：或者继承nn.Module类的话，就要写出forward函数
class net1(nn.Module):
    def __init__(self):
        super(net1, self).__init__()
        self.seq = nn.Sequential(
                        nn.Conv2d(1,20,5),
                         nn.ReLU(),
                          nn.Conv2d(20,64,5),
                       nn.ReLU()
                       )
    def forward(self, x):
        return self.seq(x)

    #注意：按照下面这种利用for循环的方式也是可以得到同样结果的
    #def forward(self, x):
    #    for s in self.seq:
    #        x = s(x)
    #    return x

 #对net1进行输入
input = torch.randn(16, 1, 20, 20)
net1 = net1()
print(net1(input).shape)
#torch.Size([16, 64, 12, 12])



#而对于nn.ModuleList：

#例1：若按照下面这么写，则会产生错误
modlist = nn.ModuleList([
         nn.Conv2d(1, 20, 5),
         nn.ReLU(),
         nn.Conv2d(20, 64, 5),
         nn.ReLU()
         ])
print(modlist)
#ModuleList(
#  (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#  (1): ReLU()
#  (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#  (3): ReLU()
#)

input = torch.randn(16, 1, 20, 20)
print(modlist(input))
#产生NotImplementedError




#例2：写出forward函数
class net2(nn.Module):
    def __init__(self):
        super(net2, self).__init__()
        self.modlist = nn.ModuleList([
                       nn.Conv2d(1, 20, 5),
                       nn.ReLU(),
                        nn.Conv2d(20, 64, 5),
                        nn.ReLU()
                        ])

    #这里若按照这种写法则会报NotImplementedError错
    #def forward(self, x):
    #    return self.modlist(x)

    #注意：只能按照下面利用for循环的方式
    def forward(self, x):
        for m in self.modlist:
            x = m(x)
        return x

input = torch.randn(16, 1, 20, 20)
net2 = net2()
print(net2(input).shape)
#torch.Size([16, 64, 12, 12])


# 如果完全直接用 nn.Sequential，确实是可以的，但这么做的代价就是失去了部分灵活性，不能自己去定制 forward 函数里面的内容了。

# 一般情况下 nn.Sequential 的用法是来组成卷积块 (block)，然后像拼积木一样把不同的 block 拼成整个网络，让代码更简洁，更加结构化。

# 不同点2：
# nn.Sequential可以使用OrderedDict对每层进行命名，上面已经阐述过了；

# 不同点3：
# nn.Sequential里面的模块按照顺序进行排列的，所以必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。而nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起，这些模块之间并没有什么先后顺序可言。见下面代码：

class net3(nn.Module):
    def __init__(self):
        super(net3, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10,20), nn.Linear(20,30), nn.Linear(5,10)])
    def forward(self, x):
        x = self.linears[2](x)
        x = self.linears[0](x)
        x = self.linears[1](x)

        return x

net3 = net3()
print(net3)
#net3(
#  (linears): ModuleList(
#    (0): Linear(in_features=10, out_features=20, bias=True)
#    (1): Linear(in_features=20, out_features=30, bias=True)
#    (2): Linear(in_features=5, out_features=10, bias=True)
#  )
#)

input = torch.randn(32, 5)
print(net3(input).shape)
#torch.Size([32, 30])


# 不同点4：
# 有的时候网络中有很多相似或者重复的层，我们一般会考虑用 for 循环来创建它们，而不是一行一行地写，比如：

# layers = [nn.Linear(10, 10) for i in range(5)]
# 那么这里我们使用ModuleList：

class net4(nn.Module):
    def __init__(self):
        super(net4, self).__init__()
        layers = [nn.Linear(10, 10) for i in range(5)]
        self.linears = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x

net = net4()
print(net)
# net4(
#   (linears): ModuleList(
#     (0): Linear(in_features=10, out_features=10, bias=True)
#     (1): Linear(in_features=10, out_features=10, bias=True)
#     (2): Linear(in_features=10, out_features=10, bias=True)
#   )
# )








def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)



# 方法一：类继承nn.Module，必须实现forward函数
class ResBlock1(nn.Module):
     def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock1, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

     def forward(self, x):
           res = self.body(x).mul(self.res_scale)
           res += x
           return res


#方法2：类继承nn.Sequential,但是这么做失去部分灵活性
class ResBlock2(nn.Sequential):
     def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock2, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

     # def forward(self, x):
     #       res = self.body(x).mul(self.res_scale)
     #       res += x
     #       return res


conv = default_conv
input1 = torch.randn(16, 64, 20, 20)


resb = ResBlock1(conv, 64, 5)
a1 = resb(input1)





resb = ResBlock2(conv, 64, 5)
a2 = resb(input1)







class Upsampler(nn.Module):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        super(Upsampler, self).__init__()
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                # # Pixelshuffle会将shape为 (*, r^2C, H, W) 的Tensor给reshape成 (*, C, rH,rW) 的Tensor
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

       #  super(Upsampler, self).__init__(*m)

        self.body = nn.Sequential(*m)

    def forward(self, x):
           res = self.body(x)
           return res


conv = default_conv
input1 = torch.randn(16, 64, 20, 20)

us = Upsampler(conv, 4, 64, act=False)
out = us(input1)

print(f"out.shape = {out.shape}")
#  out.shape = torch.Size([16, 64, 80, 80])



class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                # # Pixelshuffle会将shape为 (*, r^2C, H, W) 的Tensor给reshape成 (*, C, rH,rW) 的Tensor
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

conv = default_conv
input1 = torch.randn(16, 64, 20, 20)

us = Upsampler(conv, 4, 64, act=False)
out = us(input1)

print(f"out.shape = {out.shape}")
#  out.shape = torch.Size([16, 64, 80, 80])
















