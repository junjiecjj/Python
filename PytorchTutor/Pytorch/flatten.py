#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://www.jianshu.com/p/12a8207149b0




"""

pytorch中flatten函数

torch.flatten()

torch.nn.Flatten()
 对于torch.nn.Flatten()，因为其被用在神经网络中，输入为一批数据，第一维为batch，通常要把一个数据拉成一维，
 而不是将一批数据拉为一维。所以torch.nn.Flatten()默认从第二维开始平坦化。


"""

# torch.flatten()
import torch
x=torch.arange(2*4*2).reshape(2,4,2)
print(f"x.shape = {x.shape}")
 
z=torch.flatten(x)
print(f"z.shape = {z.shape}")
 
w=torch.flatten(x,1)
print(f"w.shape = {w.shape}")


import torch
x=torch.arange(2*3*4*4).reshape(2,3,4,4)
print(f"x.shape = {x.shape}")
 
z=torch.flatten(x)
print(f"z.shape = {z.shape}")
 
w=torch.flatten(x,1)
print(f"w.shape = {w.shape}")

#  torch.flatten(x,0,1)代表在第一维和第二维之间平坦化
x=torch.arange(2*4*2).reshape(2,4,2)
print(f"x.shape = {x.shape}")
print(f"x = \n{x}")
 
w=torch.flatten(x,0,1) #第一维长度2，第二维长度为4，平坦化后长度为2*4
print(f"w.shape = {w.shape}")
print(f"w = \n{w}")





# torch.nn.Flatten()

import torch
from torch import nn, optim

#随机32个通道为1的5*5的图
x=torch.randn(32,3,5,5)


#网络模型结构
class Testflatten(nn.Module):
    def __init__(self):
        super(Testflatten, self).__init__()
        
        self.conv = torch.nn.Conv2d(3, 6, 3, 1, 1)
        self.flat = torch.nn.Flatten()


    def forward(self,img):
         print(f"img.shape = {img.shape}")
         a = self.conv(img)
         print(f"a.shape = {a.shape}")
         b = self.flat(a)
         print(f"b.shape = {b.shape}")
         return b


model=torch.nn.Sequential(
    #输入通道为1，输出通道为6，3*3的卷积核，步长为1，padding=1
    torch.nn.Conv2d(3,6,3,1,1),   # 3x5x5 --> 
    torch.nn.Flatten()
)

f = Testflatten()

output=f(x)

print(f"output.shape = {output.shape}")












































































































































