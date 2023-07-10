#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 11:07:19 2023

@author: jack

本文件主要测试的是：
(一): 获取模型的参数;
(二)：加载模型的参数和改变模型的参数;
(三)：查看在模型训练过程中，什么时候模型参数会改变，什么时候模型参数会产生梯度；答案是在optimizer.step()之后模型参数会改变，在loss.backward()之后会产生梯度;
"""


import sys,os
import torch
from torch.autograd import Variable
import torch.nn as nn
import imageio
import matplotlib
matplotlib.use('TkAgg')

import torch.optim as optim


"""
在使用pytorch过程中，我发现了torch中存在3个功能极其类似的方法，它们分别是model.parameters()、model.named_parameters()和model.state_dict()，下面就具体来说说这三个函数的差异
首先，说说比较接近的model.parameters()和model.named_parameters()。这两者唯一的差别在于，named_parameters()返回的list中，每个元祖打包了2个内容，分别是layer-name和layer-param，而parameters()只有后者。后面只谈model.named_parameters()和model.state_dict()间的差别。

它们的差异主要体现在3方面：

返回值类型不同
存储的模型参数的种类不同
返回的值的require_grad属性不同
首先说第一个，这很简单，model.state_dict()是将layer_name : layer_param的键值信息存储为dict形式，而model.named_parameters()则是打包成一个元祖然后再存到list当中；
第二，model.state_dict()存储的是该model中包含的所有layer中的所有参数；而model.named_parameters()则只保存可学习、可被更新的参数，model.buffer()中的参数不包含在model.named_parameters()中
最后，model.state_dict()所存储的模型参数tensor的require_grad属性都是False，而model.named_parameters()的require_grad属性都是True

"""

#===============================================================================================================
#                                            打印每一层的参数名和参数值
#===============================================================================================================

# 定义一个简单的网络
class net(nn.Module):
    def __init__(self, num_class=10):
        super(net, self).__init__()
        self.fc1 = torch.nn.Linear(8, 4)
        self.fc2 = torch.nn.Linear(4, num_class)


    def forward(self, x):
        return self.fc2(self.fc1(x))

model = net()  # .to("cuda:0")

print(f"模型结构为：\n{model}, \n模型参数为:")
for name, param in  model.named_parameters():
    if param.requires_grad:
        #print(f"{name}: {param.size()}, {param.requires_grad} ")
        print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ")


#======================================== 1: state_dict ===============================================
#打印某一层的参数名
for name in model.state_dict():
    print(name)
#Then  I konw that the name of target layer is '1.weight'

#schemem1(recommended)
print(f"\n\nmodel.state_dict()['fc1.weight'] = \n    {model.state_dict()['fc1.weight']}")


#======================================== 2 : state_dict ===============================================
#打印每一层的参数名和参数值
params = {}
lc = model.state_dict()
for key in lc:
    params[key] = lc[key] # .detach().cpu().numpy()
    print(f" {key}, {lc[key].is_leaf}, {lc[key].shape}, {lc[key].device}, {lc[key].requires_grad}, {lc[key].type()} \n  {lc[key]}")
    # print(name)
    # print(ae.state_dict()[name])


params = {}
lc = model.state_dict().items()
for key, var in lc:
    params[key] = var # .detach().cpu().numpy()
    # print("key:"+str(key)+",var:"+str(var))
    print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var}" )
    # print(f"张量{key}的Size : "+str(var.size()))

#======================================== 3: named_parameters ===============================================
#打印每一层的参数名和参数值
params = list(model.named_parameters())  #get the index by debuging
l = len(params)
for i in range(l):
    # print(params[i][0])              # name
    # print(params[i][1].data)         # data
    print(f" params[{i}][0] = {params[i][0]}, \n params[{i}][1].data = \n      {params[i][1].data}")


#======================================== 4: named_parameters ===============================================
## 打印每一层的参数名和参数值
##  schemem1(recommended)
params = {}
for name, param in model.named_parameters():
    # print(f"  name = {name}\n  param = \n    {param}")
    params[name] = param.data.detach()# .cpu().numpy()
    print(f"{name}, {param.data.is_leaf}, {param.size()}, {param.device}, {param.requires_grad}, {param.type()} :\n  {param.data}")

## Print dict, 1
for key, pam in params.items():
    print(f"{key}: {pam.device},  \n    {pam}")

## Print dict, 2
for key in params:
    print(f"{key}: {pam.device},  \n    {params[key]}")

#======================================== 5: parameters ===============================================
#打印出参数矩阵及值
for parameters in model.parameters():
        print(f"{parameters.is_leaf}, {parameters.shape}, {parameters.requires_grad} {parameters.type()}, ")


#===============================================================================================================
#                                            模型参数加载
#===============================================================================================================

# 定义一个简单的网络
class net(nn.Module):
    def __init__(self, num_class=10):
        super(net, self).__init__()
        self.fc1 = torch.nn.Linear(8, 4)
        self.fc2 = torch.nn.Linear(4, num_class)


    def forward(self, x):
        return self.fc2(self.fc1(x))

model = net().to("cuda:0")

## 加载时的参数可以是在cpu上，模型在哪加载后的参数就在哪，与加载前的参数的device无关，只与模型的device有关

params = {}
for key, var in model.state_dict().items():
    params[key] = var.clone().cpu() #.detach()
    print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()} \n  {var}" )


tmp_param = {}
for i , (key, val) in enumerate(params.items()):
    tmp_param[key] = torch.ones_like(val) + i

#======================================== 1 : 加载字典 ===============================================
print(f"model.state_dict() = \n{model.state_dict()} \n\n")

model.load_state_dict(tmp_param)
print(f"model.state_dict() = \n{model.state_dict()} \n\n")


model.load_state_dict(params)
print(f"model.state_dict() = \n{model.state_dict()} \n\n")
for key, var in model.state_dict().items():
    print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()} \n  {var}" )


#======================================== 1 : 加载字典 ===============================================


import copy

print(f"model.state_dict() = \n{model.state_dict()} \n\n")

param_sd = copy.deepcopy(params)
param_sd["fc2.bias"] += 10
model.load_state_dict(param_sd)
print(f"model.state_dict() = \n{model.state_dict()} \n\n")


#======================================== 2: 加载 model.state_dict().items():" error ===============================================
import copy

model.load_state_dict(model.state_dict().items())
## TypeError: Expected state_dict to be dict-like, got <class 'odict_items'>.


#======================================== 3: 加载 model.state_dict() :" ===============================================
import copy

model = net()  # .to("cuda:0")

sd = copy.deepcopy(model.state_dict())
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, param in model.state_dict().items():
    model.state_dict()[key].add_(10)
print(f"model.state_dict() = {model.state_dict()} \n\n")


model.load_state_dict(sd)
print(f"model.state_dict() = {model.state_dict()} \n\n")





#===============================================================================================================
#                                            改变模型参数
#===============================================================================================================

# 定义一个简单的网络
class net(torch.nn.Module):
    def __init__(self, num_class=10):
        super(net, self).__init__()
        self.fc1 = torch.nn.Linear(8, 4)
        self.fc2 = torch.nn.Linear(4, num_class)


    def forward(self, x):
        return self.fc2(self.fc1(x))

model = net().to("cuda:0")




orig_params = {}
for key, var in model.state_dict().items():
    orig_params[key] = var.clone()#.cpu()   # .detach().cpu().numpy()
    print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()} \n  {var}" )


tmp_param = {}
for i , (key, val) in enumerate(orig_params.items()):
    tmp_param[key] = torch.ones_like(val.clone() ) + i



# for key, var in model.state_dict().items():
#     orig_params[key] += tmp_param[key]
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

#================================  random mask =================================================
print(f"model.state_dict() = {model.state_dict()} \n\n")

mask = {}
for name, param in model.state_dict().items():
    p = torch.ones_like(param)*0.6
    if torch.is_floating_point(param):
        mask[name] = torch.bernoulli(p)
    else:
        mask[name] = torch.bernoulli(p).long()

## 1
## 无效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    model.state_dict()[key]  =  model.state_dict()[key]*mask[key]
print(f"model.state_dict() = {model.state_dict()} \n\n")

## 2
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    model.state_dict()[key] *= mask[key]
print(f"model.state_dict() = {model.state_dict()} \n\n")


## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    model.state_dict()[key].mul_(mask[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")



## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    var.mul_(mask[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")

## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 无效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    var = var*(mask[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")


## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    var *= (mask[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")



## 4
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, param in model.named_parameters():
    param.data  *= mask[key]
print(f"model.state_dict() = {model.state_dict()} \n\n")


## 5
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, param in model.named_parameters():
    param.data   =  param.data * mask[key]
print(f"model.state_dict() = {model.state_dict()} \n\n")


#======================================== 0: model.load_state_dict(tmp_param) 有效" ===============================================
model.load_state_dict(tmp_param)
print(f"model.state_dict() = {model.state_dict()} \n\n")


#======================================== var: 加法 有效 ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    var.add_(10)
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== var: 加法 无效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    var = var + 10
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== var: 加法 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    var += 10
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== var: 加法: 有效 ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    var.add_(tmp_param[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== var: 加法: 无效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    var = var + tmp_param[key]
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== var: 加法 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    var += tmp_param[key]
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== var: 等于 有效 ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    var.copy_(tmp_param[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== var: 等于 无效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    var = tmp_param[key]
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== var 等于 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    var.copy_(torch.ones_like(var))
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== var 乘法 有效  ===============================================
## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    var.mul_(10)
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== var 乘法 无效  ===============================================
## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 无效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    var = var*(10)
print(f"model.state_dict() = {model.state_dict()} \n\n")


#======================================== var 乘法 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    var *= (10)
print(f"model.state_dict() = {model.state_dict()} \n\n")


#======================================== var 乘法 有效  ===============================================
## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    var.mul_(mask[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== var 乘法 无效  ===============================================
## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 无效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    var = var*(mask[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")


#======================================== var 乘法 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    var *= (mask[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")

#####################################################################################################################
#======================================== model.state_dict() :: 加法 有效 ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    model.state_dict()[key].add_(10)
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.state_dict() :: 加法 无效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    model.state_dict()[key] = model.state_dict()[key] + 10
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.state_dict() :: 加法 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    model.state_dict()[key] += 10
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.state_dict() : 加法: 有效 ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    model.state_dict()[key].add_(tmp_param[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.state_dict() : 加法: 无效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    model.state_dict()[key] = model.state_dict()[key] + tmp_param[key]
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.state_dict() : 加法 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    model.state_dict()[key] += tmp_param[key]
print(f"model.state_dict() = {model.state_dict()} \n\n")

#========================================  model.state_dict() : 等于 有效 ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    # print(f"0: {var}, {tmp_param[key]}")
    model.state_dict()[key].copy_(tmp_param[key])
    # print(f"1: {model.state_dict()[key]}, {tmp_param[key]}")
print(f"model.state_dict() = {model.state_dict()} \n\n")


#======================================== model.state_dict() : 等于 无效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    model.state_dict()[key] = tmp_param[key]
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.state_dict()  等于 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, var in model.state_dict().items():
    model.state_dict()[key].copy_(torch.ones_like(var))
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.state_dict()  乘法 有效  ===============================================
## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    model.state_dict()[key].mul_(10)
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.state_dict()  乘法 无效  ===============================================
## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 无效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    model.state_dict()[key] = model.state_dict()[key]*10
print(f"model.state_dict() = {model.state_dict()} \n\n")


#======================================== model.state_dict()  乘法 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    model.state_dict()[key] *= 10
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.state_dict()  乘法 有效  ===============================================
## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    model.state_dict()[key].mul_(mask[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.state_dict()  乘法 无效  ===============================================
## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 无效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    model.state_dict()[key] = model.state_dict()[key]*(mask[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")


#======================================== model.state_dict()  乘法 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, var in model.state_dict().items():
    # var.mul_(mask[key])
    model.state_dict()[key] *= (mask[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")

##############################################################################################################################################

#======================================== model.named_parameters() : param.data:: 加法 有效 ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, param in model.named_parameters():
    param.data.add_(10)
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.named_parameters() : param.data:: 加法 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, param in model.named_parameters():
    param.data = param.data + 10
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.named_parameters() : param.data :: 加法 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, param in model.named_parameters():
    param.data += 10
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.named_parameters() : param.data: 加法: 有效 ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, param in model.named_parameters():
    param.data.add_(tmp_param[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.named_parameters() : param.data: 加法: 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, param in model.named_parameters():
    param.data = param.data + tmp_param[key]
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.named_parameters() : param.data: 加法 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, param in model.named_parameters():
    param.data += tmp_param[key]
print(f"model.state_dict() = {model.state_dict()} \n\n")

#========================================  model.named_parameters() : param.data =  等于 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, param in model.named_parameters():
    param.data = tmp_param[key].clone()  # must add .clone(), or  tmp_param will change when param.data  changed
print(f"model.state_dict() = {model.state_dict()} \n\n")


#========================================  model.named_parameters() : param.data =  等于 有效 ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, param in model.named_parameters():
    # print(f"0: {var}, {tmp_param[key]}")
    param.data.copy_(tmp_param[key])
    # print(f"1: {model.state_dict()[key]}, {tmp_param[key]}")
print(f"model.state_dict() = {model.state_dict()} \n\n")


#======================================== model.named_parameters() : param.data =   等于 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for key, param in model.named_parameters():
    param.data.copy_(torch.ones_like(param))
print(f"model.state_dict() = {model.state_dict()} \n\n")



#======================================== model.named_parameters() : param.data   乘法 有效  ===============================================
## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, param in model.named_parameters():
    # var.mul_(mask[key])
    param.data.mul_(10)
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.named_parameters() : param.data  乘法 有效  ===============================================
## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, param in model.named_parameters():
    # var.mul_(mask[key])
    param.data = param.data*10
print(f"model.state_dict() = {model.state_dict()} \n\n")


#======================================== model.named_parameters() : param.data  乘法 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, param in model.named_parameters():
    # var.mul_(mask[key])
    param.data *= 10
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.named_parameters() : param.data   乘法 有效  ===============================================
## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, param in model.named_parameters():
    # var.mul_(mask[key])
    param.data.mul_(mask[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== model.named_parameters() : param.data 乘法 有效  ===============================================
## 3
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, param in model.named_parameters():
    # var.mul_(mask[key])
    param.data = param.data*(mask[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")


#======================================== model.named_parameters() : param.data  乘法 有效  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")
## 有效
for key, param in model.named_parameters():
    # var.mul_(mask[key])
    param.data *= (mask[key])
print(f"model.state_dict() = {model.state_dict()} \n\n")


#======================================== 2: model.named_parameters() : param.data.add_  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for name, param in model.named_parameters():
    param.data.add_(tmp_param[name].clone())
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== 2: model.named_parameters() : model.state_dict()[name].add_  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for name, param in model.named_parameters():
    model.state_dict()[name].add_(tmp_param[name].clone())
print(f"model.state_dict() = {model.state_dict()} \n\n")

#======================================== 2: model.named_parameters() : model.state_dict()[name].copy_  ===============================================
model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")

for name, param in model.named_parameters():
    model.state_dict()[name].copy_(tmp_param[name].clone())
print(f"model.state_dict() = {model.state_dict()} \n\n")



#===============================================================================================================
#                                           is_leaf – 查看张量是否为叶张量
# https://blog.csdn.net/hxxjxw/article/details/122281531
# https://blog.csdn.net/zphuangtang/article/details/112788037
# https://blog.csdn.net/m0_46653437/article/details/112934222
# https://zhuanlan.zhihu.com/p/279758736
#===============================================================================================================
"""
  在Pytorch中，默认情况下，非叶节点的梯度值在反向传播过程中使用完后就会被清除，不会被保留。只有叶节点的梯度值能够被保留下来。

      对于任意一个张量来说，我们可以用 tensor.is_leaf 来判断它是否是叶子张量（leaf tensor）

      在Pytorch神经网络中，我们反向传播backward()就是为了求叶子节点的梯度。在pytorch中，神经网络层中的权值w的tensor均为叶子节点。它们的require_grad都是True，但它们都属于用户创建的，所以都是叶子节点。而反向传播backward()也就是为了求它们的梯度

       在调用backward()时，只有当requires_grad和is_leaf同时为真时，才会计算节点的梯度值

为什么需要叶子节点？
        那些非叶子节点，是通过用户所定义的叶子节点的一系列运算生成的，也就是这些非叶子节点都是中间变量，一般情况下，用户不回去使用这些中间变量的导数，所以为了节省内存，它们在用完之后就被释放了

在Pytorch的autograd机制中，当tensor的requires_grad值为True时，在backward()反向传播计算梯度时才会被计算。在所有的require_grad=True中，

默认情况下，非叶子节点的梯度值在反向传播过程中使用完后就会被清除，不会被保留(即调用loss.backward() 会将计算图的隐藏变量梯度清除)。
默认情况下，只有叶子节点的梯度值能够被保留下来。
被保留下来的叶子节点的梯度值会存入tensor的grad属性中，在 optimizer.step()过程中会更新叶子节点的data属性值，从而实现参数的更新。
这样可以节省很大部分的显存

上面的话，也就是说，并不是每个requires_grad()设为True的tensor都会在backward的时候得到相应的grad.它还必须为leaf。这就说明. is_leaf=True 成为了在 requires_grad()下判断是否需要保留 grad的前提条件

      只有是叶张量的tensor在反向传播时才会将本身的grad传入的backward的运算中.。如果想得到当前自己创建的，requires_grad为True的tensor在反向传播时的grad, 可以用retain_grad()这个属性(或者是hook机制)

detach()将节点剥离成叶子节点
      如果需要使得某一个节点成为叶子节点，只需使用detach()即可将它从创建它的计算图中分离开来。即detach()函数的作用就是把一个节点从计算图中剥离，使其成为叶子节点

什么样节点会是叶子节点
①所有requires_grad为False的张量，都约定俗成地归结为叶子张量

         就像我们训练模型的input，它们都是require_grad=False，因为他们不需要计算梯度(我们训练网络训练的是网络模型的权重，而不需要训练输入)。它们是一个计算图都是起始点，如下图的a

②requires_grad为True的张量, 如果他们是由用户创建的,则它们是叶张量(leaf Tensor)。

        例如各种网络层，nn.Linear(), nn.Conv2d()等, 他们是用户创建的，而且其网络参数也需要训练，所以requires_grad=True

这意味着它们不是运算的结果,因此gra_fn为None


requires_grad: 如果需要为张量计算梯度，则为True，否则为False。我们使用pytorch创建tensor时，可以指定requires_grad为True（默认为False），
grad_fn： grad_fn用来记录变量是怎么来的，方便计算梯度，y = x*3,grad_fn记录了y由x计算的过程。
grad：当执行完了backward()之后，通过x.grad查看x的梯度值。

创建一个Tensor并设置requires_grad=True，requires_grad=True说明该变量需要计算梯度。
由于x是直接创建的，所以它没有grad_fn，而y是通过一个加法操作创建的，所以y有grad_fn
像x这种直接创建的称为叶子节点，叶子节点对应的grad_fn是None。
requires_grad属性是可以改变的
通过.requires_grad_()来用in-place的方式改变requires_grad属性：

"""

### is_leaf – 查看张量是否为叶张量
##==========================================================================
## 1.所有被需要计算梯度的变量直接赋值(不进行任何操作)创建的都是叶张量 – 不包含任何操作

a = torch.rand(10, requires_grad=True)  # 直接赋给a，所以a.is_leaf为True
print(f"0:  a.is_leaf = {a.is_leaf}")

a = torch.rand(10, requires_grad=True, device="cuda") # 直接创建赋给a的，所以为True
print(f"1:  a.is_leaf = {a.is_leaf}")

a = torch.rand(10, requires_grad=True) + 5  # 运算后赋给a，所以a.is_leaf为False
print(f"2:  a.is_leaf = {a.is_leaf}")


a = torch.rand(10, requires_grad=True).cuda()  # 将数据移到gpu上再赋值给a，所以也是False
print(f"3:  a.is_leaf = {a.is_leaf}")

##==========================================================================
## 2.所有不需要计算梯度张量都是叶张量

# all_leaf is False
a = torch.rand(10)    # 非梯度tensor -- 总是为False
print(f"4:  a.is_leaf = {a.is_leaf}")

a = torch.rand(10) + 5
print(f"5:  a.is_leaf = {a.is_leaf}")


a = torch.rand(10).cuda()
print(f"6:  a.is_leaf = {a.is_leaf}")

##==========================================================================
## 3.由不需要梯度的张量创建的新的需要梯度的张量是叶张量

# all_leaf is True
a = torch.rand(10).requires_grad_()# 由非梯度tensor变成梯度tensor后直接赋给，可以成为叶张量
print(f"7:  a.is_leaf = {a.is_leaf}")


#由非梯度tensor移动到gpu上再变成梯度tensor后直接赋给，可以成为叶张量
a = torch.rand(10).cuda().requires_grad_()
print(f"8:  a.is_leaf = {a.is_leaf}")


#===============================================================================================================
#                                                      PyTorch求导相关
#===============================================================================================================

"""
PyTorch是动态图，即计算图的搭建和运算是同时的，随时可以输出结果；而TensorFlow是静态图。

在pytorch的计算图里只有两种元素：数据（tensor）和 运算（operation）

运算包括了：加减乘除、开方、幂指对、三角函数等可求导运算

数据可分为：叶子节点（leaf node）和非叶子节点；叶子节点是用户创建的节点，不依赖其它节点；它们表现出来的区别在于反向传播结束之后，非叶子节点的梯度会被释放掉，只保留叶子节点的梯度，这样就节省了内存。如果想要保留非叶子节点的梯度，可以使用retain_grad()方法。

torch.tensor 具有如下属性：

查看 是否可以求导 requires_grad
查看 运算名称 grad_fn
查看 是否为叶子节点 is_leaf
查看 导数值 grad
针对requires_grad属性，自己定义的叶子节点默认为False，而非叶子节点默认为True，神经网络中的权重默认为True。判断哪些节点是True/False的一个原则就是从你需要求导的叶子节点到loss节点之间是一条可求导的通路。


当我们想要对某个Tensor变量求梯度时，需要先指定requires_grad属性为True，指定方式主要有两种：

x = torch.tensor(1.).requires_grad_() # 第一种

x = torch.tensor(1., requires_grad=True) # 第二种

PyTorch提供两种求梯度的方法：backward() and torch.autograd.grad() ，他们的区别在于前者是给叶子节点填充.grad字段，而后者是直接返回梯度给你，我会在后面举例说明。还需要知道y.backward()其实等同于torch.autograd.backward(y)

使用backward()函数反向传播计算tensor的梯度时，并不计算所有tensor的梯度，而是只计算满足这几个条件的tensor的梯度：1.类型为叶子节点、2.requires_grad=True、3.依赖该tensor的所有tensor的requires_grad=True。所有满足条件的变量梯度会自动保存到对应的grad属性里。

"""


x = torch.tensor(2., requires_grad=True)

a = torch.add(x, 1)
b = torch.add(x, 2)
y = torch.mul(a, b)

y.backward()
print(x.grad)

print("requires_grad: ", x.requires_grad, a.requires_grad, b.requires_grad, y.requires_grad)
print("is_leaf: ", x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
print("grad: ", x.grad, a.grad, b.grad, y.grad)

# 使用detach()切断
# 不会再往后计算梯度，假设有模型A和模型B，我们需要将A的输出作为B的输入，但训练时我们只训练模型B，那么可以这样做：
# input_B = output_A.detach()
# 如果还是以前面的为例子，将a切断，将只有b一条通路，且a变为叶子节点。
x = torch.tensor([2.], requires_grad=True)

a = torch.add(x, 1).detach()
b = torch.add(x, 2)
y = torch.mul(a, b)

y.backward()

print("requires_grad: ", x.requires_grad, a.requires_grad, b.requires_grad, y.requires_grad)
print("is_leaf: ", x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
print("grad: ", x.grad, a.grad, b.grad, y.grad)


#===============================================================================================================
#                          测试模型参数在模型训练过程中什么时候改变，结果为：在optimizer.step()之后会改变
#===============================================================================================================
import sys,os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import torch.optim as optim


# 定义一个简单的网络
class net(nn.Module):
    def __init__(self, num_class=10):
        super(net, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, num_class)


    def forward(self, x):
        return self.fc2(self.fc1(x))

model = net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)  # 传入的是所有的参数
x = torch.randn((3, 8))
label = torch.randint(0,10,[3]).long()


for key, var in model.named_parameters():
    print(f"{key:<10}, {var.is_leaf}, {var.size()}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n{var.data}")


for key, var in model.state_dict().items():
    print(f"0: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var} \n\n" )
# for key, var in model.named_parameters():
    # print(f"0: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")



output = model(x)
#print(f"epoch = {epoch}, x.shape = {x.shape}, output.shape = {output.shape}")
loss = loss_fn(output, label)
for key, var in model.state_dict().items():
    print(f"1: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var}" )
print("\n\n")

optimizer.zero_grad()
for key, var in model.state_dict().items():
    print(f"2: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var} " )
print("\n\n")

loss.backward()
for key, var in model.state_dict().items():
    print(f"3: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var} " )
print("\n\n")

optimizer.step()
for key, var in model.state_dict().items():
    print(f"4: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var} " )
print("\n\n")




#===============================================================================================================
#                     测试模型训练过程中什么时候会产生参数的梯度，结果为：在loss.backward()之后会产生梯度
#===============================================================================================================
import sys,os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import torch.optim as optim


# 定义一个简单的网络
class net(nn.Module):
    def __init__(self, num_class=10):
        super(net, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, num_class)


    def forward(self, x):
        return self.fc2(self.fc1(x))

model = net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)  # 传入的是所有的参数
x = torch.randn((3, 8))
label = torch.randint(0,10,[3]).long()


for key, var in model.named_parameters():
    print(f"{key:<10}, {var.is_leaf}, {var.size()}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n{var.data}")


# for key, var in model.state_dict().items():
    # print(f"0: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var} \n\n" )
for key, var in model.named_parameters():
    print(f"0: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")



output = model(x)
#print(f"epoch = {epoch}, x.shape = {x.shape}, output.shape = {output.shape}")
loss = loss_fn(output, label)

for key, var in model.named_parameters():
    print(f"1: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")
print("\n\n")

optimizer.zero_grad()
for key, var in model.named_parameters():
    print(f"2: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")
print("\n\n")

loss.backward()
for key, var in model.named_parameters():
    print(f"3: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")
print("\n\n")

optimizer.step()
for key, var in model.named_parameters():
    print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")
print("\n\n")


# for key, var in model.state_dict().items():
#     print(f"4: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var} " )
# for key, var in model.named_parameters():
    # print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")


"""

模型训练过程: 梯度清零，loss求导(这一步获得梯度), optim.step()(这一步梯度更新，模型参数改变)

梯度剪裁 torch.nn.utils.clip_grad_norm_()的使用应该在loss.backward()之后，optimizer.step()之前;

为了获取梯度,只能使用如下方法: model.state_dict().items()是行不通的;
for k, v in net.named_parameters():
    print(v.grad)
"""















































































































































































































































































































































































































































































