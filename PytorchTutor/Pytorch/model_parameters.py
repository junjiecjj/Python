#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
本文件主要测试的是：
(一): 获取模型的参数;
(二)：加载模型的参数和改变模型的参数;


验证模型保存时保存的.pt文件大小是否改变，以及随着训练的进行，模型的参数数值是否改变。结果表明：

1. 一旦模型确定，则模型的pt大小是确定的，而不管模型的参数怎么变。
2. 随着训练过程的持续，模型的参数一直在变。
3. 随着训练过程的推荐，冻结的那些层的参数不会改变。

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
    (1) 返回值类型不同: 首先说第一个，这很简单，model.state_dict()是将layer_name : layer_param的键值信息存储为dict形式，而model.named_parameters()则是打包成一个元祖然后再存到list当中；
    (2) 存储的模型参数的种类不同:第二，model.state_dict()存储的是该model中包含的所有layer中的所有参数；而model.named_parameters()则只保存可学习、可被更新的参数，model.buffer()中的参数不包含在model.named_parameters()中
    (3) 返回的值的require_grad属性不同:最后，model.state_dict()所存储的模型参数tensor的require_grad属性都是False，而model.named_parameters()的require_grad属性都是True

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





















"""
验证模型保存时保存的.pt文件大小是否改变，以及随着训练的进行，模型的参数数值是否改变。结果表明：

1. 一旦模型确定，则模型的pt大小是确定的，而不管模型的参数怎么变。
2. 随着训练过程的持续，模型的参数一直在变。
3. 随着训练过程的推荐，冻结的那些层的参数不会改变。
"""

import sys,os
import torch
from torch.autograd import Variable


import torch.nn as nn
import imageio


import matplotlib
matplotlib.use('TkAgg')


import torch.optim as optim





#===================================================================================
# 测试在init和forward部分，模型的层的定义和调用对模型结构的关系
#===================================================================================

# 定义一个简单的网络
class net(nn.Module):
    def __init__(self, num_class=10):
        super(net, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, num_class)


    def forward(self, x):
        return self.fc2(self.fc1(x))

model = net()

print(f"模型结构为：\n{model}, \n模型参数为:\n ")
for name, param in  model.named_parameters():
    if param.requires_grad:
        #print(f"{name}: {param.size()}, {param.requires_grad} ")
        print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ")

# 模型结构为：
# net(
#   (fc1): Linear(in_features=8, out_features=4, bias=True)
#   (fc2): Linear(in_features=4, out_features=10, bias=True)
# ),
# 模型参数为:

# fc1.weight               : size=torch.Size([4, 8]), requires_grad=True
# fc1.bias                 : size=torch.Size([4]), requires_grad=True
# fc2.weight               : size=torch.Size([10, 4]), requires_grad=True
# fc2.bias                 : size=torch.Size([10]), requires_grad=True


#==================================================================================
# 定义一个简单的网络
class net1(nn.Module):
    def __init__(self, num_class=10):
        super(net1, self).__init__()
        self.fc2 = nn.Linear(4, num_class)
        self.fc1 = nn.Linear(8, 4)


    def forward(self, x):
        return self.fc2(self.fc1(x))


model = net1()

print(f"模型结构为：\n{model}, \n模型参数为:\n ")
for name, param in  model.named_parameters():
    if param.requires_grad:
        #print(f"{name}: {param.size()}, {param.requires_grad} ")
        print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ")

# 模型结构为：
# net(
#   (fc2): Linear(in_features=4, out_features=10, bias=True)
#   (fc1): Linear(in_features=8, out_features=4, bias=True)
# ),
# 模型参数为:

# fc2.weight               : size=torch.Size([10, 4]), requires_grad=True
# fc2.bias                 : size=torch.Size([10]), requires_grad=True
# fc1.weight               : size=torch.Size([4, 8]), requires_grad=True
# fc1.bias                 : size=torch.Size([4]), requires_grad=True

#==================================================================================
# 定义一个简单的网络
class net1(nn.Module):
    def __init__(self, num_class=10):
        super(net1, self).__init__()
        self.fc2 = nn.Linear(4, num_class)
        self.fc1 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(self.fc1(x))


model = net1()

print(f"模型结构为：\n{model}, \n模型参数为:\n ")
for name, param in  model.named_parameters():
    if param.requires_grad:
        #print(f"{name}: {param.size()}, {param.requires_grad} ")
        print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ")

# 模型结构为：
# net1(
#   (fc2): Linear(in_features=4, out_features=10, bias=True)
#   (fc1): Linear(in_features=8, out_features=4, bias=True)
#   (fc3): Linear(in_features=8, out_features=4, bias=True)
# ),
# 模型参数为:

# fc2.weight               : size=torch.Size([10, 4]), requires_grad=True
# fc2.bias                 : size=torch.Size([10]), requires_grad=True
# fc1.weight               : size=torch.Size([4, 8]), requires_grad=True
# fc1.bias                 : size=torch.Size([4]), requires_grad=True
# fc3.weight               : size=torch.Size([4, 8]), requires_grad=True
# fc3.bias                 : size=torch.Size([4]), requires_grad=True

#==================================================================================
# 定义一个简单的网络
class net1(nn.Module):
    def __init__(self, num_class=10):
        super(net1, self).__init__()
        self.fc2 = nn.Linear(4, num_class)
        self.fc1 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(self.fc1(x))


model = net1()

for name, param in model.named_parameters():
    if "fc1" in name:
        param.requires_grad = False


print(f"模型结构为：\n{model}, \n模型参数为:\n ")
for name, param in  model.named_parameters():
    print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ")

# 模型结构为：
# net1(
#   (fc2): Linear(in_features=4, out_features=10, bias=True)
#   (fc1): Linear(in_features=8, out_features=4, bias=True)
#   (fc3): Linear(in_features=8, out_features=4, bias=True)
# ),
# 模型参数为:

# fc2.weight               : size=torch.Size([10, 4]), requires_grad=True
# fc2.bias                 : size=torch.Size([10]), requires_grad=True
# fc1.weight               : size=torch.Size([4, 8]), requires_grad=False
# fc1.bias                 : size=torch.Size([4]), requires_grad=False
# fc3.weight               : size=torch.Size([4, 8]), requires_grad=True
# fc3.bias                 : size=torch.Size([4]), requires_grad=True


#  由以上几个例子可见，模型的结构只由模型定义的顺序决定，与模型层的调用先后没关系，即使某层定义了，没被调用，也会存在于模型结构中。


#===================================================================================
## 测试保存模型，模型大小是否改变
#===================================================================================
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


for epoch in range(1000):
    x = torch.randn((3, 8))
    label = torch.randint(0,10,[3]).long()

    output = model(x)
    #print(f"epoch = {epoch}, x.shape = {x.shape}, output.shape = {output.shape}")
    loss = loss_fn(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"epoch = {epoch+1}, model.fc1.weight = \n{model.fc1.weight}\n")
        #time.sleep(2)
        PATH = "/home/jack/snap/model/{}_epoch.pt".format(epoch+1)
        torch.save(model.state_dict(), PATH)


print(f"\n\n")
model1 = net()
path1 = "/home/jack/snap/model/{}_epoch.pt".format(20)
model1.load_state_dict(torch.load(path1))
print(f"model1.fc1.weight = {model1.fc1.weight}", )
#print(f"model1.fc2.weight = {model1.fc2.weight}\n",)



model2 = net()
path2 = "/home/jack/snap/model/{}_epoch.pt".format(100)
model2.load_state_dict(torch.load(path2))
print(f"model2.fc1.weight = {model2.fc1.weight}", )
#print(f"model2.fc2.weight = {model2.fc2.weight}\n",)




#===================================================================================
# 测试训练过程参数是否改变以及怎么冻结参数
#===================================================================================

#===================================================================================
# 情况一：不冻结参数时
#===================================================================================
model = net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)  # 传入的是所有的参数

# 训练前的模型参数
print(f"model.fc1.weight = {model.fc1.weight}", )
print(f"model.fc2.weight = {model.fc2.weight}\n", )



for epoch in range(1000):
    x = torch.randn((3, 8))
    label = torch.randint(0,10,[3]).long()

    output = model(x)
    # print(f"epoch = {epoch}, x.shape = {x.shape}, output.shape = {output.shape}")
    loss = loss_fn(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# 训练后的模型参数
print(f"model.fc1.weight = {model.fc1.weight}", )
print(f"model.fc2.weight = {model.fc2.weight}\n",)




#===================================================================================
# 情况二：采用方式一冻结fc1层时
# 方式一
#===================================================================================


model = net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)  # 传入的是所有的参数

# 训练前的模型参数
print(f"model.fc1.weight = {model.fc1.weight}", )
print(f"model.fc2.weight = {model.fc2.weight}\n", )


for name, param in model.named_parameters():
    if "fc1" in name:
        param.requires_grad = False



for epoch in range(1000):
    x = torch.randn((3, 8))
    label = torch.randint(0,10,[3]).long()

    output = model(x)
    # print(f"epoch = {epoch}, x.shape = {x.shape}, output.shape = {output.shape}")
    loss = loss_fn(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch+1) % 100 == 0:
        #print(f"epoch = {epoch+1}, model.fc1.weight = \n{model.fc1.weight}\n")
        #time.sleep(2)
        PATH = "/home/jack/snap/model/{}_epoch.pt".format(epoch+1)
        torch.save(model.state_dict(), PATH)


print(f"\n\n")
model1 = net()
path1 = "/home/jack/snap/model/{}_epoch.pt".format(100)
model1.load_state_dict(torch.load(path1))
print(f"model1.fc1.weight = {model1.fc1.weight}", )
print(f"model1.fc2.weight = {model1.fc2.weight}\n\n", )
#print(f"model1.fc2.weight = {model1.fc2.weight}\n",)



model2 = net()
path2 = "/home/jack/snap/model/{}_epoch.pt".format(1000)
model2.load_state_dict(torch.load(path2))
print(f"model2.fc1.weight = {model2.fc1.weight}", )
print(f"model2.fc2.weight = {model2.fc2.weight}", )
#print(f"model2.fc2.weight = {model2.fc2.weight}\n",)



#===================================================================================
# 情况二：采用方式一冻结fc1层时
# 方式一
#===================================================================================


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

# 训练前的模型参数
print(f"model.fc1.weight = {model.fc1.weight}", )
print(f"model.fc2.weight = {model.fc2.weight}\n", )




model.train()
torch.set_grad_enabled(True)

for name, param in model.named_parameters():
    if "fc1" in name:
        param.requires_grad = False

for epoch in range(1000):
    x = torch.randn((3, 8))
    label = torch.randint(0,10,[3]).long()

    output = model(x)
    # print(f"epoch = {epoch}, x.shape = {x.shape}, output.shape = {output.shape}")
    loss = loss_fn(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch+1) % 100 == 0:
        #print(f"epoch = {epoch+1}, model.fc1.weight = \n{model.fc1.weight}\n")
        #time.sleep(2)
        PATH = "/home/jack/snap/model/{}_epoch.pt".format(epoch+1)
        torch.save(model.state_dict(), PATH)


print(f"\n\n")
model1 = net()
path1 = "/home/jack/snap/model/{}_epoch.pt".format(100)
model1.load_state_dict(torch.load(path1))
print(f"model1.fc1.weight = {model1.fc1.weight}", )
print(f"model2.fc2.weight = {model1.fc2.weight}\n\n", )
#print(f"model1.fc2.weight = {model1.fc2.weight}\n",)



model2 = net()
path2 = "/home/jack/snap/model/{}_epoch.pt".format(1000)
model2.load_state_dict(torch.load(path2))
print(f"model2.fc1.weight = {model2.fc1.weight}", )
print(f"model2.fc2.weight = {model2.fc2.weight}", )
#print(f"model2.fc2.weight = {model2.fc2.weight}\n",)



#===================================================================================
# 情况三：采用方式二冻结fc1层时
# 方式二
#===================================================================================

model = net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc2.parameters(), lr=1e-2)  # 优化器只传入fc2的参数


# 训练前的模型参数
print(f"model.fc1.weight = {model.fc1.weight}", )
print(f"model.fc2.weight = {model.fc2.weight}\n", )

for epoch in range(1000):
    x = torch.randn((3, 8))
    label = torch.randint(0,3,[3]).long()
    output = model(x)

    loss = loss_fn(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch+1) % 100 == 0:
        #print(f"epoch = {epoch+1}, model.fc1.weight = \n{model.fc1.weight}\n")
        #time.sleep(2)
        PATH = "/home/jack/snap/model/{}_epoch.pt".format(epoch+1)
        torch.save(model.state_dict(), PATH)


print(f"\n\n")
model1 = net()
path1 = "/home/jack/snap/model/{}_epoch.pt".format(100)
model1.load_state_dict(torch.load(path1))
print(f"model1.fc1.weight = {model1.fc1.weight}", )
print(f"model2.fc1.weight = {model2.fc1.weight}\n\n", )
#print(f"model1.fc2.weight = {model1.fc2.weight}\n",)



model2 = net()
path2 = "/home/jack/snap/model/{}_epoch.pt".format(1000)
model2.load_state_dict(torch.load(path2))
print(f"model2.fc1.weight = {model2.fc1.weight}", )
print(f"model2.fc2.weight = {model2.fc2.weight}", )
#print(f"model2.fc2.weight = {model2.fc2.weight}\n",)
