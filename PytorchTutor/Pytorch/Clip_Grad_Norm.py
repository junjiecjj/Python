#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:48:08 2023

@author: jack

此文件是详细测试 torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm, norm_type=2)的细节：

    model.parameters()的参数的梯度是矩阵序列，且每个矩阵的大小不等, [W1, W2, W3, Wn]， Wi是矩阵；
    1. 首先，会计算每个模型参数梯度矩阵的L2范数，得到[w1, w2, w3, wn], 其中wi是Wi的L2范数，即矩阵元素绝对值的平方和再开平方, 得到的wi是数字;
    2. 再求[w1, w2, w3, wn] 的向量范数,将结果定义为total_norm;
    3. 最后定义了一个“裁剪系数”变量clip_coef，为传入参数max_norm和total_norm的比值（+1e-6防止分母为0的情况）。如果max_norm > total_norm，即没有溢出预设上限，则不对梯度进行修改。反之则以clip_coef为系数对全部梯度进行惩罚，使最后的全部梯度范数归一化至max_norm的值

    注意该方法返回了一个 total_norm，实际应用时可以通过该方法得到网络参数梯度的范数，以便确定合理的max_norm值。
"""

def Quantilize(params, G = None, B = 8):
    if type(B) != int or (G != None and type(G) != int):
        raise ValueError("B 必须是 int, 且 G 不为None时也必须是整数!!!")
    if G == None:
        G =  2**B - 1
    params = torch.clamp(torch.round(params * G), min = -2**(B-1), max = 2**(B-1) - 1, )
    return params




import sys,os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import torch.optim as optim
import copy


# 定义一个简单的网络
class net(nn.Module):
    def __init__(self, num_class=10):
        super(net, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, num_class)

    def forward(self, x):
        return self.fc2(self.fc1(x))

model = net()

orig_params = {}
for key, var in model.state_dict().items():
    orig_params[key] = var.clone()#.cpu()   # .detach().cpu().numpy()
    print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()} \n  {var}" )


tmp_param = {}
for i , (key, val) in enumerate(orig_params.items()):
    tmp_param[key] = torch.ones_like(val) + i



#====================================================================================================================

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)  # 传入的是所有的参数
x = torch.randn((3, 8))
label = torch.randint(0,10,[3]).long()

for key, var in model.named_parameters():
    print(f"0: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")


output = model(x)
#print(f"epoch = {epoch}, x.shape = {x.shape}, output.shape = {output.shape}")
loss = loss_fn(output, label)

for key, var in model.named_parameters():
    print(f"1: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")
print("\n\n")

## 优化器梯度清零
optimizer.zero_grad()
for key, var in model.named_parameters():
    print(f"2: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")
print("\n\n")

## 梯度反向传播
loss.backward()
for key, var in model.named_parameters():
    print(f"3: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")
print("\n\n")

## 梯度反向传播后计算梯度的L2范数: 平方和开平方, 方法一
parameters = [p.grad.clone() for p in model.parameters() if p.grad is not None]
grad_norm = []
for gd in parameters:
    if len(gd.size()) == 2:
        grad_norm.append( torch.linalg.matrix_norm(gd, ord = 'fro').item() )
    else:
        grad_norm.append( torch.linalg.vector_norm(gd, ord = 2).item() )
total_norm = torch.linalg.vector_norm( torch.tensor(grad_norm), ord = 2,)
print(f"0:  total_norm = {total_norm}")

## 梯度反向传播后计算梯度的L2范数: 平方和开平方, 方法二
total_norm1 = torch.norm(torch.stack([torch.norm(p  , 2)  for p in parameters]), 2)
print(f"1:  total_norm1 = {total_norm1}")


torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
for key, var in model.named_parameters():
    print(f"4: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")
print("\n\n")

optimizer.step()
for key, var in model.named_parameters():
    print(f"5: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")
print("\n\n")


optimizer.zero_grad()
for key, var in model.named_parameters():
    print(f"6: {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var.data} ")
print("\n\n")


model.load_state_dict(orig_params)
print(f"model.state_dict() = {model.state_dict()} \n\n")



















