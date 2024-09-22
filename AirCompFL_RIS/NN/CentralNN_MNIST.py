#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat 2024/08/24

@author: Junjie Chen
"""


##  系统库
import os
# import sys
import torch
import torchvision

from torch import nn
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')

import socket, getpass
# 获取当前系统主机名
host_name = socket.gethostname()
# 获取当前系统用户名
user_name = getpass.getuser()
# 获取当前系统用户目录
home = os.path.expanduser('~')

##  自己编写的库
# checkpoint
import Utility
from MetricsLog import Accumulator
import models

#===============================================================================================
# 设置随机数种子
Utility.set_random_seed(9999, )
Utility.set_printoption(3)
device = torch.device('cuda' if torch.cuda.is_available()  else "cpu")

def data_tf_cnn_mnist(x):
    ## 1
    x = torchvision.transforms.ToTensor()(x)
    x = (x - 0.5) / 0.5
    x = x.reshape((-1, 28, 28))
    return x

def validata(model, dataloader, device = None):
    model.eval()
    if not device:
        device = next(model.parameters()).device
    metric =  Accumulator(2)
    with torch.no_grad():
        for X, y in dataloader:
            # print(f"X.shape = {X.shape}, y.shape = {y.shape}, size(y) = {size(y)}/{y.size(0)}") # X.shape = torch.Size([128, 1, 28, 28]), y.shape = torch.Size([128]), size(y) = 128
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            acc = (y_hat.argmax(axis=1) == y).sum().item()
            metric.add(acc, y.size(0))  # size(y)
    acc = metric[0] / metric[1]
    return acc

trainset = torchvision.datasets.MNIST(root = home+'/FL_semantic/Data/', train = True, download = True, transform = data_tf_cnn_mnist)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True, pin_memory = 0, num_workers=  6, )
testset = torchvision.datasets.MNIST(root = home+'/FL_semantic/Data/', train = False, download = True, transform = data_tf_cnn_mnist)
testloader = torch.utils.data.DataLoader(testset, batch_size = 128, shuffle = False, num_workers = 6, )

# model = models.mnist_cnn().to(device)
model = models.Mnist_2NN().to(device)
# model = models.Mnist_MLP().to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, betas = (0.9, 0.999), eps = 1e-08,)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.004, momentum = 0.9, weight_decay = 0.0001 )
milestone = list(np.arange(10, 1000, 10))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestone, gamma = 0.96)
lossfn = torch.nn.CrossEntropyLoss(reduction='mean')

epochs = 50
for epoch in range(epochs):
    metric = Accumulator(4)
    model.train()
    lr = scheduler.get_last_lr()
    print(f"\nEpoch : {epoch+1}/{epochs}({100.0*(epoch+1)/epochs:0>5.2f}%), lr = {lr}")
    for batch, (X, y) in enumerate(trainloader):
        # print(f"1 {X.min()}, {X.max()}, {y.shape}, ")
        X, y = X.to(device), y.to(device)
        # print(f"2  {X.shape}, {y.shape}, ")
        y_hat = model(X)
        # print(f"3 {X.shape}, {y.shape}, {y_hat.shape}, {y_hat.min()}, {y_hat.max()}")
        loss = lossfn(y_hat, y)

        optimizer.zero_grad()       # 必须在反向传播前先清零。
        loss.backward()
        optimizer.step()
        # print(f"y.shape = {y.shape}, y_hat.shape = {y_hat.shape}")
        with torch.no_grad():
            acc = (y_hat.argmax(axis=1) == y).sum().item()
        metric.add(loss.item(), acc, X.size(0), 1)
        # 输出训练状态
        if (batch+1) % 100 == 0:
            frac2 = (batch + 1) / len(trainloader)
            print("    [epoch: {:*>5d}/{}, batch: {:*>5d}/{}({:0>6.2%})]\tLoss: {:.4f} \t  Train acc:{:4.2f} ".format(epoch+1, epochs, batch+1, len(trainloader), frac2, loss.item(), acc/X.shape[0] ))
    # 学习率递减
    # scheduler.step()
    # epoch 的平均 loss
    epoch_avg_loss = metric[0]/metric[3]
    # epoch 的 train data 正确率
    epoch_train_acc = metric[1]/metric[2]
    # test data accuracy
    test_acc = validata(model, testloader, device =  device)
    print(f"  Epoch: {epoch+1}/{epochs} | loss = {epoch_avg_loss:.4f} | train acc: {epoch_train_acc:.3f}, test acc: {test_acc:.3f} | \n")

### 保存网络中的参数,
# torch.save(model.state_dict(), "/home/jack/FL_semantic/LeNet_model/LeNet_Minst_classifier.pt")

print("\n#============ 训练完毕,  =================\n")
acc1 =  validata(model, testloader, device =  device)
print(f"train data acc = {acc1}")
acc2 =  validata(model, testloader, device =  device,)
print(f"test data acc = {acc2}")

# classifier = LeNets.LeNet_3().to(args.device)
# pretrained_model = "/home/jack/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_2023-06-01-22:20:58.pt"
# # 加载已经预训练的模型(没gpu没cuda支持的时候加载模型到cpu上计算)
# classifier.load_state_dict(torch.load(pretrained_model, map_location = args.device))




























































































































































































































































































































































































































































































































































































































