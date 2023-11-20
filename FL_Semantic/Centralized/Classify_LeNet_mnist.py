#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 20:43:54 2023

@author: jack
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
#===============================================================================================

# 设置随机数种子
Utility.set_random_seed(1,  deterministic = True, benchmark = True)
Utility.set_printoption(3)

class Accumulator(object):
    """ For accumulating sums over n variables. """
    def __init__(self,  n):
        self.data = [0.0] * n
        return
    def add(self, *Args):
        self.data = [a + float(b) for a, b in zip(self.data, Args)]
        return
    def reset(self):
        self.data = [0.0] * len(self.data)
        return
    def __getitem__(self, idx):
        return self.data[idx]


class LeNet_3(nn.Module):
    def __init__(self, ):
        super(LeNet_3, self).__init__()
        # input shape: 1 * 1 * 28 * 28
        self.conv = nn.Sequential(
            ## conv layer 1
            ## conv: 1, 28, 28 -> 10, 24, 24
            nn.Conv2d(1, 10, kernel_size = 5),
            ## 10, 24, 24 -> 10, 12, 12
            nn.MaxPool2d(kernel_size = 2, ),
            nn.ReLU(),

            ## 10, 12, 12 -> 20, 8, 8
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            ## 20, 8, 8 -> 20, 4, 4
            nn.MaxPool2d(kernel_size = 2, ),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            ## full connect layer 1
            nn.Linear(320, 50), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.LogSoftmax(dim = 1)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

def data_tf_cnn_mnist(x):
    ## 1
    x = torchvision.transforms.ToTensor()(x)
    x = (x - 0.5) / 0.5
    x = x.reshape((-1, 28, 28))

    # # 2
    # x = np.array(x, dtype='float32') / 255
    # x = (x - 0.5) / 0.5
    # x = x.reshape((1, 28, 28))  # ( 1, 28, 28)
    # x = torch.from_numpy(x)
    return x

trainset = torchvision.datasets.MNIST(root = home+'/SemanticNoise_AdversarialAttack/Data/',          # 表示 MNIST 数据的加载的目录
                                      train = True,                                                    # 表示是否加载数据库的训练集，false的时候加载测试集
                                      download = True,                                                 # 表示是否自动下载 MNIST 数据集
                                      transform = data_tf_cnn_mnist)                                   # 表示是否需要对数据进行预处理，none为不进行预处理
trainloader = torch.utils.data.DataLoader( trainset,
                                           batch_size = 128,
                                           shuffle = True,
                                           pin_memory = 0,
                                           num_workers=  6, )
testset = torchvision.datasets.MNIST(root = home+'/SemanticNoise_AdversarialAttack/Data/',      # 表示 MNIST 数据的加载的目录
                                    train = False,                                              # 表示是否加载数据库的训练集，false的时候加载测试集
                                    download = True,                                            # 表示是否自动下载 MNIST 数据集
                                    transform = data_tf_cnn_mnist)                              # 表示是否需要对数据进行预处理，none为不进行预处理
testloader = torch.utils.data.DataLoader(  testset,
                                           batch_size = 128,
                                           shuffle = False,
                                           num_workers = 6, )


## 如果不想用CPU且存在GPU, 则用GPU; 否则用CPU;
device = torch.device('cuda' if torch.cuda.is_available()  else "cpu")
model =  LeNet_3().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.002, betas=(0.9, 0.999), eps=1e-08,)
milestone = list(np.arange(1,1000,10))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestone, gamma = 0.96)
lossfn = torch.nn.CrossEntropyLoss(reduction='sum')


class LeNetMinst_PreTrain(object):
    def __init__(self, train_loader, test_loader, model, loss_fn, optim):
        # self.scale = args.scale
        # #print(f"trainer  self.scale = {self.scale} \n")
        self.trainloader = train_loader
        self.testloader  = test_loader
        self.net = model
        self.lossfn = loss_fn
        self.optim = optim
        return

    def train(self):
        epochs = 1000
        for epoch in range(epochs):
            metric = Accumulator(3)
            self.net.train()
            print(f"\nEpoch : {epoch+1}/{epochs}({100.0*(epoch+1)/epochs:0>5.2f}%)")
            for batch, (X, y) in enumerate(self.trainloader):
                self.net.zero_grad()
                # print(f"1 {X.min()}, {X.max()}, {y.shape}, ")
                X, y = X.to(device), y.to(device)
                # print(f"2  {X.shape}, {y.shape}, ")
                y_hat = self.net(X)
                # print(f"3 {X.shape}, {y.shape}, {y_hat.shape}, {y_hat.min()}, {y_hat.max()}")
                loss = self.lossfn(y_hat, y)

                self.optim.zero_grad()       # 必须在反向传播前先清零。
                loss.backward()
                self.optim.step()
                # print(f"y.shape = {y.shape}, y_hat.shape = {y_hat.shape}")
                with torch.no_grad():
                    acc = (y_hat.argmax(axis=1) == y).sum().item()
                metric.add(loss.item(), acc, X.size(0))
                # 输出训练状态
                if (batch+1) % 100 == 0:
                    frac1 = (epoch + 1) /  epochs
                    frac2 = (batch + 1) / len(self.trainloader)
                    print("    [epoch: {:*>5d}/{}({:0>6.2%}), batch: {:*>5d}/{}({:0>6.2%})]\tLoss: {:.4f} \t  Train acc:{:4.2f} ".format(epoch+1, epochs, frac1, batch+1, len(self.trainloader), frac2, loss.item(), acc/X.shape[0] ))
            # 学习率递减
            scheduler.step()
            # epoch 的平均 loss
            epoch_avg_loss = metric[0]/metric[2]
            # epoch 的 train data 正确率
            epoch_train_acc = metric[1]/metric[2]
            # test data accuracy
            test_acc = self.validata(self.net, self.testloader, device =  device)
            print(f"  Epoch: {epoch+1}/{epochs}({(epoch+1)*100.0/epochs:5.2f}%) | loss = {epoch_avg_loss:.4f} | train acc: {epoch_train_acc:.3f}, test acc: {test_acc:.3f} | \n")
        ### 保存网络中的参数, 速度快，占空间少
        # torch.save(model.state_dict(), f"/home/jack/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_{tm.start_str}.pt")   # 训练和测试都归一化

        print("\n#============ 训练完毕,  =================\n")

        acc1 = self.validata(self.net, self.trainloader,)
        print(f"train data acc = {acc1}")
        acc2 = self.validata(self.net, self.testloader,)
        print(f"test data acc = {acc2}")
        return acc1, acc2


    def validata(self, model, dataloader, device = None):
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

p = LeNetMinst_PreTrain(trainloader, testloader, model, lossfn, optimizer)
ac1, ac2 = p.train()
print(f"train acc = {ac1}, test acc = {ac2}\n")





# classifier = LeNets.LeNet_3().to(args.device)
# pretrained_model = "/home/jack/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_2023-06-01-22:20:58.pt"
# # 加载已经预训练的模型(没gpu没cuda支持的时候加载模型到cpu上计算)
# classifier.load_state_dict(torch.load(pretrained_model, map_location = args.device))































































































































































































































































































































































































































































































































































































