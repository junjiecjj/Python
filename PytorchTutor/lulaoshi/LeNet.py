#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 20:49:23 2022

@author: jack

https://lulaoshi.info/machine-learning/convolutional/lenet


"""



import pandas as pd
import numpy as np
import torch, torchvision, sys
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable

#  nn.Conv2d的输入必须为4维的(N,C,H,W)


#网络模型结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        # 输入 1 * 28 * 28
        self.conv = nn.Sequential(
            # 卷积层1
            # 在输入基础上增加了padding，28 * 28 -> 32 * 32
            # 1 * 32 * 32 -> 6 * 28 * 28
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.Sigmoid(),
            # 6 * 28 * 28 -> 6 * 14 * 14
            nn.MaxPool2d(kernel_size=2, stride=2), # kernel_size, stride
            # 卷积层2
            # 6 * 14 * 14 -> 16 * 10 * 10 
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.Sigmoid(),
            # 16 * 10 * 10 -> 16 * 5 * 5
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            # 全连接层1
            nn.Linear(in_features=16 * 5 * 5, out_features=120), nn.Sigmoid(),
            # 全连接层2
            nn.Linear(in_features=120, out_features=84), nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        # print(f"img.shape = {img.shape}\nfeature.shape = {feature.shape}\noutput.shape = {output.shape}\nfeature.view(img.shape[0], -1).shape = {feature.view(img.shape[0], -1).shape}")
        """
          img.shape = torch.Size([256, 1, 28, 28])
          feature.shape = torch.Size([256, 16, 5, 5])
          output.shape = torch.Size([256, 10])
          feature.view(img.shape[0], -1).shape = torch.Size([256, 400])
        """
        return output


#训练模型

#load_data_fashion_mnist()方法返回训练集和测试集。
def load_data_fashion_mnist(batch_size, resize=None, root='~/公共的/MLData/FashionMNIST'):
    """Use torchvision.datasets module to download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


#在训练过程中，我们希望看到每一轮迭代的准确度，构造一个evaluate_accuracy方法，计算当前一轮迭代的准确度（模型预测值与真实值之间的误差大小）：
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                # set the model to evaluation mode (disable dropout)
                net.eval() 
                # get the acc of this batch
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                # change back to train mode
                net.train() 

            n += y.shape[0]
    return acc_sum / n


#接着，我们可以构建一个train()方法，用来训练神经网络：
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def train(net, train_iter, test_iter, batch_size, optimizer, num_epochs, device=try_gpu()):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            # X.shape = torch.Size([256, 1, 28, 28]), y.shape = torch.Size([256])
            
            y_hat = net(X)      # y_hat.shape = torch.Size([256, 10])
            l = loss(y_hat, y)  # l = 2.3009374141693115
            l = Variable(l, requires_grad = True)
            # print(f"y_hat.shape = {y_hat.shape}, l.shape = {l.shape}")
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item() #  l.cpu().item() = 2.3009374141693115
            
            # (y_hat.argmax(dim=1) == y).sum().cpu().item() = 25
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        if epoch % 3 == 0:
            print(f'epoch {epoch + 1} : loss {train_l_sum / batch_count:.3f}, train acc {train_acc_sum / n:.3f}, test acc {test_acc:.3f}')


#在整个程序的主逻辑中，设置必要的参数，读入训练和测试数据并开始训练：
#def main():
batch_size = 256
lr, num_epochs = 0.9, 10

net = LeNet()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
# load data
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
# train
train(net, train_iter, test_iter, batch_size, optimizer, num_epochs)

















































































































































































































































































































































































































































































































































































