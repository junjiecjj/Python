#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 20:49:23 2022

@author: jack

https://lulaoshi.info/machine-learning/convolutional/lenet


"""



import pandas as pd
import numpy as np
import torch, torchvision 
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import os , sys
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lrs
import collections
import matplotlib.pyplot as plt

import argparse


sys.path.append("..")
from model  import common



# 定义LeNet模型
class LeNet_csdn(nn.Module):
    def __init__(self, args):
        super(LeNet_csdn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


#  nn.Conv2d的输入必须为4维的(N,C,H,W)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        # input shape: 1 * 28 * 28
        self.conv = nn.Sequential(
            # conv layer 1
            # add padding: 28 * 28 -> 32 * 32
            # conv: 1 * 32 * 32 -> 6 * 28 * 28
            nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=5, padding=2), nn.Sigmoid(),
            # 6 * 28 * 28 -> 6 * 14 * 14
            nn.MaxPool2d(kernel_size=2, stride=2), # kernel_size, stride
            # conv layer 2
            # 6 * 14 * 14 -> 16 * 10 * 10 
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.Sigmoid(),
            # 16 * 10 * 10 -> 16 * 5 * 5
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            # full connect layer 1
            nn.Linear(in_features=16 * 5 * 5, out_features=120), nn.Sigmoid(),
            # full connect layer 2
            nn.Linear(in_features=120, out_features=84), nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        """
        img.shape = torch.Size([256, 1, 28, 28])
        feature.shape = torch.Size([256, 16, 5, 5])
        output.shape = torch.Size([256, 10])
        feature.view(img.shape[0], -1).shape = torch.Size([256, 400])
        """
        return output

# save to mlutils so that other program can use it
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train_cjj(net, train_iter, test_iter, batch_size, optimizer, loss, num_epochs, device= try_gpu()):

    net = net.to(device)

    timer = common.Timer_lu()

    print("training on", device)
    print(f"len(train_iter) = {len(train_iter)}")

    for epoch in range(num_epochs):
        metric = common.Accumulator(3)
        #print(f"len(train_iter) = {len(train_iter)}\n")
        # len(train_iter) = 235
        print(f"\nEpoch = {epoch}")

        for batch, (X, y) in enumerate(train_iter):
            timer.start()
            X = X.to(device)
            y = y.to(device)
            # print(f"X.shape = {X.shape}, y.shape = {y.shape}")
            y_hat = net(X)
            l = loss(y_hat, y)  # l = 2.3009374141693115
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # print(f"y.shape = {y.shape}, y_hat.shape = {y_hat.shape}")
            with torch.no_grad():
                acc = common.accuracy(y_hat, y)
                metric.add(l * X.shape[0], acc, X.shape[0])

            ttmp = timer.stop()

            if (batch+1)%200 == 0:
                print("    Epoch:%d/%d, batch:%d/%d, loss:%.3f, acc:%.3f, time:%.3f(min)"% (epoch+1, num_epochs, batch+1, len(train_iter), l, acc, ttmp/60.0))
        test_acc = common.evaluate_accuracy_gpu(net, test_iter)
        train_l = metric[0]/metric[2]
        train_acc = metric[1]/metric[2]

        print(f'epoch {epoch + 1} : loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')

    print(f"total training time {timer.sum()/60.0:.2f}(min), {metric[2] * num_epochs / timer.sum():.2f} images/sec on {str(device)}")


def main(args):
    torch.manual_seed(2)
    net = LeNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss = torch.nn.CrossEntropyLoss()
    # load data
    train_iter, test_iter = common.load_data_fashion_mnist(batch_size=args.batch_size)
    # train
    train_cjj(net, train_iter, test_iter, args.batch_size, optimizer, loss, args.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    main(args)
    
    











































































































































































































































































































































































































































































































































































