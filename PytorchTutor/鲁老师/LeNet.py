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
import os , sys
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lrs
import collections
import matplotlib.pyplot as plt
import mlutils.pytorch as mlutils
import argparse

#  nn.Conv2d的输入必须为4维的(N,C,H,W)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        # input shape: 1 * 28 * 28
        self.conv = nn.Sequential(
            # conv layer 1
            # add padding: 28 * 28 -> 32 * 32
            # conv: 1 * 32 * 32 -> 6 * 28 * 28
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.Sigmoid(),
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


def train_cjj(net, train_iter, test_iter, batch_size, optimizer, loss, num_epochs, device=mlutils.try_gpu()):

    net = net.to(device)
    
    timer = mlutils.Timer_lu()

    print("training on", device)
    print(f"len(train_iter) = {len(train_iter)}")

    for epoch in range(num_epochs):
        metric = mlutils.Accumulator(3)
        #print(f"len(train_iter) = {len(train_iter)}\n")
        # len(train_iter) = 235
        print(f"\nEpoch = {epoch}")

        for batch, (X, y) in enumerate(train_iter):
            timer.start()
            X = X.to(device)
            y = y.to(device)
            #print(f"X.requires_grad = {X.requires_grad}, y.requires_grad = {y.requires_grad}")
            y_hat = net(X)
            l = loss(y_hat, y)  # l = 2.3009374141693115
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            with torch.no_grad():
                acc = mlutils.accuracy(y_hat, y)
                metric.add(l * X.shape[0], acc, X.shape[0])

            ttmp = timer.stop()
            # metric[0] = l * X.shape[0], metric[2] = X.shape[0]
            train_l = metric[0]/metric[2]
            # metric[1] = number of correct predictions, metric[2] = X.shape[0]
            train_acc = metric[1]/metric[2]
            if (batch+1)%200 == 0:
                print("    Epoch:%d/%d, batch:%d/%d, loss:%.3f, acc:%.3f, time:%.3f(min)"% (epoch+1, num_epochs, batch+1, len(train_iter), l, acc, ttmp/60.0))
        test_acc = mlutils.evaluate_accuracy_gpu(net, test_iter)
        #if epoch % 1 == 0:
        print(f'epoch {epoch + 1} : loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    # after training, calculate images/sec
    # variable `metric` is defined in for loop, but in Python it can be referenced after for loop
    print(f"total training time {timer.sum()/60.0:.2f}(min), {metric[2] * num_epochs / timer.sum():.2f} images/sec on {str(device)}")


def main(args):

    net = LeNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss = torch.nn.CrossEntropyLoss()
    # load data
    train_iter, test_iter = mlutils.load_data_fashion_mnist(batch_size=args.batch_size)
    # train
    train_cjj(net, train_iter, test_iter, args.batch_size, optimizer, loss, args.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    main(args)
    
    











































































































































































































































































































































































































































































































































































