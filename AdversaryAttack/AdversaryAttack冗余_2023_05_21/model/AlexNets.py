#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:56:49 2022

@author: jack

https://lulaoshi.info/machine-learning/convolutional/alexnet
https://github.com/luweizheng/machine-learning-notes/tree/master/neural-network/cnn

"""

import pandas as pd
import numpy as np
import torch, torchvision, sys
from torch import nn, optim
import torch.nn.functional as F
import argparse
import os
import time
sys.path.append("..") 
import mlutils.pytorch as mlutils

ind = 1

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # convolution layer will change input shape into: floor((input_shape - kernel_size + padding + stride) / stride)
        # input shape: 1 * 224 * 224
        # convolution part
        self.conv = nn.Sequential(
            # conv layer 1
            # floor((224 - 11 + 2 + 4) / 4) = floor(54.75) = 54
            # conv: 1 * 224 * 224 -> 96 * 54 * 54 
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            # floor((54 - 3 + 2) / 2) = floor(26.5) = 26
            # 96 * 54 * 54 -> 96 * 26 * 26
            nn.MaxPool2d(kernel_size=3, stride=2), 
            # conv layer 2: decrease kernel size, add padding to keep input and output size same, increase channel number
            # floor((26 - 5 + 4 + 1) / 1) = 26
            # 96 * 26 * 26 -> 256 * 26 * 26
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            # floor((26 - 3 + 2) / 2) = 12
            # 256 * 26 * 26 -> 256 * 12 * 12
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 3 consecutive conv layer, smaller kernel size
            # floor((12 - 3 + 2 + 1) / 1) = 12
            # 256 * 12 * 12 -> 384 * 12 * 12
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            # 384 * 12 * 12 -> 384 * 12 * 12
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            # 384 * 12 * 12 -> 256 * 12 * 12
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            # floor((12 - 3 + 2) / 2) = 5
            # 256 * 5 * 5
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # fully connect part 
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            # Use the dropout layer to mitigate overfitting
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # Output layer. 
            # the number of classes in Fashion-MNIST is 10
            nn.Linear(4096, 10),
        )

    def forward(self, img):
         # img.shape = torch.Size([128, 1, 224, 224])
         # ind = 1
         feature = self.conv(img)  
         # feature.shape = torch.Size([128, 256, 5, 5])
         
         output = self.fc(feature.view(img.shape[0], -1))
         # feature.view(img.shape[0], -1).shape = torch.Size([128, 6400])
         # output.shape = torch.Size([128, 10])
         
         #if ind == 1:
         #     print(f"img.shape = {img.shape}")
         #     print(f"feature.shape = {feature.shape}")
         #     print(f"feature.view(img.shape[0], -1).shape = {feature.view(img.shape[0], -1).shape}")
         #     print(f"output.shape = {output.shape}")
         #ind += 1
         return output




#load_data_fashion_mnist()方法定义了读取数据的方式，Fashion-MNIST原来是1 × 28 × 28的大小。resize在原图的基础上修改了图像的大小，可以将图片调整为我们想要的大小。

def train(net, train_iter, test_iter, batch_size, optimizer, num_epochs, device=mlutils.try_gpu()):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    timer = mlutils.Timer()
    # in one epoch, it will iterate all training samples
    for epoch in range(num_epochs):
        print(f"len(train_iter) = {len(train_iter)}\n")
        # Accumulator has 3 parameters: (loss, train_acc, number_of_images_processed)
        metric = mlutils.Accumulator(3)
        # all training samples will be splited into batch_size
        for X, y in train_iter:
            timer.start()
            # set the network in training mode
            net.train()
            # move data to device (gpu)
            X = X.to(device)  # X.shape = torch.Size([128, 1, 224, 224])
            y = y.to(device)  # y.shape = torch.Size([128])
            y_hat = net(X)    # torch.Size([128, 10])
            l = loss(y_hat, y)   # tensor(2.3036, grad_fn=<NllLossBackward0>)
            #print("1\n")
            #print(f"X.shape={X.shape}, y.shape={y.shape},y_hat.shape={y_hat.shape}")
            #print(f"y = {y}")
            """
            """
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            #print("2\n")
            with torch.no_grad():
                # all the following metrics will be accumulated into variable `metric`
                metric.add(l * X.shape[0], mlutils.accuracy(y_hat, y), X.shape[0])
            #print("3\n")
            timer.stop()
            # metric[0] = l * X.shape[0], metric[2] = X.shape[0]
            train_l = metric[0]/metric[2]
            # metric[1] = number of correct predictions, metric[2] = X.shape[0]
            train_acc = metric[1]/metric[2]
            print("4\n")
        test_acc = mlutils.evaluate_accuracy_gpu(net, test_iter)
        #if epoch % 1 == 0:
        print(f'epoch {epoch + 1} : loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    # after training, calculate images/sec
    # variable `metric` is defined in for loop, but in Python it can be referenced after for loop
    print(f'total training time {timer.sum():.2f}, {metric[2] * num_epochs / timer.sum():.1f} images/sec ' f'on {str(device)}')

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
            if (batch+1)%10 == 0:
                print("    Epoch:%d/%d, batch:%3d/%d, loss:%.3f, acc:%.3f, time:%.3f(min)"% (epoch+1, num_epochs, batch+1, len(train_iter), l, acc, ttmp/60.0))
        test_acc = mlutils.evaluate_accuracy_gpu(net, test_iter)
        #if epoch % 1 == 0:
        print(f'epoch {epoch + 1} : loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    # after training, calculate images/sec
    # variable `metric` is defined in for loop, but in Python it can be referenced after for loop
    print(f"total training time {timer.sum()/60.0:.2f}(min), {metric[2] * num_epochs / timer.sum():.2f} images/sec on {str(device)}")



def main(args):

    net = AlexNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss = torch.nn.CrossEntropyLoss()
    # load data
    train_iter, test_iter =  mlutils.load_data_fashion_mnist(batch_size=args.batch_size, resize=224)
    # train
    train_cjj(net, train_iter, test_iter, args.batch_size, optimizer,loss, args.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    main(args)










































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































