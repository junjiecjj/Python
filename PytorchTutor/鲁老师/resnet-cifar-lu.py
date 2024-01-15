#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 20:49:23 2022

@author: jack

https://lulaoshi.info/machine-learning/convolutional/resnet

MNIST：10类共70000张28x28的0-9的手写数字图片，每类有7000张图片，其中60000张为训练集，10000张为测试集。

CIFAR10：10类共60000张32x32彩色图像，每类有6000张图片，其中50000个训练集，10000张为测试集。 

CIFAR100：100类共60000张32x32彩色图像，每类有600张图片，其中50000个训练集，10000张为测试集，共100类分为20个超类，所以每个图像都带有一个“精细”标签（它所属的类）和一个“粗”标签（它所属的超类）。



"""

import time
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import torchvision
import argparse
import sys
sys.path.append("..") 




class Accumulator:
    """For accumulating sums over n variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)


def try_gpu(i=0):
    """Return gpu device if exists, otherwise return cpu device."""
    # if torch.cuda.device_count() >= i + 1:
    #     return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    # Set the model to evaluation mode
    net.eval()  
    if not device:
        device = next(iter(net.parameters())).device
        print(f"device = {device}")
    # Accumulator has 2 parameters: (number of correct predictions, number of predictions)
    metric = Accumulator(2)
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]


class Residual(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, in_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        # 1×1 convolutional layer can change output channel number
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    """return multiple resnet blocks"""
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

def resnet18():
    """return resnet"""
    # convolutional and max pooling layer 
    b1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # resnet block
    b2 = nn.Sequential(*resnet_block(in_channels=64, num_channels=64, num_residuals=2, first_block=True))
    b3 = nn.Sequential(*resnet_block(in_channels=64, num_channels=128, num_residuals=2))
    b4 = nn.Sequential(*resnet_block(in_channels=128, num_channels=256, num_residuals=2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

    return net

def load_data_cifar10(batch_size, resize=None, root='~/公共的/MLData/CIFAR10'):
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True,  transform=transform_train)
    cifar_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True,  transform=transform_test)

    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def train(net, train_iter, test_iter, batch_size, optimizer, num_epochs, device=try_gpu()):
    net = net.to(device)
    print("training on", device)
    print(f"len(train_iter) = {len(train_iter)}")
    loss = torch.nn.CrossEntropyLoss()
    timer = Timer()
    # in one epoch, it will iterate all training samples
    for epoch in range(num_epochs):
        # Accumulator has 3 parameters: (loss, train_acc, number_of_images_processed)
        metric = Accumulator(3)
        # all training samples will be splited into batch_size
        print(f"Epoch = {epoch}\n")
        for batch, (X, y) in enumerate(train_iter):
            timer.start()
            # set the network in training mode
            net.train()
            # move data to device (gpu)
            X = X.to(device)
            y = y.to(device)
            #print(f"y = {y}")
            y_hat = net(X)
            l = loss(y_hat, y)
            #print(f"X.shape = {X.shape}, y.shape = {y.shape}.y_hat.shape = {y_hat.shape}, l = {l}")
            # X.shape = torch.Size([128, 1, 96, 96]), y.shape = torch.Size([128]).y_hat.shape = torch.Size([128, 10]), l = 2.303614854812622
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            with torch.no_grad():
                # all the following metrics will be accumulated into variable `metric`
                acc = accuracy(y_hat, y)
                metric.add(l * X.shape[0], acc, X.shape[0])
                #print(f"metric.data = {metric.data}")
            timer.stop()
            # metric[0] = l * X.shape[0], metric[2] = X.shape[0]
            train_l = metric[0]/metric[2]
            # metric[1] = number of correct predictions, metric[2] = X.shape[0]
            train_acc = metric[1]/metric[2]
            print("    Epoch:%d/%d, batch:%d/%d, loss:%.3f, acc:%.3f"% (epoch, num_epochs, batch, len(train_iter), l, acc))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        if epoch % 1 == 0:
            print(f'epoch {epoch + 1} : loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    # after training, calculate images/sec
    # variable `metric` is defined in for loop, but in Python it can be referenced after for loop
    print(f'total training time {timer.sum():.2f}, {metric[2] * num_epochs / timer.sum():.1f} images/sec ' f'on {str(device)}')




def main(args):
    net = resnet18()
    #print(f"net = \n{net}\n")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # load data
    train_iter, test_iter = load_data_cifar10(batch_size=args.batch_size)
    # X.shape = torch.Size([128, 3, 32, 32]), y.shape = torch.Size([128]).y_hat.shape = torch.Size([128, 10])
    # train
    train(net, train_iter, test_iter, args.batch_size, optimizer, args.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    args = parser.parse_args()
    main(args)
