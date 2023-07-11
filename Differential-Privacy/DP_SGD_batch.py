#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:12:01 2023

@author: jack
"""



#  系统库
import numpy as np
import os, sys
import torch, torchvision
# from torch import nn
# import torch.nn.functional as F
# from torch.autograd import Variable
import torch.optim.lr_scheduler as lrs
import collections
import matplotlib.pyplot as plt
import argparse




# 初始化随机数种子
def set_random_seed(seed = 10, deterministic = False, benchmark = False):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
    return
#============================================================================================================================
#                                        记录当前每个epcoh的 相关指标, 记录历史
#============================================================================================================================

class TraRecorder(object):
    def __init__(self,  Len = 3,  name = "Train", compr = '', tra_snr = 'noiseless'):
        self.name =  name
        self.len = Len
        self.metricLog = np.empty((0, self.len))
        self.cn = self.__class__.__name__
        if compr != '' :
            self.title = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}} = {}\mathrm{{(dB)}}$'.format(compr, tra_snr)
            self.basename = f"{self.cn}_compr={compr:.1f}_trainSnr={tra_snr}(dB)"
        else:
            self.title = ""
            self.basename = f"{self.cn}"
        return

    def addlog(self, epoch):
        self.metricLog = np.append(self.metricLog , np.zeros( (1, self.len )), axis=0)
        self.metricLog[-1, 0] = epoch
        return

    def assign(self,  metrics = ''):
        if len(metrics) != self.len - 1:
            print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")
            raise ValueError("len is inconsistent")
        self.metricLog[-1, 1:] = metrics
        return

    def __getitem__(self, idx):
        return self.metricLog[-1, idx + 1]

    def save(self, path,   ):
        torch.save(self.metricLog, os.path.join(path, f"{self.basename}_DPSGD_batch.pt"))
        return


## model
class LeNet_3(torch.nn.Module):
    def __init__(self, ):
        super(LeNet_3, self).__init__()
        # input shape: 1 * 1 * 28 * 28
        self.conv = torch.nn.Sequential(
            ## conv layer 1
            ## conv: 1, 28, 28 -> 10, 24, 24
            torch.nn.Conv2d(1, 10, kernel_size = 5),
            ## 10, 24, 24 -> 10, 12, 12
            torch.nn.MaxPool2d(kernel_size = 2, ),
            torch.nn.ReLU(),
            ## 10, 12, 12 -> 20, 8, 8
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.Dropout2d(),
            ## 20, 8, 8 -> 20, 4, 4
            torch.nn.MaxPool2d(kernel_size = 2, ),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            ## full connect layer 1
            torch.nn.Linear(320, 50), torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(50, 10),
            torch.nn.LogSoftmax(dim = 1)
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def validata( model, dataloader, device = None):
    model.eval()
    if not device:
        device = next(model.parameters()).device

    acc = 0.0
    examples = 0.0
    with torch.no_grad():
        for X, label in dataloader:
            # print(f"X.shape = {X.shape}, y.shape = {y.shape}, size(y) = {size(y)}/{y.size(0)}") # X.shape = torch.Size([128, 1, 28, 28]), y.shape = torch.Size([128]), size(y) = 128
            X        = X.to(device)
            y_hat    = model(X).cpu().argmax(dim=1)
            acc      += (y_hat == label).float().sum().item()
            examples += X.size(0)
    avg_acc = acc/examples
    return avg_acc


set_random_seed()
parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--batch_size', type=int,   default = 128,    help = 'batch size')
parser.add_argument('--epochs',     type=int,   default = 100,    help = 'number of train epochs')
parser.add_argument('--lr',         type=float, default = 0.001,  help = 'learning rate')
parser.add_argument('--C',          type=float, default = 0.1,    help = 'grad clip')
parser.add_argument('--sigma',      type=float, default = 0.01,   help = 'noise std')
args = parser.parse_args()


data_root='/home/jack/公共的/MLData/'
trainset       = torchvision.datasets.MNIST(root = data_root, train = True,  download = True, transform = torchvision.transforms.ToTensor() )
testset        = torchvision.datasets.MNIST(root = data_root, train = False, download = True, transform = torchvision.transforms.ToTensor() )
train_loader   = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle = True, )
test_loader    = torch.utils.data.DataLoader(testset,  batch_size = args.batch_size, shuffle = False, )
device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

model = LeNet_3().to(device)
lossFun = torch.nn.CrossEntropyLoss(reduction='none')
optim   = torch.optim.Adam(model.parameters(), lr = args.lr, betas = (0.5, 0.999), eps = 1e-8)

TraRecord_DPSGD_batch = TraRecorder(Len = 4)

for epoch in range( args.epochs):
    torch.cuda.empty_cache()  # 释放显存
    model.train()
    TraRecord_DPSGD_batch.addlog(epoch)
    print(f"\nEpoch : {epoch+1}/{args.epochs}({100.0*(epoch+1)/args.epochs:0>5.2f}%)")

    sum_examples = 0.0
    sum_acc = 0.0
    sum_loss = 0.0
    for batch, (X, y) in enumerate(train_loader):
        optim.zero_grad()       # 必须在反向传播前先清零。
        sum_examples += X.size(0)
        # model.zero_grad()
        X, y   =  X.to(device), y.to(device)
        y_hat  = model(X)
        losses = lossFun(y_hat, y)

        ## 初始化记录裁剪和添加噪声的容器
        ## losses = torch.mean(loss.reshape(batch_size, -1), dim=1)
        clipped_grads = {}
        for key, param in model.named_parameters():
            clipped_grads[key] = torch.zeros_like(param)

        for los in losses:
            los.backward(retain_graph=True)
            # 裁剪梯度，C为边界值，使得模型参数梯度在[-C,C]范围内
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.C)
            # 存储裁剪后的梯度
            for key, param in model.named_parameters():
                clipped_grads[key].add_(param.grad)
            model.zero_grad()

        for key, param in model.named_parameters():
            # 初始化噪声
            noise = torch.normal(mean = 0, std = args.sigma, size = param.shape ).to(device)
            # 添加高斯噪声
            clipped_grads[key].add_(noise)
            param.grad = clipped_grads[key] / X.size(0)
        optim.step()

        batch_acc = (y_hat.argmax(axis = 1) == y).float().sum().item()
        sum_acc += batch_acc
        sum_loss += losses.mean().item()
        # 输出训练状态
        if (batch + 1) % 100 == 0:
            frac1 = (epoch + 1) /  args.epochs
            frac2 = (batch + 1) / len(train_loader)
            print("    [epoch: {:*>5d}/{}({:0>6.2%}), batch: {:*>5d}/{}({:0>6.2%})]\tLoss: {:.4f} \t  Train acc:{:4.2f} ".format(epoch+1,  args.epochs, frac1, batch+1, len(train_loader), frac2, losses.mean().item(), batch_acc, ))

    test_acc = validata(model, test_loader)
    train_acc = sum_acc/sum_examples
    train_loss = sum_loss/len(train_loader)
    TraRecord_DPSGD_batch.assign([test_acc, train_acc, train_loss])
    print("  ******")
    print(f"  {epoch+1}/{args.epochs}({(epoch+1)*100.0/args.epochs:5.2f}%): train loss = {train_loss:.4f}, train acc: {train_acc:.3f} | test acc: {test_acc:.3f}")
    print("  ******\n")


TraRecord_DPSGD_batch.save("./")


















