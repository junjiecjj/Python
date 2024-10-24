
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""

#sys.path.append(os.getcwd())
from model import common
# 或
# from .  import common
import sys, os
sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()



import math
import torch
import torch.nn.functional as F
import torch.nn.parallel as P
from torch import nn, Tensor
from einops import rearrange
import copy
import datetime

def Calculate_filters(comp_ratio, F=5, n=3*48*48):
    K = (comp_ratio * n) / F**2
    return int(K)

class  DeepSc1(nn.Module):
    def __init__(self):
        super(DeepSc1, self).__init__()
        padding = [2, 2]
        comp_ratio = 0.17
        n_colors = 3
        patch_size = 48
        e0 =  patch_size
        encoder1_size = int((e0 + 2 * padding[0] - 5) / 2 + 1)
        encoder2_size = int((encoder1_size + 2 * padding[1] - 5) / 2 + 1)
        self.last_channel = Calculate_filters( comp_ratio, encoder2_size)
        self.c1 = nn.Conv2d( n_colors,  n_colors, 5, 2, padding[0])
        self.c2 = nn.Conv2d( n_colors, self.last_channel, 5, 2, padding[1])
        self.c3 = nn.BatchNorm2d(self.last_channel)


        self.d1 = nn.ConvTranspose2d(self.last_channel,  n_colors, 5, 2, padding[1])
        self.d2 = nn.ConvTranspose2d( n_colors,  n_colors, 5, 2, padding[0])
        self.d3 = nn.Conv2d( n_colors,  n_colors, 4, 1, 3)
        self.d4 = nn.BatchNorm2d(n_colors)

    def forward(self, img, idx_scale=0, snr=10, compr_idx=0):
        print(f"img.shape = {img.shape}")

        e1 = self.c1(img)
        print(f"e1.shape = {e1.shape}")

        e2 = self.c2(e1)
        print(f"e2.shape = {e2.shape}")

        #e3 = self.c3(e2)
        #print(f"e3.shape = {e3.shape}")

        d1 = self.d1(e2)
        print(f"d1.shape = {d1.shape}")

        d2 = self.d2(d1)
        print(f"d2.shape = {d2.shape}")

        d3 = self.d3(d2)
        print(f"d3.shape = {d3.shape}")

        #d4 = self.d4(d3)
        #print(f"d4.shape = {d4.shape}")

        return d3

# net = DeepSc1()

# X = torch.randn(16,3,48,48)

# Y = net(X)

#print(f"X.shape = {X.shape}, Y.shape = {Y.shape}\n")




#网络模型结构
class DeepSc(nn.Module):
    def __init__(self , args, ckp):
        super(DeepSc, self).__init__()
        self.args = args

        self.print_parameters(ckp)

        # 输入 1 * 28 * 28
        # 卷积层1
        # 在输入基础上增加了padding，28 * 28 -> 32 * 32
        # 1 * 32 * 32 -> 6 * 28 * 28
        self.conv1 = common.conv2d_prelu(in_channels=3, out_channels=16, kernel_size=5,stride=2, pad=4)

        # 6 * 28 * 28 -> 6 * 14 * 14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # kernel_size, stride
        # 卷积层2
        # 6 * 14 * 14 -> 16 * 10 * 10
        self.conv2 = common.conv2d_prelu(in_channels=16, out_channels=32, kernel_size=5, stride=2, pad=2)

        self.conv3 = common.conv2d_prelu(in_channels=32, out_channels=32, kernel_size=5, stride=1, pad=2)

        self.conv4 = common.conv2d_prelu(in_channels=32, out_channels=32, kernel_size=5, stride=1, pad=1)

        self.conv5 = common.conv2d_prelu(in_channels=32, out_channels=3, kernel_size=5, stride=1, pad=1)

        # 16 * 10 * 10 -> 16 * 5 * 5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层1
        self.l1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)

        # 全连接层2
        self.l2 = nn.Linear(in_features=120, out_features=84)

        self.l3 = nn.Linear(in_features=84, out_features=10)

        self.deconv1 = common.convTrans2d_prelu(3, 32, 5, 1, 1)
        self.deconv2 = common.convTrans2d_prelu(32, 32, 5, 1, 1)
        self.deconv3 = common.convTrans2d_prelu(32, 32, 5, 1, 2)
        self.deconv4 = common.convTrans2d_prelu(32, 16, 5, 2, 2)
        self.deconv5 = common.convTrans2d_prelu(16, 3, 6, 2, 3)

        print(color.fuchsia(f"\n#================================ DeepSC 准备完毕 =======================================\n"))


    def forward(self, img, idx_scale=0, snr=10, compr_idx=0):
        #print(f"107 img.shape = {img.shape}")
        # img.shape = torch.Size([16, 3, 48, 48])

        e1 = self.conv1(img)
        #print(f"e1.shape = {e1.shape}")
        # e1.shape = torch.Size([16, 16, 26, 26])

        #e2 = self.pool1(e1)
        #print(f"e2.shape = {e2.shape}")

        e3 = self.conv2(e1)
        #print(f"e3.shape = {e3.shape}")
        # e3.shape = torch.Size([16, 32, 13, 13])

        #e4 = self.pool2(e3)
        #print(f"e4.shape = {e4.shape}")

        e5 = self.conv3(e3)
        #print(f"e5.shape = {e5.shape}")
        # e5.shape = torch.Size([16, 32, 13, 13])

        e6 = self.conv4(e5)
        #print(f"e6.shape = {e6.shape}")
        # e6.shape = torch.Size([16, 32, 11, 11])

        e7 = self.conv5(e6)
        #print(f"e7.shape = {e7.shape}")
        # e7.shape = torch.Size([16, 3, 9, 9])

        d1 = self.deconv1(e7)
        #print(f"d1.shape = {d1.shape}")
        # d1.shape = torch.Size([16, 32, 11, 11])

        d2 = self.deconv2(d1)
        #print(f"d2.shape = {d2.shape}")
        # d2.shape = torch.Size([16, 32, 13, 13])

        d3 = self.deconv3(d2)
        #print(f"d3.shape = {d3.shape}")
        # d3.shape = torch.Size([16, 32, 13, 13])

        d4 = self.deconv4(d3)
        #print(f"d4.shape = {d4.shape}")
        # d4.shape = torch.Size([16, 16, 25, 25])

        d5 = self.deconv5(d4)
        #print(f"d5.shape = {d5.shape}")
        # d5.shape = torch.Size([16, 3, 48, 48])

        return d5

    def save(self, apath, compratio, snr, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.saveModelEveryEpoch:
            save_dirs.append(os.path.join(apath, 'model_CompRatio={}_SNR={}_Epoch={}.pt'.format(compratio, snr, epoch) ) )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def print_parameters(self, ckp):
        print(f"#=====================================================================================",  file=ckp.log_file)
        print(ckp.now,  file=ckp.log_file)
        print(f"#=====================================================================================",  file=ckp.log_file)
        print(f"{self}", file=ckp.log_file)
        print(f"#======================================== Parameters =============================================",  file=ckp.log_file)
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.size()}, {param.requires_grad} ")
                print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ", file=ckp.log_file)

        return

    #  apath=/cache/results/ipt/model, resume = 0,
    def load(self, apath, cpu=False):
        load_from = None
        load_from1 = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if os.path.isfile(os.path.join(self.args.pretrain, 'model_latest.pt')):
            load_from1 = torch.load(os.path.join(apath, 'model_latest.pt'), **kwargs)
            print(f"在Ipt中加载最原始的模型\n")
        else:
            print(f"Ipt中没有最原始的模型\n")
        if load_from1:
            self.model.load_state_dict(load_from1, strict=False)


        if os.path.isfile(os.path.join(apath, 'model_latest.pt')):
            load_from = torch.load(os.path.join(apath, 'model_latest.pt'), **kwargs)
            print(f"在Ipt中加载最近一次模型\n")
        else:
            print(f"Ipt中没有最近一次模型\n")
        if load_from:
            self.model.load_state_dict(load_from, strict=False)
