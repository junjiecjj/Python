#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:50:42 2023

@author: jack

https://www.jianshu.com/p/7d3a17f00312
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator


# Generator  【 N(0,1)->N(3,1)】
class Generator(nn.Module):
    def __init__(self, g_in = 10, g_out = 10, h_dim = 100):
        super(Generator,self).__init__()
        self.net = nn.Sequential(
            #输入 z:[batch,1]
            nn.Linear(g_in, h_dim),
            nn.LeakyReLU(True),
            nn.Linear(h_dim, g_out)
        )
    def forward(self,z):
        output = self.net(z)
        return output



# Generator  【 N(0,1)->N(3,1)】
#  Discriminator  【data -> pred】
class Discriminator(nn.Module):
    def __init__(self, d_in = 10, h_dim = 100):
        super(Discriminator,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(d_in, h_dim),
            nn.LeakyReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        output = self.net(x)
        return output


def real_data_generator(batchsz = 128, dim = 10):
    while True:
        data = np.random.randn(batchsz, dim)
        data = data + 3
        yield data



# normfun正态分布，概率密度函数
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

def drow_image(G, epoch):
    # 1 画出想要拟合的分布
    data=np.random.randn(batchsz, 10) + 3
    x = np.arange(0,6, 0.2)
    y = normfun(x, 3, 1)
    plt.plot(x, y, 'r', linewidth=3)
    plt.hist( data.flatten() , bins=50, color='grey', alpha=0.5, rwidth=0.9, density=True)

    # 2 画出目前生成器生成的分布
    x = torch.randn(batchsz, data_dim).to(device)
    data=G(x).cpu().detach().numpy()
    mean = data.mean()
    std = data.std()
    print(mean, std)
    x = np.arange(np.floor(data.min())-5, np.ceil(data.max())+5, 0.2)
    y = normfun(x, mean, std)
    plt.plot(x, y, 'g', linewidth=3)
    plt.hist(data.flatten(), bins=50, color='b', alpha=0.5, rwidth=0.9, density=True)

    # plt 的设置
    title = 'epoch' + epoch.__str__()
    plt.title(title)
    plt.xlabel('value')
    plt.ylabel('Probability')
    title = "/home/jack/SemanticNoise_AdversarialAttack/tmpout/"+title+ ".png"
    plt.savefig(title)
    plt.show()

batchsz = 128
data_dim = 10


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# def main():
torch.manual_seed(66)
np.random.seed(66)
data_iter = real_data_generator(batchsz = 128)
G = Generator().to(device)
D = Discriminator().to(device)
optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))
optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))

for epoch in range(2000):
    # 1、train Discrimator firstly
    for _ in range(3):
        # 1.1 获取真实数据
        x = next(data_iter)
        x_real = torch.from_numpy(x).float().to(device)  # numpy -> tensor
        # 对真实数据判别
        pred_real = D(x_real)
        # max pred_real
        loss_real = -pred_real.mean()
        # print(f"1 x.shape = {x.shape}, x_real.shape = {x_real.shape}, pred_real.shape = {pred_real.shape}, loss_real = {loss_real}")
        # 1 x.shape = (128, 1), x_real.shape = torch.Size([128, 1]), pred_real.shape = torch.Size([128, 1]), loss_real = -0.5694580078125

        # 1.2 获取随机数据
        z = torch.randn(batchsz, data_dim).to(device)
        #  生成假数据
        x_fake = G(z).detach()  # tf.stop_gradient()
        # 对假数据判别
        pred_fake = D(x_fake)
        loss_fake = pred_fake.mean()
        # 计算损失函数
        loss_D = loss_real + loss_fake
        # optimize
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()
        # print(f"2 z.shape = {z.shape}, x_fake.shape = {x_fake.shape}, pred_fake.shape = {pred_fake.shape}, loss_D = {loss_D}")
        # 2 z.shape = torch.Size([128, 1]), x_fake.shape = torch.Size([128, 1]), pred_fake.shape = torch.Size([128, 1]), loss_D = -0.08284503221511841

    # 2、train Generator
    # 2.1 获取随机数据
    z = torch.randn(batchsz, data_dim).to(device)
    # 2.2 Generator生成假数据
    xf = G(z)
    # 2.3 Discrimator对假数据进行判别
    predf = D(xf)
    # 2.4 得到损失函数
    loss_G = -predf.mean()
    # optimize
    optim_G.zero_grad()
    loss_G.backward()
    optim_G.step()
    # print(f"3 z.shape = {z.shape}, xf.shape = {xf.shape}, predf.shape = {predf.shape}, loss_D = {loss_D}")
    # 3 z.shape = torch.Size([128, 1]), xf.shape = torch.Size([128, 1]), predf.shape = torch.Size([128, 1]), loss_D = -0.11275461316108704

    if epoch % 10 == 0:
        print("轮：",epoch,"   ",loss_D.item(), loss_G.item())
        drow_image(G, epoch)


# if __name__ == '__main__':
#     main()



































