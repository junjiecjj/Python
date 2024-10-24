#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:39:27 2023

@author: jack

https://github.com/RaySunWHUT/Generative-adversarial-networks/tree/main/GANs_1D

https://blog.csdn.net/qq_40994260/article/details/114699755

https://www.jianshu.com/p/7d3a17f00312

https://www.pytorchtutorial.com/50-lines-of-codes-for-gan/

1   https://mathpretty.com/10808.html
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

import seaborn as sns



#  这个函数输入时一组数据, 输出是这组数据的均值, 方差, 偏度和峰度, 我们用这四个数据来判断这组数据是否是服从高斯分布的数据.
def get_numerical_characteristics(data):
    """
    返回数据(data)的 4 个数字特征(numerical characteristics):
    1. mean：均值
    2. std：标准差
    3. skewness: 偏度
    4. kurtosis: 峰度

    :param data: 数据
    :return: 一维数据: torch.Size([4])
    """
    mean = torch.mean(data)
    diffs = data - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    z_scores = diffs / std

    # 偏度：数据分布偏斜方向、程度的度量, 是数据分布非对称程度的数字特征
    # 定义: 偏度是样本的三阶标准化矩
    skewness = torch.mean(torch.pow(z_scores, 3.0))

    # excess kurtosis, should be 0 for Gaussian
    # 峰度(kurtosis): 表征概率密度分布曲线在平均值处峰值高低的特征数
    # 若峰度(kurtosis) > 3, 峰的形状会比较尖, 会比正态分布峰陡峭
    kurtoses = torch.mean(torch.pow(z_scores, 4.0)) - 3.0

    # reshape(1, )：将常量转化为torch.Size([1])型张量(Tensor)
    final = torch.cat((mean.reshape(1, ), std.reshape(1, ), skewness.reshape(1, ), kurtoses.reshape(1, )))

    return final



# 首先是用来生成服从高斯分布的样本, 即training set中的数据. 这个数据是生成真实的数据, 被用于模仿。
def get_moments(ds):
    """
    - Return the first 4 moments of the data provided
    - 返回一个数据的四个指标, 分别是均值, 方差, 偏度, 峰读
    - 我们希望通过这四个指标, 来判断我们生成的数据是否是需要的数据
    """
    finals = []
    for d in ds:
        mean = torch.mean(d) # d的均值
        diffs = d - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5) # d的方差
        zscores = diffs / (std + 0.001) # 对原始数据 zscores = (d-mean)/std
        skews = torch.mean(torch.pow(zscores, 3.0)) # 峰度
        kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # excess kurtosis, should be 0 for Gaussian
        final = torch.cat((mean.reshape(1,), std.reshape(1,), skews.reshape(1,), kurtoses.reshape(1,)))
        # 这里返回的是高斯分布的四个特征
        finals.append(final)
    return torch.stack(finals)


# 这是输入生成器G中的数据, 用来生成服从均匀分布的数据.
def get_generator_input_sampler(m, n):
    """
    Uniform-dist data into generator, _NOT_ Gaussian
    Input
    - m: 表示batchsize
    - n: 表示feature count
    Output
    - 返回的是生成数据的分布
    """
    return torch.rand(m, n)

def get_distribution_sampler(mu, sigma, batchSize, FeatureNum):
    """
    Generate Target Data, Gaussian
    Input
    - mu: 均值
    - sugma: 方差
    Output
    """
    return Variable(torch.Tensor(np.random.normal(mu, sigma, (batchSize, FeatureNum))))


data_mean = 4
data_stddev = 1.25
batch_size = 1
# featureNum = 50000
# d_real_data = get_distribution_sampler(data_mean, data_stddev, batch_size, featureNum)

# fig, ax1 = plt.subplots()
# ax1.hist(d_real_data, bins=100,  color=sns.desaturate("indianred", .8), alpha=.4)
# ax1.set_ylabel("Count", fontsize='12')
# plt.show()
# plt.close()





class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.f = f # 激活函数
    def forward(self, x):
        x = self.map1(x)
        x = self.relu(x)
        x = self.map2(x)
        x = self.relu(x)
        x = self.map3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.f = f
    def forward(self, x):
        x = self.relu(self.map1(x))
        x = self.relu(self.map2(x))
        x = self.f(self.map3(x))# 最后生成的是概率
        return x



# ----------
# 初始化网络
# ----------
d_input_size = 4
d_hidden_size = 10
d_output_size = 1
discriminator_activation_function = torch.sigmoid

D = Discriminator(input_size=d_input_size,
                  hidden_size=d_hidden_size,
                  output_size=d_output_size,
                  f=discriminator_activation_function)

g_input_size = 50
g_hidden_size = 200
g_output_size = 500
generator_activation_function = torch.tanh

G = Generator(input_size=g_input_size,
              hidden_size=g_hidden_size,
              output_size=g_output_size,
              f=generator_activation_function)


featureNum = g_output_size # 一组样本有500个服从正太分布的数据
minibatch_size = 10 # batch_size的大小
num_epochs = 2001
d_steps = 20 # discriminator的训练轮数
g_steps = 20 # generator的训练轮数


# ----------------------
# 初始化优化器和损失函数
# ----------------------
d_learning_rate = 0.0001
g_learning_rate = 0.0001
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)
d_exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max = d_steps*5, eta_min=0.00001)
g_exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max = g_steps*5, eta_min=0.00001)

G_mean = [] # 生成器生成的数据的均值
G_std = [] # 生成器生成的数据的方差

for epoch in range(num_epochs):
    # -------------------
    # Train the Detective
    # -------------------
    for d_index in range(d_steps):
        # Train D on real+fake
        d_exp_lr_scheduler.step()
        D.zero_grad()
        # Train D on real, 这里的label是1
        d_real_data = get_distribution_sampler(data_mean, data_stddev, minibatch_size, featureNum) # 真实的样本 d_real_data.shape = torch.Size([10, 500]),
        d_real_decision = D(get_moments(d_real_data))                                              # 求出数据的四个重要特征, d_real_decision.shaep = torch.Size([10, 1]),
        d_real_error = criterion(d_real_decision, Variable(torch.ones([minibatch_size, 1])))       # 计算error  d_real_error = 0.5087181329727173 
        d_real_error.backward() # 进行反向传播 
        # print(f"1  d_real_data.shape = {d_real_data.shape}, d_real_decision.shaep = {d_real_decision.shape}, d_real_error = {d_real_error} ")
        # d_real_data.shape = torch.Size([10, 500]), d_real_decision.shaep = torch.Size([10, 1]), d_real_error = 0.5087181329727173 

        # Train D on fake, 这里的label是0
        d_gen_input = get_generator_input_sampler(minibatch_size, g_input_size)  # d_gen_input.shape = torch.Size([10, 50]),
        d_fake_data = G(d_gen_input)                                             # d_fake_data.shaep = torch.Size([10, 500]),
        d_fake_decision = D(get_moments(d_fake_data))                            # d_fake_decision.shape = torch.Size([10, 1]),
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([minibatch_size, 1])))  # d_fake_error = 2.6401114463806152
        d_fake_error.backward()
        # Optimizer
        d_optimizer.step()
        
        # print(f"2  d_gen_input.shape = {d_gen_input.shape}, d_fake_data.shaep = {d_fake_data.shape}, d_fake_decision.shape = {d_fake_decision.shape}, d_fake_error = {d_fake_error }")
        # d_gen_input.shape = torch.Size([10, 50]), d_fake_data.shaep = torch.Size([10, 500]), d_fake_decision.shape = torch.Size([10, 1]), d_fake_error = 2.6401114463806152
    # -------------------
    # Train the Generator
    # -------------------
    for g_index in range(g_steps):
        # Train G on D's response(使得G生成的x让D判断为1)
        g_exp_lr_scheduler.step()
        G.zero_grad()
        gen_input = get_generator_input_sampler(minibatch_size, g_input_size)   # gen_input.shape = torch.Size([10, 50]),
        g_fake_data = G(gen_input)                                              # 使得generator生成样本  g_fake_data.shaep = torch.Size([10, 500]),
        dg_fake_decision = D(get_moments(g_fake_data))                          # D来做的判断  dg_fake_decision.shape = torch.Size([10, 1]),
        g_error = criterion(dg_fake_decision, Variable(torch.ones([minibatch_size, 1]))) # g_error = 0.08104792982339859
        G_mean.append(g_fake_data.mean().item())
        G_std.append(g_fake_data.std().item())
        g_error.backward()
        g_optimizer.step()
        
        # print(f"3  gen_input.shape = {gen_input.shape}, g_fake_data.shaep = {g_fake_data.shape}, dg_fake_decision.shape = {dg_fake_decision.shape}, g_error = {g_error }")
        # gen_input.shape = torch.Size([10, 50]), g_fake_data.shaep = torch.Size([10, 500]), dg_fake_decision.shape = torch.Size([10, 1]), g_error = 0.08104792982339859

    if epoch%10==0:
        print("Epoch: {}, G data's Mean: {}, G data's Std: {}".format(epoch, G_mean[-1], G_std[-1]))
        print("Epoch: {}, Real data's Mean: {}, Real data's Std: {}".format(epoch, d_real_data.mean().item(), d_real_data.std().item()))
        print('-'*10)


# ----------------------
# 计算每个范围的数据个数
# ----------------------
binRange = np.arange(0, 8, 0.1)
hist1,_ = np.histogram(g_fake_data.squeeze().detach().numpy(), bins=binRange)
# --------
# 绘制图像
# --------
fig, ax1 = plt.subplots()
fig.set_size_inches(20, 10)
plt.set_cmap('RdBu')
x = np.arange(len(binRange)-1)
w=0.3
# 绘制多个bar在同一个图中, 这里需要控制width
plt.bar(x, hist1, width=w*3, align='center')
# 设置坐标轴的标签
ax1.yaxis.set_tick_params(labelsize=15) # 设置y轴的字体的大小
ax1.set_xticks(x) # 设置xticks出现的位置
# 创建xticks
xticksName = []
for i in range(len(binRange)-1):
    xticksName = xticksName + ['{}<x<{}'.format(str(np.round(binRange[i],1)), str(np.round(binRange[i+1],1)))]
ax1.set_xticklabels(xticksName)
# 设置坐标轴名称
ax1.set_ylabel("Count", fontsize='xx-large')
plt.show()




fig, ax1 = plt.subplots()
fig.set_size_inches(20, 10)

ax1.plot(G_mean)

plt.show()




fig, ax1 = plt.subplots()
fig.set_size_inches(20, 10)

ax1.plot(G_std)

plt.show()























