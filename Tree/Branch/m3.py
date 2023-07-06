#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 21:09:20 2022

@author: jack
"""

from  .  import m4


def printSelf():
	print('In m3')

m4.printSelf()

class DCGAN:
    def __init__(self, input_dim=784, g_dim=100, max_step=100, sample_size=256, d_iter=3, kind='normal'):
        self.input_dim = input_dim       # 图像的展开维度，即判别网络的输入维度
        self.g_dim = g_dim               # 随机噪声维度，即生成网络的输入维度
        self.max_step = max_step         # 整个模型的迭代次数
        self.sample_size = sample_size   # 训练过程中小批量采样的个数的一半
        self.d_iter = d_iter             # 每次迭代，判别网络训练的次数
        self.kind = kind                 # 随机噪声分布类型

