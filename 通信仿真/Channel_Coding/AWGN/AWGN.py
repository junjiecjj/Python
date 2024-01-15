#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 15:49:09 2022

@author: jack

https://www.its404.com/article/sinat_41657218/106171768

https://www.shuzhiduo.com/A/rV57oe7X5P/

"""

import matplotlib.pyplot as plt
import math
import numpy as np


def awgn(x, snr, seed=7):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    np.random.seed(seed)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)   #信号功率
    npower = xpower / snr         # 噪声功率，对于均值为0的正态分布，等于噪声的方差,因为D(X) = E(X^2) - E(X)^2 = E(X^2)
    noise = np.random.randn(len(x)) * np.sqrt(npower)   #np.random.randn()
    return x + noise

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)
 
t = np.arange(0, 3, 0.001)
x = np.sin(2*np.pi*t)
snr = 2
n = wgn(x, snr)
xn = x+n # 增加了6dBz信噪比噪声的信号

xn1 = awgn(x,snr)


fig, axs = plt.subplots(3, 1)
axs[0].plot(t,x,'b-',lw=0.6)
axs[1].plot(t,xn,'r-',lw=0.6)
axs[2].plot(t,xn1,'g-',lw=0.5)

