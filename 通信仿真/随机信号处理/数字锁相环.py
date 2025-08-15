#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:23:08 2025

@author: jack
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#%%  https://blog.csdn.net/flex_tan/article/details/54884251
phase_offset = 0.0 # 载波相位补偿
freq_offset = 0.3  # 载波频率补偿
wn = 0.01          # PLL 带宽
zeta = 0.707       # pll damping factor
K = 1000           # pll loop gain
n = 500            # number of samples
## generate loop filter parameters
t1 = K/(wn*wn)     # tau1
t2 = 2*zeta / wn   # tau2

## feed-forward cofficients (numerator)
b0 = (4 * K / t1)*(1 + t2/2.0)
b1 = (8*K/t1)
b2 = (4 * K /t1) * (1. - t2/2.0)

## feed-forward cofficients (denominator)
a1 = -2.0
a2 = 1.0

## filter buffer
v0 = 0.0
v1 = 0.0
v2 = 0.0

## initialize states
phi = phase_offset # 输入信号初始相位
phi_hat = 0.0      # PLL 初始相位

delta_phi = np.zeros(n)
for i in range(n):
    # 计算输入波形及更新相位
    x = np.exp(1j * phi)
    phi += freq_offset

    # 根据相位估计计算PLL输出
    y = np.exp(1j * phi_hat)

    # 计算误差估计
    delta_phi[i] = np.angle(x * np.conjugate(y))

    # 更新缓存
    v2 = v1 # shift center register to upper register
    v1 = v0 # shift lower register to center register

    # compute new lower register
    v0 = delta_phi[i] - v1 * a1 - v2 * a2

    # 计算新的相位
    phi_hat = v0 * b0 + v1 * b1 + v2 * b2
    phi_hat = np.mod(phi_hat, 2*np.pi)

plt.plot(delta_phi)

#%%


#%%


#%%



#%%



#%%


#%%



#%%



#%%






























