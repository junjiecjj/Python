#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 19:06:16 2022

@author: jack

https://blog.csdn.net/keypig_zz/article/details/124382656

"""
import scipy.stats as st
import scipy.stats as stats
import matplotlib
# matplotlib.get_backend()
# matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator



L, N = 100, 100
x = np.ones(L) # 时域离散信号
X = np.zeros(N) + np.zeros(N)*1j # 频域频谱

# DTF变换
for k in range(N):
    for n in range(L):
        X[k] = X[k] + x[n]*np.exp(-1j*n*k/N*2*np.pi)


# IDFT变换
x_p = np.zeros(L)
for n in range(L):
    for k in range(N):
        x_p[n] = x_p[n] + 1/N*X[k]*np.exp(1j*n*k/N*2*np.pi)









fig = plt.figure(figsize=(10, 10))
a1 = plt.subplot2grid((4, 1), (0, 0))
a2 = plt.subplot2grid((4, 1), (1, 0))
a3 = plt.subplot2grid((4, 1), (2, 0))
a4 = plt.subplot2grid((4, 1), (3, 0))

a1.stem(x) #原始序列
a2.stem(np.abs(X)) #DFT频谱（幅度）
a3.stem(np.angle(X, deg=True)) # DFT频谱（相位）
a4.stem(x_p) # IDFT序列

a1.set_ylabel(r'Original sequence')

a2.set_ylabel(r'DTF:|X($\omega$)|')
a2.set_xticks([0, 25, 50,75, 99])
a2.set_xticklabels(['0', r'$\frac{\pi}{2}$',r'$\pi$',r'$\frac{3\pi}{2}$',r'$2\pi$'])

a3.set_ylabel(r'DFT:$\theta(\omega)$')
a3.set_xticks([0, 25, 50,75, 99])
a3.set_xticklabels(['0', r'$\frac{\pi}{2}$',r'$\pi$',r'$\frac{3\pi}{2}$',r'$2\pi$'])

a4.set_ylabel(r'IDFT sequence')
plt.show()



Y = np.roll(X, int(N/2)) # 平移频谱
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.stem(np.abs(Y))
ax.set_xticks([0, 25, 50,75, 99])
ax.set_xticklabels(['-$\pi$', r'$-\frac{\pi}{2}$',r'0',r'$\frac{\pi}{2}$',r'$\pi$'])
ax.set_ylabel(r'DTF:|X($\omega$)|')
plt.grid(True)
plt.show()














