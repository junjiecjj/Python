#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 01:17:15 2025

@author: jack
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import scipy


## ======================================================
## ===========  定义时域采样信号
## ======================================================
pi = np.pi


f0 = 0.5
T0 = 1/f0
Fs = 40                           # 信号采样频率
Ts = 1/Fs                         # 采样时间间隔
# N = 100                         # 采样信号的长度

t = np.arange(0, 2*T0+Ts, Ts)    # 定义信号采样的时间点 t
y1 = np.sinc(t)
y2 = np.sin(pi * t)


fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
axs.plot(t, y1, label = 'sinc(x)')
axs.plot(t, y2, label = 'sin(pi*x)')
axs.axhline(y = 0, color = 'r', ls = '--')
axs.legend()

plt.show()
plt.close()
