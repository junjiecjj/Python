#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:13:36 2025

@author: jack
"""

import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

## my lib
from Xcorrs import xcorr, correlate_maxlag, correlate_template, get_lags

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12


#%% Wiener-Khinchin定理: 自相关函数的傅里叶变换正是信号的功率谱密度

#%% 例1：滤波器设计

fs = 1000 # 采样频率
T = 1       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
f1 = 50  # 通信信号频率 (Hz)
f2 = 200

signal = np.sin(2*np.pi*f1*t) + 0.5 * np.sin(2*np.pi*f2*t)  # 通信信号
noise = 0.3 * np.random.randn(t.size)              # 白噪声
x = signal + noise;         # 最终信号


## low_pass filter
#================= IIR -巴特沃兹低通滤波器  =====================
lf = 200    # 通带截止频率200Hz
Fc = 1000   # 阻带截止频率1000Hz
Rp = 1      # 通带波纹最大衰减为1dB
Rs = 40     # 阻带衰减为40dB
#-----------------计算最小滤波器阶数--------------------
na = np.sqrt(10**(0.1*Rp)-1)
ea = np.sqrt(10**(0.1*Rs)-1)
order  = np.ceil(np.log10(ea/na)/np.log10(Fc/lf))  #巴特沃兹阶数
Wn = lf*2/fs
#---------------- 低通滤波  -----------------------
### 方法1
[Bb, Ba] = scipy.signal.butter(order, Wn, 'low')
# ## [BW, BH] = scipy.signal.freqz(Bb, Ba)
# y = scipy.signal.lfilter(Bb, Ba, x) # 进行滤波
y = scipy.signal.filtfilt(Bb, Ba, x )

















































































































































































































































































































































































