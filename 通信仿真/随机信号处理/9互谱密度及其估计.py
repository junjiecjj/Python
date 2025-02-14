#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:03:30 2025

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


fs = 1000 # 采样频率
T = 10       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
f1 = 50  # 通信信号频率 (Hz)
f2 = 150
noise_level = 0.9
x = np.sin(2*np.pi*f1*t) +  noise_level * np.random.randn(t.size)              # 白噪声
y = np.cos(2*np.pi*f2*t) +  noise_level * np.random.randn(t.size)
nfft = 1024
window = scipy.signal.windows.hamming(512)
noverlap = 256

### 直接傅里叶变换法估计的互谱密度
f_csd, Sxy = scipy.signal.csd(x, y, fs = fs, window = window, nperseg = noverlap * 2, noverlap = noverlap, nfft = nfft)

### 基于互相关函数法估计的互谱密度
x_detrend = scipy.signal.detrend(x)
y_detrend = scipy.signal.detrend(y)
Rxy, lag = xcorr(x_detrend, y_detrend, normed = True, detrend = True, maxlags = x_detrend.size - 1)
Rxy = Rxy[x_detrend.size - 1:] # 取正半轴部分
Sxy_corr = np.fft.fft(Rxy, n = nfft)
Sxy_corr = np.abs(Sxy_corr[0: int(nfft/2) + 1]) # 取前半部分（正频率）
f_xcorr = np.arange(0, nfft/2+1) * (fs/nfft)

##### plot
fig, axs = plt.subplots(2, 1, figsize = (8, 8), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(f_csd, np.abs(Sxy), color = 'b', lw = 2, label = '直接傅里叶变换法估计的互谱密度')
axs[0].set_xlabel('频率 (Hz)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("直接傅里叶变换法估计的互谱密度")
axs[0].legend()

axs[1].plot(f_xcorr, Sxy_corr, color = 'r', label = '基于互相关函数法估计的互谱密度')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('幅度',)
axs[1].set_title("基于互相关函数法估计的互谱密度")
axs[1].legend()


plt.show()
plt.close()















































































































































