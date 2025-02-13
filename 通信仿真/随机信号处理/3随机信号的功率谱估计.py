#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:12:52 2025

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
t = np.arange(0, 1, 1/fs) # 时间向量
carrier_freq = 100 # 载波频率
mod_freq = 10 # 调制频率
noise_power = 0.1 # 噪声功率

x = (1 + 1/2 * np.sin(2 * np.pi * mod_freq * t)) * np.cos(2 * np.pi * carrier_freq * t)
# x = np.sin(2 * np.pi * 50 * t)
# x = np.random.randn(t.size)
# x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
xn = x + np.sqrt(noise_power) * np.random.randn(t.size)

N = len(xn)           # 信号长度

######% 1 计算周期图法的功率谱密度
## 自带的库
# from scipy import signal

N2 = 1000
window_hann = scipy.signal.windows.hann(N2)   # haning
window_hamm = scipy.signal.windows.hamming(N2)   # haming
# f, Pxx_periodogram = scipy.signal.periodogram(xn[:N2], fs, window = window_hann, nfft = N2)

## 手写
def periodogram_method(signal, fs, N):
    X = np.fft.fft(signal, n = N)
    Pxx = np.abs(X)**2/N
    Pxx = Pxx[0:int(N/2) + 1]
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx,
f, Pxx_periodogram = periodogram_method(xn, fs, N)

######% 2 自相关函数法
def correlogram_method(signal, fs, N):
    Rxx, lag = xcorr(signal, signal, normed = True, detrend = True, maxlags = signal.size - 1)
    Rxx = Rxx[N-1:]
    Rxx = np.fft.fft(Rxx, n = N)
    Pxx = np.abs(Rxx[0: int(N/2) + 1])
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
f1, Pxx_xcorr = correlogram_method(xn, fs, N)

######% 3 Welch 方法
L = 256              # Welch方法中的子段长度
D = L//2              # 重叠长度
f2, Pxx_welch = scipy.signal.welch(xn, fs, window = 'hann', nperseg = L, noverlap = D)

### 不同方法的估计方差比较
var_periodogram = np.var(Pxx_periodogram);
var_correlogram = np.var(Pxx_xcorr)
var_welch = np.var(Pxx_welch)

##### plot
fig, axs = plt.subplots(3, 1, figsize = (12, 8), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(f, 10*np.log10(Pxx_periodogram), color = 'b', label = '周期图法')
axs[0].plot(f1, 10*np.log10(Pxx_xcorr), color = 'orange', label = '自相关函数法')
axs[0].plot(f2, 10*np.log10(Pxx_welch), color = 'g', label = 'Welch 方法')
axs[0].set_title("发射信号 X(t)")
axs[0].set_xlabel('频率 (Hz)',)
axs[0].set_ylabel('功率/频率 (dB/Hz)',)
axs[0].set_title("功率谱密度估计")
axs[0].legend()

axs[1].bar(["周期图法", "自相关函数法", "Welch 方法"], [var_periodogram, var_correlogram, var_welch], label = ["周期图法", "自相关函数法", "welch方法"], color = ["b","orange","g"])
axs[1].legend()

# axs[2].plot(lag/fs, acf, label = 'PSD',  )
# axs[2].set_title(r"互相关函数 $R_{XY}(\tau)$")
# axs[2].set_xlabel('时间延迟 (秒)')
# axs[2].set_ylabel('互相关值')

plt.show()
plt.close()





































































































































