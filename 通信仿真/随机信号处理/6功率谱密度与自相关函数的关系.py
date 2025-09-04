#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:13:36 2025

@author: jack

Wiener-Khinchin定理: 自相关函数的傅里叶变换正是信号的功率谱密度.
Parseval定理:  信号的总功率等于其功率谱密度在整个频域的积分。它本质上是能量守恒定律在信号处理中的体现。
"""

import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

## my lib
from Xcorrs import xcorr

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
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12

def freqDomainView(x, Fs, FFTN, type = 'double'): # N为偶数
    X = scipy.fftpack.fft(x, n = FFTN)
    # 消除相位混乱
    threshold = np.max(np.abs(X)) / 10000
    X[np.abs(X) < threshold] = 0
    # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
    X = X/x.size               # 将频域序列 X 除以序列的长度 N
    if type == 'single':
        Y = X[0 : int(FFTN/2)+1].copy()       # 提取 X 里正频率的部分,N为偶数
        Y[1 : int(FFTN/2)] = 2*Y[1 : int(FFTN/2)].copy()
        f = np.arange(0, int(FFTN/2)+1) * (Fs/FFTN)
        # 计算频域序列 Y 的幅值和相角
        A = abs(Y)                        # 计算频域序列 Y 的幅值
        Pha = np.angle(Y, deg=1)          # 计算频域序列 Y 的相角 (弧度制)
        R = np.real(Y)                    # 计算频域序列 Y 的实部
        I = np.imag(Y)                    # 计算频域序列 Y 的虚部
    elif type == 'double':
        f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/Fs))
        Y = scipy.fftpack.fftshift(X, )
        # 计算频域序列 Y 的幅值和相角
        A = abs(Y)                        # 计算频域序列 Y 的幅值
        Pha = np.angle(Y, deg=1)          # 计算频域序列 Y 的相角 (弧度制)
        R = np.real(Y)                    # 计算频域序列 Y 的实部
        I = np.imag(Y)                    # 计算频域序列 Y 的虚部

    return f, Y, A, Pha, R, I
#%% Wiener-Khinchin定理: 自相关函数的傅里叶变换正是信号的功率谱密度
#%% 例1：滤波器设计
fs = 1000 # 采样频率
T = 1       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
f1 = 50  # 通信信号频率 (Hz)
f2 = 200

signal = np.sin(2*np.pi*f1*t) + 0.5 * np.sin(2*np.pi*f2*t)  # 通信信号
noise = 0.3 * np.random.randn(t.size)                       # 白噪声
x = signal + noise;                                         # 最终信号

## low_pass filter
#================= IIR -巴特沃兹低通滤波器  =====================
lf = 100    # 通带截止频率200Hz
Fc = 500   # 阻带截止频率1000Hz
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
y = scipy.signal.lfilter(Bb, Ba, x) # 进行滤波
# y = scipy.signal.filtfilt(Bb, Ba, x )

fx, Yx, Ax, Phax, Rx, Ix = freqDomainView(x, fs, x.size, type = 'double')
fy, Yy, Ay, Phay, Ry, Iy = freqDomainView(y, fs, x.size, type = 'double')

##### plot
fig, axs = plt.subplots(3, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, x, color = 'b', lw = 1, label = '周期图法')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("含噪信号 (时域)")
axs[0].legend()

axs[1].plot(t, y, color = 'b', lw = 1, label = '周期图法')
axs[1].set_xlabel('时间 (s)',)
axs[1].set_ylabel('幅度',)
axs[1].set_title("滤波后的信号 (时域)")
axs[1].legend()

axs[2].plot(fx, Ax, color = 'b', lw = 2, label = '滤波前')
axs[2].plot(fy, Ay, color = 'r', lw = 0.8, label = '滤波后')
axs[2].set_xlabel('频率 (Hz)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("滤波前后的信号 (频域)")
axs[2].legend()

plt.show()
plt.close()

#%% 例2：信号调制与解调, 幅度调制（AM）
fs = 1000;                   # 采样频率 (Hz)
T = 1
t = np.arange(0, T, 1/fs)    # 时间向量

fc = 100;                   # 载波频率 (Hz)
fm = 10;                    # 调制信号频率 (Hz)
Am = 1;                     # 调制信号幅度
Ac = 1;                     # 载波信号幅度

m = Am * np.cos(2 * np.pi * fm * t)
c = Ac * np.cos(2 * np.pi * fc * t)
s = (1 + m) * c

s_demod = np.abs(scipy.signal.hilbert(s)) - 1

##### plot
fig, axs = plt.subplots(4, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, m, color = 'b', lw = 0.2, label = '周期图法')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("调制信号 (时域)")
# axs[0].legend()

axs[1].plot(t, c, color = 'b', label = '周期图法')
axs[1].set_xlabel('时间 (s)',)
axs[1].set_ylabel('幅度',)
axs[1].set_title("载波信号 (时域)")
# axs[1].legend()

axs[2].plot(t, s, color = 'b', lw = 2, label = '滤波前')
axs[2].set_xlabel('时间 (s)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("调制信号 (AM, 时域)")
# axs[2].legend()

axs[3].plot(t, s_demod, color = 'b', label = '周期图法')
axs[3].set_xlabel('时间 (s)',)
axs[3].set_ylabel('幅度',)
axs[3].set_title("解调信号 (时域)")
# axs[1].legend()

plt.show()
plt.close()


#%% 例3：噪声分析与消除
fs = 1000;                   # 采样频率 (Hz)
T = 1
t = np.arange(0, T, 1/fs)    # 时间向量

f_noise = 50
x_signal = np.sin(2*np.pi*10*t);
x_noise = 0.3 * np.sin(2*np.pi*f_noise*t);
x = x_signal + x_noise;

## low_pass filter
#================= IIR -巴特沃兹低通滤波器  =====================
lf = 100    # 通带截止频率200Hz
Fc = 500   # 阻带截止频率1000Hz
Rp = 1      # 通带波纹最大衰减为1dB
Rs = 40     # 阻带衰减为40dB
#-----------------计算最小滤波器阶数--------------------
na = np.sqrt(10**(0.1*Rp)-1)
ea = np.sqrt(10**(0.1*Rs)-1)
order  = np.ceil(np.log10(ea/na)/np.log10(Fc/lf))  #巴特沃兹阶数
f1 = (f_noise-2)/(fs/2)
f2 = (f_noise+2)/(fs/2)
#---------------- 低通滤波  -----------------------
### 方法1
[Bb, Ba] = scipy.signal.butter(4, [f1, f2], 'bandstop')
y = scipy.signal.lfilter(Bb, Ba, x) # 进行滤波

def correlogram_method(signal, fs, N):
    Rxx, lag = xcorr(signal, signal, normed = True, detrend = True, maxlags = signal.size - 1)
    Rxx = Rxx[N-1:]
    Rxx = np.fft.fft(Rxx, n = N)
    Pxx = np.abs(Rxx[0: int(N/2) + 1])
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
fx, Pxx = correlogram_method(x, fs, x.size)
fy, Pyy = correlogram_method(y, fs, y.size)

# fx, Pxx = scipy.signal.periodogram(x, fs, ) # window = window_hann, nfft = N2
# fy, Pyy = scipy.signal.periodogram(y, fs, ) # window = window_hann, nfft = N2

##### plot
fig, axs = plt.subplots(3, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

axs[0].plot(t, x, color = 'b', lw = 2, label = '周期图法')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("含噪信号 (时域)")
# axs[0].legend()

axs[1].plot(t, y, color = 'b', label = 'periodogram')
axs[1].set_xlabel('时间 (s)',)
axs[1].set_ylabel('幅度',)
axs[1].set_title("滤波后的信号 (时域)")
# axs[1].legend()

axs[2].plot(fx, 10*np.log10(Pxx), color = 'b', lw = 2, label = '滤波前')
axs[2].plot(fy, 10*np.log10(Pyy), color = 'r', lw = 0.8, label = '滤波后')
axs[2].set_xlabel('频率 (Hz)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("滤波前后的信号 (频域)")
axs[2].legend()

plt.show()
plt.close()


#%% 6.3 实例分析：如何通过自相关函数获得功率谱密度

A = 1;           # 正弦波幅度
f0 = 5;          # 正弦波频率 (Hz)
fs = 100;        # 采样频率 (Hz)
T = 1;           # 信号持续时间 (s)
sigma = 0.5;     # 噪声标准差
t = np.arange(0, T, 1/fs)    # 时间向量
x = A * np.cos(2*np.pi*f0*t) + sigma * np.random.randn(t.size)

def correlogram_method(signal, fs, N):
    Rxx, lag = xcorr(signal, signal, normed = True, detrend = True, maxlags = signal.size - 1)
    Rxx = Rxx[N-1:]
    Rxx = np.fft.fft(Rxx, n = N)
    Pxx = np.abs(Rxx[0: int(N/2) + 1])
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
f1, Pxx_xcorr = correlogram_method(x, fs, x.size)

acf_sin, lag_sin = xcorr(x, x, normed = True, detrend = 1, maxlags = x.size - 1)

##### plot
fig, axs = plt.subplots(2, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

axs[0].plot(lag_sin/fs, acf_sin, color = 'b', lw = 2, label = '周期图法')
axs[0].set_xlabel(r'时间延迟 $\tau$ (秒)',)
axs[0].set_ylabel(r'自相关函数 $R_x(\tau)$',)
axs[0].set_title(r"自相关函数 $R_x(\tau)$")
# axs[0].legend()

axs[1].plot(f1, Pxx_xcorr, color = 'b', label = 'periodogram')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel(r'功率谱密度 $S_x(f)$',)
axs[1].set_title(r"功率谱密度 $S_x(f)$")
# axs[1].legend()

plt.show()
plt.close()
























































































































































































































































































































































