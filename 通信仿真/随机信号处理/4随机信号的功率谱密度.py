#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:13:03 2025

@author: jack
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
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  场景 1: 通信领域的信号分析 #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fs = 10000 # 采样频率
T = 1       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
f_signal = 2000  # 通信信号频率 (Hz)
A_signal = 1    #  信号幅度

signal = A_signal * np.cos(2*np.pi*f_signal*t);  # 通信信号
interference = 0.3 * np.cos(2*np.pi*3000*t);     # 干扰信号 (3000 Hz)
noise = 0.1 * np.random.randn(t.size)              # 白噪声
x = signal + interference + noise;         # 最终信号

nfft = 2048
######% 1 计算周期图法的功率谱密度

N2 = x.size
window_hann = scipy.signal.windows.hann(N2)   # haning
window_hamm = scipy.signal.windows.hamming(N2)   # haming
f, Pxx_periodogram = scipy.signal.periodogram(x, fs, window = window_hann, nfft = N2) # window = window_hann, nfft = N2

## 手写
def periodogram_method(signal, fs, N):
    X = np.fft.fft(signal, n = N)
    Pxx = np.abs(X)**2/(N * fs)
    Pxx = Pxx[0:int(N/2) + 1]
    Pxx[1:int(N/2)] = 2 * Pxx[1:int(N/2)]
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
# f, Pxx_periodogram = periodogram_method(x, fs, nfft)

######% 2 自相关函数法
def correlogram_method(signal, fs, N):
    Rxx, lag = xcorr(signal, signal, normed = True, detrend = True, maxlags = signal.size - 1)
    Rxx = Rxx[N-1:]
    Rxx = np.fft.fft(Rxx, n = N)
    Pxx = np.abs(Rxx[0: int(N/2) + 1])
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
f1, Pxx_xcorr = correlogram_method(x, fs, x.size)

######% 3 Welch 方法
L = 256              # Welch方法中的子段长度
D = L//2              # 重叠长度
f2, Pxx_welch = scipy.signal.welch(x, fs, window = 'hann', nperseg = L, noverlap = D)

##### plot
fig, axs = plt.subplots(4, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, x, color = 'b', lw = 0.2, label = '周期图法')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("含干扰的通信信号")
axs[0].legend()

axs[1].plot(f, 10*np.log10(Pxx_periodogram), color = 'b', label = '周期图法')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('功率谱密度 (dB/Hz)',)
axs[1].set_title("周期图法-功率谱密度 (dB/Hz)")
axs[1].legend()

axs[2].plot(f1, 10*np.log10(Pxx_xcorr), color = 'b', label = '自相关函数法')
axs[2].set_xlabel('频率 (Hz)',)
axs[2].set_ylabel('功率谱密度 (dB/Hz)',)
axs[2].set_title("自相关函数法-功率谱密度 (dB/Hz)")
axs[2].legend()

axs[3].plot(f2, 10*np.log10(Pxx_welch), color = 'b', label = 'welch方法')
axs[3].set_xlabel('频率 (Hz)',)
axs[3].set_ylabel('功率谱密度 (dB/Hz)',)
axs[3].set_title("Welch 方法-功率谱密度 (dB/Hz)")
axs[3].legend()

plt.show()
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%% 场景 2: 场景 2: 医学图像处理中的噪声抑制 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
fs = 1000 # 采样频率
T = 1       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
f_signal = 50  # 通信信号频率 (Hz)
A_signal = 1    #  信号幅度

signal = A_signal * np.sin(2*np.pi*f_signal*t);  # 通信信号
interference = 0.5 * np.cos(2*np.pi*300*t);     # 高频噪声 (3000 Hz)
noise = 0.1 * np.random.randn(t.size)              # 白噪声
x = signal + interference + noise;         # 最终信号

nfft = 1024
######% 1 计算周期图法的功率谱密度

N2 = x.size
window_hann = scipy.signal.windows.hann(N2)   # haning
window_hamm = scipy.signal.windows.hamming(N2)   # haming
f, Pxx_periodogram = scipy.signal.periodogram(x, fs, window = window_hann, nfft = N2) # window = window_hann, nfft = N2

## 手写
def periodogram_method(signal, fs, N):
    X = np.fft.fft(signal, n = N)
    Pxx = np.abs(X)**2/(N * fs)
    Pxx = Pxx[0:int(N/2) + 1]
    Pxx[1:int(N/2)] = 2 * Pxx[1:int(N/2)]
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
# f, Pxx_periodogram = periodogram_method(x, fs, nfft)

######% 2 自相关函数法
def correlogram_method(signal, fs, N):
    Rxx, lag = xcorr(signal, signal, normed = True, detrend = True, maxlags = signal.size - 1)
    Rxx = Rxx[N-1:]
    Rxx = np.fft.fft(Rxx, n = N)
    Pxx = np.abs(Rxx[0: int(N/2) + 1])
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
f1, Pxx_xcorr = correlogram_method(x, fs, x.size)

######% 3 Welch 方法
L = 256              # Welch方法中的子段长度
D = L//2              # 重叠长度
f2, Pxx_welch = scipy.signal.welch(x, fs, window = 'hann', nperseg = L, noverlap = D)

##### plot
fig, axs = plt.subplots(4, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, x, color = 'b', lw = 0.2, label = '周期图法')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("含高频噪声的医学信号")
axs[0].legend()

axs[1].plot(f, 10*np.log10(Pxx_periodogram), color = 'b', label = '周期图法')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('功率谱密度 (dB/Hz)',)
axs[1].set_title("周期图法-功率谱密度 (dB/Hz)")
axs[1].legend()

axs[2].plot(f1, 10*np.log10(Pxx_xcorr), color = 'b', label = '自相关函数法')
axs[2].set_xlabel('频率 (Hz)',)
axs[2].set_ylabel('功率谱密度 (dB/Hz)',)
axs[2].set_title("自相关函数法-功率谱密度 (dB/Hz)")
axs[2].legend()

axs[3].plot(f2, 10*np.log10(Pxx_welch), color = 'b', label = 'welch方法')
axs[3].set_xlabel('频率 (Hz)',)
axs[3].set_ylabel('功率谱密度 (dB/Hz)',)
axs[3].set_title("Welch 方法-功率谱密度 (dB/Hz)")
axs[3].legend()

plt.show()
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  场景 3: 地震波分  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fs = 100 # 采样频率
T = 10       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
f1 = 2  # 通信信号频率 (Hz)
f2 = 5

signal = np.cos(2*np.pi*f1*t) + 0.5 * np.cos(2*np.pi*f2*t)  # 通信信号
noise = 0.2 * np.random.randn(t.size)              # 白噪声
x = signal + noise;         # 最终信号

nfft = 2048
######% 1 计算周期图法的功率谱密度

N2 = 512
window_hann = scipy.signal.windows.hann(N2)   # haning
window_hamm = scipy.signal.windows.hamming(N2)   # haming
f, Pxx_periodogram = scipy.signal.periodogram(x, fs, window = window_hann, nfft = N2) # window = window_hann, nfft = N2

## 手写
# def periodogram_method(signal, fs, N):
#     X = np.fft.fft(signal, n = N)
#     Pxx = np.abs(X)**2/N
#     Pxx = Pxx[0:int(N/2) + 1]
#     f = np.arange(0, N/2+1) * (fs/N)
#     return f, Pxx,
def periodogram_method(signal, fs, N):
    X = np.fft.fft(signal, n = N)
    Pxx = np.abs(X)**2/(N * fs)
    Pxx = Pxx[0:int(N/2) + 1]
    Pxx[1:int(N/2)] = 2 * Pxx[1:int(N/2)]
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
# f, Pxx_periodogram = periodogram_method(x, fs, nfft)

######% 2 自相关函数法
def correlogram_method(signal, fs, N):
    Rxx, lag = xcorr(signal, signal, normed = True, detrend = True, maxlags = signal.size - 1)
    Rxx = Rxx[N-1:]
    Rxx = np.fft.fft(Rxx, n = N)
    Pxx = np.abs(Rxx[0: int(N/2) + 1])
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
f1, Pxx_xcorr = correlogram_method(x, fs, x.size)

######% 3 Welch 方法
L = 256              # Welch方法中的子段长度
D = L//2              # 重叠长度
f2, Pxx_welch = scipy.signal.welch(x, fs, window = 'hann', nperseg = L, noverlap = D)

##### plot
fig, axs = plt.subplots(4, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, x, color = 'b', lw = 0.2, label = '模拟地震波信号')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("模拟地震波信号")
axs[0].legend()

axs[1].plot(f, 10*np.log10(Pxx_periodogram), color = 'b', label = '周期图法')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('功率谱密度 (dB/Hz)',)
axs[1].set_title("周期图法-功率谱密度 (dB/Hz)")
axs[1].legend()

axs[2].plot(f1, 10*np.log10(Pxx_xcorr), color = 'b', label = '自相关函数法')
axs[2].set_xlabel('频率 (Hz)',)
axs[2].set_ylabel('功率谱密度 (dB/Hz)',)
axs[2].set_title("自相关函数法-功率谱密度 (dB/Hz)")
axs[2].legend()

axs[3].plot(f2, 10*np.log10(Pxx_welch), color = 'b', label = 'welch方法')
axs[3].set_xlabel('频率 (Hz)',)
axs[3].set_ylabel('功率谱密度 (dB/Hz)',)
axs[3].set_title("Welch 方法-功率谱密度 (dB/Hz)")
axs[3].legend()

plt.show()
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  场景 4: 音频处理和降噪  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fs = 8000 # 采样频率
T = 2       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
f_signal = 1000  #  通信信号频率 (Hz)
A_signal = 1     #  信号幅度

signal = A_signal * np.cos(2*np.pi*f_signal*t)     # 通信信号
noise = 0.3 * np.random.randn(t.size)              # 白噪声
x = signal + noise;         # 最终信号

nfft = 2048
######% 1 计算周期图法的功率谱密度

N2 = nfft
window_hann = scipy.signal.windows.hann(N2)   # haning
window_hamm = scipy.signal.windows.hamming(N2)   # haming
f, Pxx_periodogram = scipy.signal.periodogram(x, fs, window = window_hann, nfft = N2) # window = window_hann, nfft = N2

## 手写
def periodogram_method(signal, fs, N):
    X = np.fft.fft(signal, n = N)
    Pxx = np.abs(X)**2/(N * fs)
    Pxx = Pxx[0:int(N/2) + 1]
    Pxx[1:int(N/2)] = 2 * Pxx[1:int(N/2)]
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
# f, Pxx_periodogram = periodogram_method(x, fs, nfft)

######% 2 自相关函数法
def correlogram_method(signal, fs, N):
    Rxx, lag = xcorr(signal, signal, normed = True, detrend = True, maxlags = signal.size - 1)
    Rxx = Rxx[N-1:]
    Rxx = np.fft.fft(Rxx, n = N)
    Pxx = np.abs(Rxx[0: int(N/2) + 1])
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
f1, Pxx_xcorr = correlogram_method(x, fs, x.size)

######% 3 Welch 方法
L = 256              # Welch方法中的子段长度
D = L//2              # 重叠长度
f2, Pxx_welch = scipy.signal.welch(x, fs, window = 'hann', nperseg = L, noverlap = D)

##### plot
fig, axs = plt.subplots(4, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, x, color = 'b', lw = 0.2, label = '周期图法')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("音频处理和降噪")
axs[0].legend()

axs[1].plot(f, 10*np.log10(Pxx_periodogram), color = 'b', label = '周期图法')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('功率谱密度 (dB/Hz)',)
axs[1].set_title("周期图法-功率谱密度 (dB/Hz)")
axs[1].legend()

axs[2].plot(f1, 10*np.log10(Pxx_xcorr), color = 'b', label = '自相关函数法')
axs[2].set_xlabel('频率 (Hz)',)
axs[2].set_ylabel('功率谱密度 (dB/Hz)',)
axs[2].set_title("自相关函数法-功率谱密度 (dB/Hz)")
axs[2].legend()

axs[3].plot(f2, 10*np.log10(Pxx_welch), color = 'b', label = 'welch方法')
axs[3].set_xlabel('频率 (Hz)',)
axs[3].set_ylabel('功率谱密度 (dB/Hz)',)
axs[3].set_title("Welch 方法-功率谱密度 (dB/Hz)")
axs[3].legend()

plt.show()
plt.close()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  4.3 功率谱密度的计算方  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%





















































































































































































































