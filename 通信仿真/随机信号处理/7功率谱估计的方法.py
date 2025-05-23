#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:14:45 2025

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


#%%  7.1 基于自相关函数的功率谱估计
def correlogram_method(signal, fs, N):
    Rxx, lag = xcorr(signal, signal, normed = True, detrend = True, maxlags = signal.size - 1)
    Rxx = Rxx[N-1:] # 取正半轴部分
    Rxx = np.fft.fft(Rxx, n = N)
    Pxx = np.abs(Rxx[0: int(N/2) + 1]) # 取前半部分（正频率）
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx

fs = 1000   # 采样频率
T = 1       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
f1 = 50     # 通信信号频率 (Hz)
f2 = 150
f3 = 300

x = np.sin(2*np.pi*f1*t) + 0.5 * np.sin(2*np.pi*f2*t) + 0.2 * np.sin(2 * np.pi * f3 * t) # 信号
x = x - np.mean(x)

hamming_window = scipy.signal.windows.hamming(x.size)
rect_window = scipy.signal.windows.boxcar(x.size)
blackman_window = scipy.signal.windows.blackman(x.size)

x_hamming = x * hamming_window;
x_rect = x * rect_window;
x_blackman = x * blackman_window;

r_ham, lag       = xcorr(x_hamming, x_hamming, normed = True, detrend = 1, maxlags = x_hamming.size - 1)
r_rect, lag      = xcorr(x_rect, x_rect, normed = True, detrend = 1, maxlags = x_hamming.size - 1)
r_blackman, lag  = xcorr(x_blackman, x_blackman, normed = True, detrend = 1, maxlags = x_hamming.size - 1)

f1, S_ham         = correlogram_method(x_hamming, fs, x_hamming.size)
f2, S_rect        = correlogram_method(r_rect, fs, x_rect.size)
f3, S_blackman    = correlogram_method(r_blackman, fs, x_blackman.size)

##### plot
fig, axs = plt.subplots(3, 4, figsize = (16, 12), constrained_layout = True)

# x
axs[0,0].plot(t, x, color = 'b', lw = 0.2, label = '去均值后的信号')
axs[0,0].set_xlabel('时间 (s)',)
axs[0,0].set_ylabel('幅度',)
axs[0,0].set_title("去均值后的信号")
axs[0,0].legend()

axs[0,1].plot(t, x_hamming, color = 'b', label = 'Hamming窗')
axs[0,1].set_xlabel('时间 (s)',)
axs[0,1].set_ylabel('幅度',)
axs[0,1].set_title("滤波后的信号 (时域)")
axs[0,1].legend()

axs[0,2].plot(t, x_rect, color = 'b', label = 'Hamming窗')
axs[0,2].set_xlabel('时间 (s)',)
axs[0,2].set_ylabel('幅度',)
axs[0,2].set_title("矩形窗")
axs[0,2].legend()

axs[0,3].plot(t, x_blackman, color = 'b', label = 'Hamming窗')
axs[0,3].set_xlabel('时间 (s)',)
axs[0,3].set_ylabel('幅度',)
axs[0,3].set_title("Blackman窗")
axs[0,3].legend()

axs[1,1].plot(lag/fs, r_ham, color = 'b', label = 'Hamming窗下的自相关')
axs[1,1].set_xlabel('时间滞后 (s)',)
axs[1,1].set_ylabel('自相关值',)
axs[1,1].set_title("Hamming窗下的自相关")
axs[1,1].legend()

axs[1,2].plot(lag/fs, r_rect, color = 'b', label = '矩形窗下的自相关')
axs[1,2].set_xlabel('时间滞后 (s)',)
axs[1,2].set_ylabel('自相关值',)
axs[1,2].set_title("矩形窗下的自相关")
axs[1,2].legend()

axs[1,3].plot(lag/fs, r_blackman, color = 'b', label = 'Blackman窗下的自相关')
axs[1,3].set_xlabel('时间滞后 (s)',)
axs[1,3].set_ylabel('自相关值',)
axs[1,3].set_title("Blackman窗下的自相关")
axs[1,3].legend()

axs[2,1].plot(f1, S_ham, color = 'b', label = 'Hamming窗下的功率谱')
axs[2,1].set_xlabel('频率 (Hz)',)
axs[2,1].set_ylabel('功率谱密度',)
axs[2,1].set_title("Hamming窗下的功率谱")
axs[2,1].legend()

axs[2,2].plot(f2, S_rect, color = 'b', label = '矩形窗下的功率谱')
axs[2,2].set_xlabel('频率 (Hz)',)
axs[2,2].set_ylabel('功率谱密度',)
axs[2,2].set_title("矩形窗下的功率谱")
axs[2,2].legend()

axs[2,3].plot(f3, S_blackman, color = 'b', label = 'Blackman窗下的功率谱')
axs[2,3].set_xlabel('频率 (Hz)',)
axs[2,3].set_ylabel('功率谱密度',)
axs[2,3].set_title("Blackman窗下的功率谱")
axs[2,3].legend()

plt.suptitle(f"不同窗函数下的《基于自相关函数的功率谱估计》", fontsize = 22)
plt.show()
plt.close()

#%% 7.2 快速傅里叶变换（FFT）法
fs = 1000 # 采样频率
T = 1       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
f1 = 50  # 通信信号频率 (Hz)
f2 = 150
f3 = 300

x = np.sin(2*np.pi*f1*t) + 0.5 * np.sin(2*np.pi*f2*t) + 0.2 * np.sin(2 * np.pi * f3 * t) # 信号
x = x - np.mean(x)
window = scipy.signal.windows.hann(x.size)

plt.figure(figsize = (16, 12), constrained_layout = True)
#% -------- 信号长度的影响 --------
#% 选择不同的信号长度 (确保信号长度不超过原始信号长度)
Ns = [256, 512, 1000];  # 信号长度，不超过1000点
for i in range(len(Ns)):
    x_segment = x[:Ns[i]]    # 提取不同长度的信号段
    x_windowed = x_segment * window[:Ns[i]]
    N = 2**(int(np.ceil(np.log2(x_windowed.size))))
    X = np.fft.fft(x_windowed, N)
    f = np.arange(0, N/2) * (fs/N)

    # 计算功率谱密度 (PSD)
    Pxx = np.abs(X[:int(N/2)])**2 / (fs * N)

    plt.subplot(3, 3, i+1)
    plt.plot(f, 10 * np.log10(Pxx), 'b')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度 (dB/Hz)')
    plt.title(f'信号长度 {Ns[i]}')

#% -------- 零填充的影响 --------
x_windowed = x * window # 使用完整信号，应用窗口函数
N_padding = [1024, 2048, 4096]; # 不同的零填充长度

for i in range(len(N_padding)):
    N = N_padding[i]
    X = np.fft.fft(x_windowed, n = N)
    f = np.arange(0, N/2) * (fs/N)
    Pxx = np.abs(X[:int(N/2)])**2 / (fs * N)
    plt.subplot(3, 3, i+4)
    plt.plot(f, 10 * np.log10(Pxx), 'r')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度 (dB/Hz)')
    plt.title(f'零填充长度 {N_padding[i]}')

#% -------- 重叠率的影响 --------
#% 将信号分成多个段落，计算不同重叠率下的 PSD
overlap_ratios = [0.0, 0.5, 0.75];
window_length = 256
n_overlap = [0, np.round(window_length*0.5), round(window_length*0.75)];

for i in range(len(overlap_ratios)):
    win = scipy.signal.windows.hann(window_length)
    [f, S] = scipy.signal.welch(x,  fs,  window = win, noverlap = n_overlap[i], nfft = N_padding[-1] )

    plt.subplot(3, 3, i+7)
    plt.plot(f, 10 * np.log10(S), 'g')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度 (dB/Hz)')
    plt.title(f'重叠率 {overlap_ratios[i]}')
plt.suptitle("《快速傅里叶变换（FFT）法》信号长度/零填充长度/重叠率的影响", fontsize = 22)
plt.show()
plt.close()

#############  快速傅里叶变换（FFT）法: 窗口函数比较
fs = 1000 # 采样频率
T = 1       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
f1 = 50  # 通信信号频率 (Hz)
f2 = 150
f3 = 300

x = np.sin(2*np.pi*f1*t) + 0.5 * np.sin(2*np.pi*f2*t) + 0.2 * np.sin(2 * np.pi * f3 * t) # 信号
x = x - np.mean(x)
plt.figure(figsize = (18, 12), constrained_layout = True)
windows = ['Hanning', 'Hamming', 'Blackman']
windowfs = [scipy.signal.windows.hann, scipy.signal.windows.hamming, scipy.signal.windows.blackman]
for i in range(len(windows)):
    windowsf = windowfs[i](x.size)
    windowed_signal = x * windowsf
    N = 2**(int(np.ceil(np.log2(windowed_signal.size))))
    X = np.fft.fft(windowed_signal, n = N)
    X_magnitude = 2 * np.abs(X/N)[:int(N/2)]
    f = np.arange(0, N/2) * (fs/N)
    Pxx = np.abs(X[:int(N/2)])**2 / (fs * N)

    plt.subplot(3, 4, 1 + i*4)
    plt.plot(windowsf, 'r')
    plt.xlabel('样本点')
    plt.ylabel('幅值')
    plt.title(f'{windows[i]}窗函数')

    plt.subplot(3, 4, 2 + i*4)
    plt.plot(t, windowed_signal, 'g')
    plt.xlabel('时间 (s)')
    plt.ylabel('幅值')
    plt.title(f'应用{windows[i]}窗后的信号')

    plt.subplot(3, 4, 3 + i*4)
    plt.plot(f, X_magnitude, 'k')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅值')
    plt.title(f'{windows[i]}窗-FFT')

    plt.subplot(3, 4, 4 + i*4)
    plt.plot(f, 10*np.log10(Pxx), 'm')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度 (dB/Hz)')
    plt.title(f'{windows[i]}窗-功率谱密度')
plt.suptitle("《快速傅里叶变换（FFT）法》窗口函数比较", fontsize = 22)
plt.show()
plt.close()

#%% 7.3 滑动窗法（Welch 方法）
# Welch 方法的步骤：
#     信号分段：将输入信号划分为多个长度为L的重叠段。每段之间的重叠率通常为 50% 或更高。
#     窗口化处理：对每一段信号应用一个窗口函数w(t)，得到窗口化后的信号 X_w(t) = X_m(t)*w(t)
#     FFT 计算：对每段窗口化后的信号 X_w(t)进行快速傅里叶变换（FFT），得到频谱 X_w(f)。
#     功率谱密度计算：计算每段信号的功率谱密度 P_m(f) = |X_w(f)|^2/L
#     功率谱密度平均：将所有段的功率谱密度进行平均，得到最终的功率谱估计 P_{wlech}(f) = \sum_{m=1}^M P_m(f) / M
#     其中，M为段的数量。

fs = 1000                 # 采样频率
T = 1                     # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
f1 = 100                  # 通信信号频率 (Hz)
f2 = 200
x = np.cos(2*np.pi*f1*t) + 0.5 * np.sin(2*np.pi*f2*t) + np.random.randn(t.size)
segment_lengths = [64, 128, 256]      # 不同的分段长度
overlap_ratios = [0.25, 0.5, 0.75]    # 不同的重叠率

#>>>>>>>>>>>>>>>> 1. 分析分段长度的影响
plt.figure(figsize = (16, 4), constrained_layout = True)
for i in range(len(segment_lengths)):
    [f, S] = scipy.signal.welch(x, fs, nperseg = segment_lengths[i], noverlap = segment_lengths[i]//2, )
    plt.subplot(1, 3, i+1)
    plt.plot(f, 10 * np.log10(S))
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率/频率 (dB/Hz)')
    plt.title(f'分段长度 = {segment_lengths[i]}')
    plt.suptitle("《滑动窗法（Welch 方法）》分析分段长度影响", fontsize = 22)
#>>>>>>>>>>>>>>>> 2. 分析重叠率的影响
plt.figure(figsize = (16, 4), constrained_layout = True)
for i in range(len(overlap_ratios)):
    [f, S] = scipy.signal.welch(x, fs, nperseg = 128, noverlap = 128 * overlap_ratios[i], )
    plt.subplot(1, 3, i+1)
    plt.plot(f, 10 * np.log10(S))
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率/频率 (dB/Hz)')
    plt.title(f'重叠率 = {overlap_ratios[i]}')
    plt.suptitle("《滑动窗法（Welch 方法）》分析重叠率的影响", fontsize = 22)
#>>>>>>>>>>>>>>>> 3. 分析窗口函数的影响
window_types = ['boxcar', 'hamming', 'hann', 'blackman']  # 不同的窗口函数
windowfs = [scipy.signal.windows.boxcar, scipy.signal.windows.hamming, scipy.signal.windows.hann, scipy.signal.windows.blackman]

plt.figure(figsize = (10, 8), constrained_layout = True)
for i in range(len(window_types)):
    widf = windowfs[i](128)
    [f, S] = scipy.signal.welch(x, fs, window = widf, nperseg = 128, noverlap = 128 * 0.5, )
    plt.subplot(2, 2, i+1)
    plt.plot(f, 10 * np.log10(S))
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率/频率 (dB/Hz)')
    plt.title(f'{window_types[i]}窗口')
    plt.suptitle("《滑动窗法（Welch 方法）》分析窗口函数的影响", fontsize = 22)

#%% 7.4 周期图法

fs = 1000 # 采样频率
T = 1       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
f1 = 50  # 通信信号频率 (Hz)
f2 = 120
x = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + 1/2 * np.random.randn(t.size)
x = x - np.mean(x)

N1 = 256
N2 = 512
window_hamm = scipy.signal.windows.hamming(N1)   # haming
f1, Pxx_periodogram1 = scipy.signal.periodogram(x, fs, window = window_hamm, nfft = N1 ) #
window_hamm = scipy.signal.windows.hamming(N2)   # haming
f2, Pxx_periodogram2 = scipy.signal.periodogram(x, fs, window = window_hamm, nfft = N2 ) #

window_hann = scipy.signal.windows.hann(N2)   # haning
window_hamm = scipy.signal.windows.hamming(N2)   # haming
fhan, Pxx_periodhan = scipy.signal.periodogram(x, fs, window = window_hann, nfft = N2 ) #
fhamm, Pxx_periodhamm = scipy.signal.periodogram(x, fs, window = window_hamm, nfft = N2 ) #

##### plot
fig, axs = plt.subplots(2, 1, figsize = (8, 10), constrained_layout = True)

# x
axs[0].plot(f1, 10 * np.log10(Pxx_periodogram1), color = 'b', lw = 2, label = f'N = {N1}')
axs[0].plot(f2, 10 * np.log10(Pxx_periodogram2), color = 'r', lw = 2, label = f'N = {N2}')
axs[0].set_xlabel('频率 (Hz)',)
axs[0].set_ylabel('功率谱 (dB)',)
axs[0].set_title("不同信号长度下的功率谱")
axs[0].legend()

# 功率谱密度
axs[1].plot(fhan, 10 * np.log10(Pxx_periodhan) , color = 'g', label = 'Hanning窗')
axs[1].plot(fhamm, 10 * np.log10(Pxx_periodhamm) , color = 'r', label = 'Hamming窗')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('功率谱 (dB)',)
axs[1].set_title("不同窗口函数下的功率谱")
axs[1].legend()

plt.suptitle("《周期图法periodogram》信号长度/窗口函数的影响", fontsize = 22)
plt.show()
plt.close()















































































































































































































































































































































