#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:13:20 2025

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
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]    # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12

fs = 1000                  # 采样频率
T = 1                      # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs)  # 时间向量
f1 = 50                    # 通信信号频率 (Hz)
f2 = 120

signal = np.cos(2*np.pi*f1*t) + 0.5 * np.cos(2*np.pi*f2*t)  # 通信信号
noise =  np.random.randn(t.size)                            # 白噪声
x = signal + noise                                          # 最终信号
nfft = x.size
######% 1 计算周期图法的功率谱密度

N2 = x.size
window_hann = scipy.signal.windows.hann(N2)   # haning
window_hamm = scipy.signal.windows.hamming(N2)   # haming
# f, Pxx_periodogram = scipy.signal.periodogram(x, fs, window = window_hann, nfft = N2) # window = window_hann, nfft = N2

# ## 手写
# def periodogram_method(signal, fs, N):
#     X = np.fft.fft(signal, n = N)
#     Pxx = np.abs(X)**2/N
#     Pxx = Pxx[0:int(N/2) + 1]
#     f = np.arange(0, N/2+1) * (fs/N)
#     return f, Pxx,
## 手写:计算周期图法的功率谱密度
def periodogram_method(signal, fs, N):
    X = np.fft.fft(signal, n = N)
    Pxx = np.abs(X)**2/(N * fs)
    Pxx = Pxx[0:int(N/2) + 1]
    Pxx[1:int(N/2)] = 2 * Pxx[1:int(N/2)]
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
f, Pxx_periodogram = periodogram_method(x, fs, nfft)

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

#####>>>>>>>>>>>>>>>>    plot 1 计算周期图法的功率谱密度
fig, axs = plt.subplots(3, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

# % 非负性
axs[0].plot(f, Pxx_periodogram, color = 'b', lw = 0.2, label = '周期图法')
axs[0].set_xlabel('频率 (Hz)',)
axs[0].set_ylabel('功率谱密度',)
axs[0].set_title("功率谱密度的非负性")
# axs[0].legend()

# % 对称性
axs[1].plot(f, Pxx_periodogram, color = 'b', label = 'S_X(f)')
axs[1].plot(-f, Pxx_periodogram, color = 'r', label = 'S_X(-f)')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('功率谱密度 (dB/Hz)',)
axs[1].set_title("功率谱密度的对称性")
axs[1].legend()

# % Parseval定理验证: 信号的总功率等于其功率谱密度在整个频域的积分。
total_power_time_domain = np.mean(x**2)
total_power_freq_domain = np.sum(Pxx_periodogram) * (f[2] - f[1]) # * 2/fs

axs[2].bar(["时域总功率", "频域总功率", ], [total_power_time_domain, total_power_freq_domain], label = ["时域总功率", "频域总功率",  ], color = ["b","orange","g"])
axs[2].legend()

plt.suptitle("periodogram")
plt.show()
plt.close()

#####>>>>>>>>>>>>>>>> plot 2 自相关函数法
fig, axs = plt.subplots(3, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

# % 非负性
axs[0].plot(f1, Pxx_xcorr, color = 'b', lw = 0.2, label = '周期图法')
axs[0].set_xlabel('频率 (Hz)',)
axs[0].set_ylabel('功率谱密度',)
axs[0].set_title("功率谱密度的非负性")
# axs[0].legend()

# % 对称性
axs[1].plot(f1, Pxx_xcorr, color = 'b', label = 'S_X(f)')
axs[1].plot(-f1, Pxx_xcorr, color = 'r', label = 'S_X(-f)')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('功率谱密度 (dB/Hz)',)
# axs[1].set_title("功率谱密度的对称性")
# # axs[1].legend()

# % Parseval定理验证
total_power_time_domain = np.mean(x**2)
total_power_freq_domain = np.sum(Pxx_xcorr) * (f1[2] - f1[1]) * 2/fs

axs[2].bar(["时域总功率", "频域总功率", ], [total_power_time_domain, total_power_freq_domain], label = ["时域总功率", "频域总功率",  ], color = ["b","orange","g"])
axs[2].legend()

plt.suptitle("xcorr")
plt.show()
plt.close()

#####>>>>>>>>>>>>>>>>  plot 3 Welch 方法
fig, axs = plt.subplots(3, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

# % 非负性
axs[0].plot(f2, Pxx_welch, color = 'b', lw = 0.2, label = '周期图法')
axs[0].set_xlabel('频率 (Hz)',)
axs[0].set_ylabel('功率谱密度',)
axs[0].set_title("功率谱密度的非负性")
# axs[0].legend()

# % 对称性
axs[1].plot(f2, Pxx_welch, color = 'b', label = 'S_X(f)')
axs[1].plot(-f2, Pxx_welch, color = 'r', label = 'S_X(-f)')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('功率谱密度 (dB/Hz)',)
# axs[1].set_title("功率谱密度的对称性")
# # axs[1].legend()

# % Parseval定理验证
total_power_time_domain = np.mean(x**2)
total_power_freq_domain = np.sum(Pxx_welch) * (f2[2] - f2[1])
axs[2].bar(["时域总功率", "频域总功率", ], [total_power_time_domain, total_power_freq_domain], label = ["时域总功率", "频域总功率",  ], color = ["b","orange","g"])
axs[2].legend()

plt.suptitle("welch")
plt.show()
plt.close()


#%% 5.2 功率谱密度的实际意义

fs = 1000 # 采样频率
T = 1       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
f1 = 50  # 通信信号频率 (Hz)
f2 = 150

signal = np.cos(2*np.pi*f1*t) + 0.5 * np.cos(2*np.pi*f2*t)  # 通信信号
noise =  0.3 * np.random.randn(t.size)              # 白噪声
x = signal + noise;         # 最终信号
nfft = x.size
######% 1 计算周期图法的功率谱密度
N2 = 512
window_hann = scipy.signal.windows.hann(N2)   # haning
window_hamm = scipy.signal.windows.hamming(N2)   # haming
# f, Pxx = scipy.signal.periodogram(x, fs, ) # window = window_hann, nfft = N2

## 手写:计算周期图法的功率谱密度
def periodogram_method(signal, fs, N):
    X = np.fft.fft(signal, n = N)
    Pxx = np.abs(X)**2/(N * fs)
    Pxx = Pxx[0:int(N/2) + 1]
    Pxx[1:int(N/2)] = 2 * Pxx[1:int(N/2)]
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
# f, Pxx = periodogram_method(x, fs, nfft)

######% 2 自相关函数法
def correlogram_method(signal, fs, N):
    Rxx, lag = xcorr(signal, signal, normed = True, detrend = True, maxlags = signal.size - 1)
    Rxx = Rxx[N-1:]
    Rxx = np.fft.fft(Rxx, n = N)
    Pxx = np.abs(Rxx[0: int(N/2) + 1])
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
f, Pxx = correlogram_method(x, fs, nfft)

######% 3 Welch 方法
L = 256              # Welch方法中的子段长度
D = L//2              # 重叠长度
# f, Pxx = scipy.signal.welch(x, fs, window = 'hann', nperseg = L, noverlap = D)

#============= IIR -巴特沃兹低通滤波器  ================
lf = 100    # 通带截止频率200Hz
Fc = 1000   # 阻带截止频率1000Hz
Rp = 1      # 通带波纹最大衰减为1dB
Rs = 40     # 阻带衰减为40dB
#-----------------计算最小滤波器阶数----------------
na = np.sqrt(10**(0.1*Rp)-1)
ea = np.sqrt(10**(0.1*Rs)-1)
order  = np.ceil(np.log10(ea/na)/np.log10(Fc/lf))  #巴特沃兹阶数
Wn = lf*2/fs
#---------------- 低通滤波  --------------------
### 方法1
[Bb, Ba] = scipy.signal.butter(order, Wn, 'low')
# ## [BW, BH] = scipy.signal.freqz(Bb, Ba)
# y = scipy.signal.lfilter(Bb, Ba, x) # 进行滤波
y = scipy.signal.filtfilt(Bb, Ba, x )

######% 1 计算周期图法的功率谱密度
N2 = 512
window_hann = scipy.signal.windows.hann(N2)   # haning
window_hamm = scipy.signal.windows.hamming(N2)   # haming
# fy, Pxxy = scipy.signal.periodogram(y, fs, ) # window = window_hann, nfft = N2

## 手写
# fy, Pxxy = periodogram_method(y, fs, nfft)

######% 2 自相关函数法
def correlogram_method(signal, fs, N):
    Rxx, lag = xcorr(signal, signal, normed = True, detrend = True, maxlags = signal.size - 1)
    Rxx = Rxx[N-1:]
    Rxx = np.fft.fft(Rxx, n = N)
    Pxx = np.abs(Rxx[0: int(N/2) + 1])
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx
fy, Pxxy = correlogram_method(y, fs, y.size)

######% 3 Welch 方法
L = 256              # Welch方法中的子段长度
D = L//2              # 重叠长度
# fy, Pxxy = scipy.signal.welch(y, fs, window = 'hann', nperseg = L, noverlap = D)

##### plot
fig, axs = plt.subplots(3, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, x, color = 'b', lw = 0.2, label = '周期图法')
axs[0].set_xlabel('时间 (秒)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("含噪声的复合信号")
# axs[0].legend()

# 功率谱密度
axs[1].plot(f, Pxx , color = 'b', label = 'S_X(f)')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('功率谱密度 (dB/Hz)',)
axs[1].set_title("信号的功率谱密度")
axs[1].legend()

# 功率谱密度
axs[2].plot(fy, Pxxy , color = 'b', label = 'S_Y(f)')
axs[2].set_xlabel('频率 (Hz)',)
axs[2].set_ylabel('功率谱密度 (dB/Hz)',)
axs[2].set_title("滤波后信号的功率谱密度")
axs[2].legend()

# plt.suptitle("welch")
plt.show()
plt.close()


# 信号的频率特征直接反映在功率谱密度中。例如：
# 正弦波信号：功率谱密度集中在信号的频率处，体现了单一频率成分。
# 方波信号：功率谱密度集中在基频及其奇次谐波处，反映了其周期性和谐波特性。
# 白噪声信号：功率谱密度在整个频率范围内均匀分布，表明频率成分的均匀性。
# 短时脉冲信号：功率谱密度分布较宽，显示了脉冲信号的宽频带特性。
#%% 1. 正弦波信号的功率谱密度分析

fs = 1000 # 采样频率
T = 1       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
x =  np.cos(2*np.pi*100*t)  # 通信信号

nfft = 1024
######% 1 计算周期图法的功率谱密度
N2 = nfft
window_hann = scipy.signal.windows.hann(N2)   # haning
window_hamm = scipy.signal.windows.hamming(N2)   # haming
f, Pxx_periodogram = scipy.signal.periodogram(x[:N2], fs, ) # window = window_hann, nfft = N2

## 手写
def periodogram_method(signal, fs, N):
    X = np.fft.fft(signal, n = N)
    Pxx = np.abs(X)**2/N
    Pxx = Pxx[0:int(N/2) + 1]
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx,
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
L = x.size              # Welch方法中的子段长度
D = L//2              # 重叠长度
f2, Pxx_welch = scipy.signal.welch(x, fs, window = 'hann', nperseg = L, noverlap = D)

##### plot
fig, axs = plt.subplots(4, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, x, color = 'b', lw = 0.2, label = '周期图法')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("正弦波信号")
axs[0].legend()

axs[1].plot(f, Pxx_periodogram, color = 'b', label = '周期图法')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('功率谱密度 (dB/Hz)',)
axs[1].set_title("正弦波信号的功率谱密度 (dB/Hz)")
axs[1].legend()

axs[2].plot(f1, Pxx_xcorr, color = 'b', label = '自相关函数法')
axs[2].set_xlabel('频率 (Hz)',)
axs[2].set_ylabel('功率谱密度 (dB/Hz)',)
axs[2].set_title("正弦波信号的功率谱密度 (dB/Hz)")
axs[2].legend()

axs[3].plot(f2, Pxx_welch, color = 'b', label = 'welch方法')
axs[3].set_xlabel('频率 (Hz)',)
axs[3].set_ylabel('功率谱密度 (dB/Hz)',)
axs[3].set_title("正弦波信号的功率谱密度 (dB/Hz)")
axs[3].legend()

plt.show()
plt.close()

#%% 2. 方波信号的功率谱密度分析
fs = 1000 # 采样频率
T = 1       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
x =  scipy.signal.square(2 * np.pi * 100 * t)  # 通信信号

nfft = 1024
######% 1 计算周期图法的功率谱密度
N2 = nfft
window_hann = scipy.signal.windows.hann(N2)   # haning
window_hamm = scipy.signal.windows.hamming(N2)   # haming
# f, Pxx_periodogram = scipy.signal.periodogram(x, fs, ) # window = window_hann, nfft = N2

## 手写
def periodogram_method(signal, fs, N):
    X = np.fft.fft(signal, n = N)
    Pxx = np.abs(X)**2/N
    Pxx = Pxx[0:int(N/2) + 1]
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx,
f, Pxx_periodogram = periodogram_method(x, fs, nfft)

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
axs[0].set_title("方波信号")
axs[0].legend()

axs[1].plot(f, Pxx_periodogram, color = 'b', label = '周期图法')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('功率谱密度 (dB/Hz)',)
axs[1].set_title("方波信号的功率谱密度 (dB/Hz)")
axs[1].legend()

axs[2].plot(f1, Pxx_xcorr, color = 'b', label = '自相关函数法')
axs[2].set_xlabel('频率 (Hz)',)
axs[2].set_ylabel('功率谱密度 (dB/Hz)',)
axs[2].set_title("方波信号的功率谱密度 (dB/Hz)")
axs[2].legend()

axs[3].plot(f2, Pxx_welch, color = 'b', label = 'welch方法')
axs[3].set_xlabel('频率 (Hz)',)
axs[3].set_ylabel('功率谱密度 (dB/Hz)',)
axs[3].set_title("方波信号的功率谱密度 (dB/Hz)")
axs[3].legend()

plt.show()
plt.close()

#%% 3. 白噪声信号的功率谱密度分析

fs = 1000 # 采样频率
T = 10       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
x =  np.random.randn(t.size)  # 通信信号

nfft = 1024
######% 1 计算周期图法的功率谱密度

N2 = nfft
window_hann = scipy.signal.windows.hann(N2)   # haning
window_hamm = scipy.signal.windows.hamming(N2)   # haming
f, Pxx_periodogram = scipy.signal.periodogram(x, fs, ) # window = window_hann, nfft = N2

## 手写
def periodogram_method(signal, fs, N):
    X = np.fft.fft(signal, n = N)
    Pxx = np.abs(X)**2/N
    Pxx = Pxx[0:int(N/2) + 1]
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx,
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
axs[0].set_title("白噪声信号")
axs[0].legend()

axs[1].plot(f, Pxx_periodogram, color = 'b', label = '周期图法')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('功率谱密度 (dB/Hz)',)
axs[1].set_title("白噪声的功率谱密度 (dB/Hz)")
axs[1].legend()

axs[2].plot(f1, Pxx_xcorr, color = 'b', label = '自相关函数法')
axs[2].set_xlabel('频率 (Hz)',)
axs[2].set_ylabel('功率谱密度 (dB/Hz)',)
axs[2].set_title("白噪声的功率谱密度 (dB/Hz)")
axs[2].legend()

axs[3].plot(f2, Pxx_welch, color = 'b', label = 'welch方法')
axs[3].set_xlabel('频率 (Hz)',)
axs[3].set_ylabel('功率谱密度 (dB/Hz)',)
axs[3].set_title("白噪声的功率谱密度 (dB/Hz)")
axs[3].legend()

plt.show()
plt.close()

#%% 4. 短时脉冲信号的功率谱密度分析

fs = 1000 # 采样频率
T = 1       # 信号持续时间 (秒)
t = np.arange(0, T, 1/fs) # 时间向量
x = np.hstack((np.zeros(450), np.ones(100), np.zeros(450)))

nfft = 1024
######% 1 计算周期图法的功率谱密度

N2 = nfft
window_hann = scipy.signal.windows.hann(N2)   # haning
window_hamm = scipy.signal.windows.hamming(N2)   # haming
f, Pxx_periodogram = scipy.signal.periodogram(x, fs, ) # window = window_hann, nfft = N2

## 手写
def periodogram_method(signal, fs, N):
    X = np.fft.fft(signal, n = N)
    Pxx = np.abs(X)**2/N
    Pxx = Pxx[0:int(N/2) + 1]
    f = np.arange(0, N/2+1) * (fs/N)
    return f, Pxx,
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
axs[0].set_title("短时脉冲信号")
axs[0].legend()

axs[1].plot(f, Pxx_periodogram, color = 'b', label = '周期图法')
axs[1].set_xlabel('频率 (Hz)',)
axs[1].set_ylabel('功率谱密度 (dB/Hz)',)
axs[1].set_title("短时脉冲的功率谱密度 (dB/Hz)")
axs[1].legend()

axs[2].plot(f1, Pxx_xcorr, color = 'b', label = '自相关函数法')
axs[2].set_xlabel('频率 (Hz)',)
axs[2].set_ylabel('功率谱密度 (dB/Hz)',)
axs[2].set_title("短时脉冲的功率谱密度 (dB/Hz)")
axs[2].legend()

axs[3].plot(f2, Pxx_welch, color = 'b', label = 'welch方法')
axs[3].set_xlabel('频率 (Hz)',)
axs[3].set_ylabel('功率谱密度 (dB/Hz)',)
axs[3].set_title("短时脉冲的功率谱密度 (dB/Hz)")
axs[3].legend()

plt.show()
plt.close()














































































































































































































































































