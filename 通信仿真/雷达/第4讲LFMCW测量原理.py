#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:38:44 2025

@author: jack
https://zhuanlan.zhihu.com/p/687473210

第4讲 -- 线性调频连续波LFMCW测量原理：测距、测速、测角

"""



import numpy as np
import scipy
import matplotlib.pyplot as plt

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22
def freqDomainView(x, Fs, FFTN = None, type = 'double'): # N为偶数
    if FFTN == None:
        FFTN = 2**int(np.ceil(np.log2(x.size)))
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

# 参数设置
maxR = 200;     # 最大探测距离
rangeRes = 1;   # 距离分辨率
maxV = 70;      # 最大检测目标的速度
fc = 77e9;      # 工作频率（载频）
c = 3e8;        # 光速
r0 = 90;        # 目标距离设置 (max = 200m)
v0 = 10;        # 目标速度设置 (min =-70m/s, max=70m/s)

# 产生信号
B = c/(2 * rangeRes)       # 150MHz
Tchirp = 5.5 * 2 * maxR / c #  扫频时间 (x-axis), 5.5 = sweep time should be at least 5 o 6 times the round trip time
S = B/Tchirp               # 调频斜率
phi = 0                    # 初相位
Nchirp = 128              #  chirp数量
Ns = 4096                  # ADC采样点数
t = np.linspace(0, Nchirp*Tchirp, Nchirp*Ns)   # 发射信号和接收信号的采样时间
Fs = Ns/Tchirp
ft = fc * t + S/2 * t**2
Tx = np.cos(2 * np.pi * ft + phi)       # 发射信号
tau = Tchirp/6                          # 时延
fr = fc * (t- tau) + S*(t - tau)**2     # 回波信号频率
Rx = np.cos(2 * np.pi * fr + phi)       # 回波信号

### 经过混频器
Mix = Tx * Rx

f_, _, A, _, _, _  = freqDomainView(Mix[:Ns], Fs, type = 'double')
fig, axs = plt.subplots(1, 1, figsize = (4, 3), constrained_layout = True)
axs.plot(f_, A)
axs.set_title("Mix Signal FFT")
axs.set_xlabel("Frequency")
axs.set_ylabel("Amplitude")
plt.show()
plt.close()


## 混频经过低通滤波器,方法1
fpass = 30e5;      #  截止频率 fpass=30MHz
# Fs = 120e6;        #  采样频率 fs=120MHz
# Mix_filtered = lowpass(Mix(1:Ns), fpass, fs_lpf);
# order = 4
# Wn = 2 * fpass / Fs
# [Bb, Ba] = scipy.signal.butter(order, Wn, 'low')
# Mix_filtered = scipy.signal.lfilter(Bb, Ba, Mix[0:Ns]) # 进行滤波
# # Mix_filtered = scipy.signal.filtfilt(Bb, Ba, Mix[0:Ns] )

## 混频经过低通滤波器,方法2
h = scipy.signal.firwin(int(12),  cutoff = fpass, fs = Fs, pass_zero = "lowpass")
Mix_filtered = scipy.signal.lfilter(h, 1, Mix[0:Ns]) # 进行滤波


## 计算差频
N_fft = 1024
f = np.arange(N_fft) / 2 * Ns
Mix_filtered_fft = np.abs(scipy.fft.fft(Mix_filtered, N_fft))
Mix_filtered_fft = 10 * np.log10(Mix_filtered_fft)

# 结果可视化
fig, axs = plt.subplots(3, 2, figsize = (10, 10), constrained_layout = True)

# 发射回波信号
axs[0,0].plot(t[0:Ns], Tx[0:Ns])
axs[0,0].set_title("发射信号时域波形图")
axs[0,0].set_xlabel("时间(s)")
axs[0,0].set_ylabel("幅值")

# 接收回波信号
axs[0,1].plot(t[0:Ns], Rx[0:Ns])
axs[0,1].set_title("回波信号时域波形图")
axs[0,1].set_xlabel("时间(s)")
axs[0,1].set_ylabel("幅值")

axs[1,0].plot(t[0:Ns], ft[0:Ns], label = "发射信号频率")
axs[1,0].plot(t[0:Ns], fr[0:Ns], label = "回波信号频率")
# axs[1,0].set_title("Frequency of Tx/Rx signal")
axs[1,0].set_xlabel("时间(s)")
axs[1,0].set_ylabel("幅值")
axs[1,0].legend()

axs[1,1].plot(t[0:Ns], Mix[0:Ns])
axs[1,1].set_title("混频信号")
axs[1,1].set_xlabel("时间(s)")
axs[1,1].set_ylabel("幅值")

axs[2,0].plot(t[0:Ns], Mix_filtered[0:Ns])
axs[2,0].set_title("差频信号")
axs[2,0].set_xlabel("时间(s)")
axs[2,0].set_ylabel("幅值")

axs[2,1].plot(f, Mix_filtered_fft)
axs[2,1].set_title("差频信号的频谱图")
axs[2,1].set_xlabel("频率")
axs[2,1].set_ylabel("幅度（dB）")

plt.show()
plt.close()



























































































