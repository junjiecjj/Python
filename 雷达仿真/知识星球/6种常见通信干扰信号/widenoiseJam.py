#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 14:47:27 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import kaiserord, firwin, lfilter, freqz
import warnings
warnings.filterwarnings('ignore')

# 产生窄带噪声片段
jnr = 20  # 干扰噪声比
Fs = 2000                    # 采样频率
T = 1/Fs                     # 采样时间
L = 2 * Fs                   # 信号长度
t = np.arange(0, L) * T      # 时间向量
P = 1                        # 噪声功率

# 生成白噪声
y = np.random.normal(0, np.sqrt(P), L)

# 绘制白噪声时域图
plt.figure()
plt.plot(1000 * t[:L//2], y[:L//2])
plt.title('White Noise')
plt.xlabel('time (milliseconds)')
plt.grid(True)
plt.show()

# 计算FFT
NFFT = 2**int(np.ceil(np.log2(L)))  # 下一个2的幂次
Y = np.fft.fft(y, NFFT) / L
f = Fs/2 * np.linspace(0, 1, NFFT//2 + 1)

# 绘制单边幅度谱
plt.figure()
plt.plot(f, 2 * 10 * np.log10(np.abs(Y[:NFFT//2 + 1])))
plt.title('Single-Sided Amplitude Spectrum of White Noise')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|Y(f)|(dB)')
plt.grid(True)
plt.show()

# 带通滤波器设计
WI = 50  # 窄带窗口
FJ = 600  # 干扰频率
fcuts = [FJ - WI/2, FJ - WI/4, FJ + WI/4, FJ + WI/2]
mags = [0, 1, 0]
devs = [0.05, 0.05, 0.05]

# 计算Kaiser窗口参数
ripple_db = -20 * np.log10(np.min(devs))
width = min(fcuts[1] - fcuts[0], fcuts[3] - fcuts[2]) / (Fs/2)  # 归一化宽度
n, beta = kaiserord(ripple_db, width)

# 设计带通滤波器
wn = [fcuts[0]/(Fs/2), fcuts[3]/(Fs/2)]  # 归一化截止频率
h = firwin(n + 1, wn, window=('kaiser', beta), pass_zero=False)
Y_bp = lfilter(h, 1, y)  # 时域滤波结果

# 绘制滤波器频率响应 - 严格按照MATLAB的freqz格式
plt.figure()
w, h_freq = freqz(h, worN=8000)
# 上子图 - 幅度响应
plt.subplot(2, 1, 1)
plt.plot(w/np.pi, 20 * np.log10(np.abs(h_freq)))
plt.ylabel('Magnitude (dB)')
plt.grid(True)
# 下子图 - 相位响应
plt.subplot(2, 1, 2)
plt.plot(w/np.pi, np.angle(h_freq))
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制原始噪声和滤波后噪声
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(1000 * t[:L//2], y[:L//2])
plt.title('White Noise')
plt.xlabel('time (milliseconds)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(1000 * t[:L//2], Y_bp[:L//2])
plt.title('Narrow Band White Noise')
plt.xlabel('time (milliseconds)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制窄带噪声干扰时域图
plt.figure(figsize=(6, 3), facecolor='white')
plt.plot(1000 * t[:L//2], Y_bp[:L//2], color=[0, 0.4, 0.8])
plt.grid(True)
plt.title('Narrowband Noise jam')
plt.xlabel('time (milliseconds)')
plt.tight_layout()
plt.show()

# 计算滤波后信号的频谱
NFFT = 2**int(np.ceil(np.log2(L)))  # 下一个2的幂次
Y = np.fft.fft(Y_bp, NFFT) / L
f = Fs/2 * np.linspace(0, 1, NFFT//2 + 1)

# 绘制滤波后信号的频谱
plt.figure(figsize=(6, 3), facecolor='white')
plt.plot(f, 2 * 10 * np.log10(np.abs(Y[:NFFT//2 + 1])), color=[0, 0.4, 0.8])
plt.grid(True)
plt.title('Single-Sided Amplitude Spectrum of Narrowband Noise jam')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|Y(f)|(dB)')
plt.tight_layout()
plt.show()
