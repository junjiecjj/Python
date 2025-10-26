#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 14:39:13 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import kaiserord, firwin, lfilter, freqz
import warnings
warnings.filterwarnings('ignore')

# 产生宽带噪声片段
Fs = 2000                    # 采样频率
T = 1/Fs                     # 采样时间
L = 2 * Fs                   # 信号长度
t = np.arange(0, L) * T      # 时间向量
P = 1                        # 噪声功率

# 生成白噪声（功率为P/600）
y = np.random.normal(0, np.sqrt(P/600), L)

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
WI = 600  # 宽带窗口
fl_kaiser = [WI, WI+50]
fl_mag = [1, 0]
fl_dev = [0.05, 0.01]

# 计算Kaiser窗口参数
ripple_db = -20 * np.log10(np.min(fl_dev))
width = (fl_kaiser[1] - fl_kaiser[0]) / (Fs/2)  # 归一化宽度
fl_n_kaiser, fl_beta = kaiserord(ripple_db, width)

# 设计滤波器
fl_wn = fl_kaiser[1] / (Fs/2)  # 归一化截止频率
h = firwin(fl_n_kaiser + 1, fl_wn, window=('kaiser', fl_beta))
Y_bp = lfilter(h, 1, y)  # 时域滤波结果

# 绘制滤波器频率响应
plt.figure()
w, h_freq = freqz(h, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5 * Fs * w / np.pi, np.abs(h_freq))
plt.title('Filter Frequency Response')
plt.ylabel('Magnitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(0.5 * Fs * w / np.pi, np.unwrap(np.angle(h_freq)))
plt.xlabel('Frequency (Hz)')
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
plt.title('Broad Band White Noise')
plt.xlabel('time (milliseconds)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 计算滤波后信号的频谱
NFFT = 2**int(np.ceil(np.log2(L)))  # 下一个2的幂次
Y = np.fft.fft(Y_bp, NFFT) / L
f = Fs/2 * np.linspace(0, 1, NFFT//2 + 1)

# 绘制滤波后信号的频谱
plt.figure()
plt.plot(f, 2 * 10 * np.log10(np.abs(Y[:NFFT//2 + 1])))
plt.title('Single-Sided Amplitude Spectrum of Narrow band White Noise')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|Y(f)|(dB)')
plt.grid(True)
plt.show()
