#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 14:41:39 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# 产生单音干扰片段
Pl = 1
Fs = 1000                    # 采样频率
T = 1/Fs                     # 采样时间
L = 1000                     # 信号长度
t = np.arange(0, L) * T      # 时间向量

# 生成50Hz正弦波
fs = 50
x = np.sin(2 * np.pi * fs * t)

# 添加高斯白噪声
jnr = 200
signal_power = np.mean(x**2)
snr_linear = 10**(jnr / 10)
noise_power = signal_power / snr_linear
noise = np.random.normal(0, np.sqrt(noise_power), len(x))
y = x + noise

# 绘制单音干扰时域图
plt.figure(figsize=(6, 3), facecolor='white')
plt.plot(Fs * t[:100], y[:100], color=[0, 0.4, 0.8])
plt.grid(True)
plt.title('single-tone jam')
plt.xlabel('time (milliseconds)')
plt.tight_layout()
plt.show()

# 计算FFT
NFFT = 2**int(np.ceil(np.log2(L)))  # 下一个2的幂次
Y = np.fft.fft(y, NFFT) / L
f = Fs/2 * np.linspace(0, 1, NFFT//2 + 1)

# 绘制单边幅度谱（对数坐标）
plt.figure(figsize=(6, 3), facecolor='white')
plt.semilogy(f, 2 * np.abs(Y[:NFFT//2 + 1]), color=[0, 0.4, 0.8])
plt.grid(True)
plt.title('Single-Sided Amplitude Spectrum of single-tone jam')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|Y(f)|')
plt.ylim([1e-4, 1])
plt.tight_layout()
plt.show()
