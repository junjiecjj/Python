#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 14:43:03 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# 产生扫频干扰片段
jnr = 1000  # 干扰噪声比
Fs = 2000                    # 采样频率
T = 1/Fs                     # 采样时间
L = 2 * Fs                   # 信号长度 2秒
t = np.arange(0, L) * T      # 时间向量

# 生成线性调频信号（扫频信号）
om = 2 * np.pi * 10
be = 2 * np.pi * 50
phi = 0
x = np.exp(1j * 0.5 * be * t**2 + 1j * om * t + 1j * phi)
# x = np.exp(1j * om * t + 1j * phi)  # 单频信号（注释掉）

x = np.real(x)  # 取实部

# 添加高斯白噪声
signal_power = np.mean(x**2)
snr_linear = 10**(jnr / 10)
noise_power = signal_power / snr_linear
noise = np.random.normal(0, np.sqrt(noise_power), len(x))
y = x + noise

# 绘制带噪声的信号
plt.figure()
plt.plot(Fs * t, y)
plt.title('Signal Corrupted with Zero-Mean Random Noise')
plt.xlabel('time (milliseconds)')
plt.grid(True)
plt.show()

# 绘制扫频干扰时域图
plt.figure(figsize=(6, 3), facecolor='white')
plt.plot(1000 * t[:L//2], y[:L//2], color=[0, 0.4, 0.8])
plt.grid(True)
plt.title('Sweeping Jam')
plt.xlabel('time (milliseconds)')
plt.tight_layout()
plt.show()

# 计算FFT
NFFT = 2**int(np.ceil(np.log2(L)))  # 下一个2的幂次
Y = np.fft.fft(y, NFFT) / L
f = Fs/2 * np.linspace(0, 1, NFFT//2 + 1)

# 绘制单边幅度谱（dB表示）
plt.figure(figsize=(6, 3), facecolor='white')
plt.plot(f, 2 * 10 * np.log10(np.abs(Y[:NFFT//2 + 1])), color=[0, 0.4, 0.8])
plt.grid(True)
plt.title('Single-Sided Amplitude Spectrum of Sweeping Jam')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|Y(f)|(dB)')
plt.xlim([0, 500])
plt.tight_layout()
plt.show()
