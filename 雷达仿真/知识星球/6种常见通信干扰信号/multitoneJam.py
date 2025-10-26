#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 14:36:54 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# 产生多正弦波干扰片段
jnr = 200  # 干扰噪声比
Q = 10     # 多正弦波数量
Pl = 1 - 0.1 * np.random.rand(Q)  # 干扰功率

Fs = 2000                    # 采样频率
T = 1/Fs                     # 采样时间
L = 2 * Fs                   # 信号长度
t = np.arange(0, L) * T      # 时间向量

# 生成等间隔频率
fs0 = 100
deltafs = 10
fs = np.arange(fs0, fs0 + Q * deltafs, deltafs)

# 生成随机相位
theta = 2 * np.pi * np.random.rand(Q)

# 生成多频正弦波干扰（带随机相位）
xx = np.cos(2 * np.pi * fs.reshape(-1, 1) @ t.reshape(1, -1) + theta.reshape(-1, 1) @ np.ones((1, len(t))))
x = np.sqrt(Pl.reshape(-1, 1)) * xx  # 带随机功率的多频正弦波干扰
x = np.sum(x, axis=0)  # 对所有正弦波求和

# 数据预处理
x_mid = (np.max(x) + np.min(x)) / 2
x_nor = 2 * (x - x_mid) / (np.max(x) - np.min(x))  # 归一化到[-1,1]
x_dec = (x_nor - np.mean(x_nor)) / np.sqrt(np.var(x_nor))  # 去中心化

# 绘制原始数据和预处理数据
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(Fs * t[:int(5 * Fs / fs[0])], x[:int(5 * Fs / fs[0])])
plt.title('original data')
plt.xlabel('time (milliseconds)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(Fs * t[:int(5 * Fs / fs[0])], x_dec[:int(5 * Fs / fs[0])])
plt.title('pre_processed data')
plt.xlabel('time (milliseconds)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 添加高斯白噪声
signal_power = np.mean(x**2)
snr_linear = 10**(jnr / 10)
noise_power = signal_power / snr_linear
noise = np.random.normal(0, np.sqrt(noise_power), len(x))
y = x + noise

# 绘制多音干扰信号
plt.figure(figsize=(6, 3), facecolor='white')
plt.plot(Fs * t[:int(10 * Fs / fs[0])], y[:int(10 * Fs / fs[0])], color=[0, 0.4, 0.8])
plt.grid(True)
plt.title('multiple-tone jam')
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
plt.title('Single-Sided Amplitude Spectrum of multiple-tone jam')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|Y(f)|')
plt.ylim([1e-4, 1])
plt.tight_layout()
plt.show()
