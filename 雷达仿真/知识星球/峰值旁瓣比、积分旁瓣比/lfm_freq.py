#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 21:16:47 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# LFM信号不同采样率下的频谱分析 - 统一绘图风格版本
# =============================================================================

# 信号参数
T = 10e-6                          # 时宽10us
B = 30e6                           # 线性调频信号的带宽30MHz
K = B / T                          # 线性调频系数
a = np.array([0.6, 0.8, 1, 1.2, 1.4, 1.6, 2, 4, ])        # 过采样倍数
Fs = a * B                         # 采样率
Ts = 1.0 / Fs                      # 采样间隔
N = T / Ts                         # 采样点数
k0 = len(a)

for k in range(k0):
    t = np.linspace(-T/2, T/2, int(N[k]))
    St = np.exp(1j * np.pi * K * t**2)

    # 计算频谱
    freq = np.linspace(-B/2, B/2, int(N[k]))
    spectrum = np.fft.fftshift(np.abs(np.fft.fft(St)))

    # 使用统一绘图风格
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(freq * 1e-6, spectrum, linewidth=2)
    ax.grid(True)
    ax.set_title(f'当采样率为{a[k]}倍时，线性调频信号的幅频特性')
    ax.set_xlabel('Frequency in MHz')
    ax.set_ylabel('幅度')
    plt.show()
    plt.close()
