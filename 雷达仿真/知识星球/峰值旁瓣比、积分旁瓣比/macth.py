#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 21:17:13 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann

# =============================================================================
# LFM信号加窗匹配滤波 - Python版本
# =============================================================================

# LFM信号的参数
T = 10e-6                          # 信号时宽
B = 30e6                           # 信号带宽
K = B / T                          # 线性调频系数
fc = 0                             # 信号载频
a = 20                             # 过采样因子
fs = a * B                         # 采样率Fs
Ts = 1 / fs                        # 采样间隔
t0 = 0                             # 时延
tc = 0                             # tc=0为基带信号，tc不为0是为非基带信号
N = int(T / Ts)                    # 采样点数

# 信号生成
t = np.linspace(-T/2, T/2, N)
st = np.exp(1j * np.pi * K * (t - tc)**2)         # 调频信号
ht = np.exp(-1j * np.pi * K * (t + tc)**2)        # 匹配滤波器

# 加窗效应
M = len(ht)                         # 窗的长度
w = hann(M)                      # 加的窗函数的类型
sout_win = np.convolve(st, (ht * w), mode='same')  # 加窗后的输出
sout_dB_win = 20 * np.log10(np.abs(sout_win) / np.max(np.abs(sout_win)))  # 加窗后输出归一化的脉压后的幅度（dB）

# 第一个图形：信号和滤波器的实部与虚部
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# 信号实部
ax[0, 0].plot(t * 1e6, np.real(st))
ax[0, 0].grid(True)
ax[0, 0].set_title('信号实部')
ax[0, 0].set_xlabel('时间/us')
ax[0, 0].set_ylabel('信号实部幅度')

# 信号虚部
ax[0, 1].plot(t * 1e6, np.imag(st))
ax[0, 1].grid(True)
ax[0, 1].set_title('信号虚部')
ax[0, 1].set_xlabel('时间/us')
ax[0, 1].set_ylabel('信号虚部幅度')

# 滤波器实部
ax[1, 0].plot(t * 1e6, np.real(ht))
ax[1, 0].grid(True)
ax[1, 0].set_title('滤波器实部')
ax[1, 0].set_xlabel('时间/us')
ax[1, 0].set_ylabel('滤波器实部幅度')

# 滤波器虚部
ax[1, 1].plot(t * 1e6, np.imag(ht))
ax[1, 1].grid(True)
ax[1, 1].set_title('滤波器虚部')
ax[1, 1].set_xlabel('时间/us')
ax[1, 1].set_ylabel('滤波器虚部幅度')

plt.tight_layout()
plt.show()
plt.close()


# 第二个图形：加窗后的脉压结果
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t * 1e6, sout_dB_win)
ax.grid(True)
ax.set_title('基带信号脉压结果')
ax.set_xlabel('时间/us')
ax.set_ylabel('压缩后的幅度（dB）')
plt.show()
plt.close()
