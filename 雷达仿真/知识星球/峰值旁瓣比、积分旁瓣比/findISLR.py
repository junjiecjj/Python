#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 21:09:58 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# LFM信号匹配滤波和ISLR计算 - 使用2x2子图形式
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

# 匹配输出
sout = np.convolve(st, ht, mode='same')
sout_dB = 20 * np.log10(np.abs(sout) / np.max(np.abs(sout)))  # 输出归一化的脉压后的幅度（dB）

# 寻找主瓣和计算ISLR
L = len(sout_dB)
maxdata = np.max(sout_dB)
I = np.argmax(sout_dB)

# 寻找第一零点
j = I
for i in range(I, L-1):
    if sout_dB[i+1] > sout_dB[i]:
        j = i
        break

# 计算主瓣功率和总功率
# 主瓣功率积分
p1 = np.sum((10**(sout_dB[2*I-j:j] / 20))**2 * (1/fs))

# 信号总功率（从主瓣中心向左右各扩展11/B秒）
left_idx = round(-11/B * fs) + I
right_idx = round(11/B * fs) + I
# 确保索引不越界
left_idx = max(0, left_idx)
right_idx = min(L, right_idx)

p = np.sum((10**(sout_dB[left_idx:right_idx] / 20))**2 * (1/fs))

# 计算ISLR
ISLR = 10 * np.log10((p - p1) / p1)

print(f"ISLR = {ISLR:.2f} dB")

# 使用2x2子图形式
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# 图1: LFM信号实部
ax[0, 0].plot(t * 1e6, np.real(st), linewidth=2)
ax[0, 0].grid(True)
ax[0, 0].set_xlabel('时间 (μs)')
ax[0, 0].set_ylabel('幅度')
ax[0, 0].set_title('LFM信号实部')

# 图2: LFM信号虚部
ax[0, 1].plot(t * 1e6, np.imag(st), linewidth=2)
ax[0, 1].grid(True)
ax[0, 1].set_xlabel('时间 (μs)')
ax[0, 1].set_ylabel('幅度')
ax[0, 1].set_title('LFM信号虚部')

# 图3: 匹配滤波输出幅度
ax[1, 0].plot(t * 1e6, np.abs(sout), linewidth=2)
ax[1, 0].grid(True)
ax[1, 0].set_xlabel('时间 (μs)')
ax[1, 0].set_ylabel('幅度')
ax[1, 0].set_title('匹配滤波输出幅度')

# 图4: 匹配滤波输出 (dB)
ax[1, 1].plot(t * 1e6, sout_dB, linewidth=2)
ax[1, 1].grid(True)
ax[1, 1].set_xlabel('时间 (μs)')
ax[1, 1].set_ylabel('幅度 (dB)')
ax[1, 1].set_title('匹配滤波输出 (dB)')
ax[1, 1].axvline(x=t[I]*1e6, color='r', linestyle='--', label='峰值位置')
ax[1, 1].axvline(x=t[j]*1e6, color='g', linestyle='--', label='第一零点')
ax[1, 1].legend()

plt.tight_layout()
plt.show()
plt.close()

# 单独绘制详细的匹配滤波结果图
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(t * 1e6, sout_dB, linewidth=2)
ax.grid(True)
ax.set_xlabel('时间 (μs)')
ax.set_ylabel('幅度 (dB)')
ax.set_title('LFM信号匹配滤波输出 - 详细视图')
ax.axvline(x=t[I]*1e6, color='r', linestyle='--', label='峰值位置')
ax.axvline(x=t[j]*1e6, color='g', linestyle='--', label='第一零点')
ax.legend()
plt.show()
plt.close()
