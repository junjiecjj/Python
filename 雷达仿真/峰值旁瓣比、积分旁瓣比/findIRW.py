#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 21:11:23 2025

@author: jack
"""


import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# LFM信号匹配滤波和IRW计算 - Python版本
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

# 计算IRW（脉冲响应宽度）
maxdata = np.max(sout_dB)
x1 = np.where(sout_dB >= maxdata - 3)[0]  # 找到所有大于等于峰值-3dB的点
I = len(x1)
IRW = I * (1 / fs)

print(f"IRW = {IRW*1e6:.2f} μs")

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

# 图4: 匹配滤波输出 (dB) 并标注-3dB点
ax[1, 1].plot(t * 1e6, sout_dB, linewidth=2, label='匹配滤波输出')
ax[1, 1].grid(True)
ax[1, 1].set_xlabel('时间 (μs)')
ax[1, 1].set_ylabel('幅度 (dB)')
ax[1, 1].set_title('匹配滤波输出 (dB)')

# 标记-3dB点
if len(x1) > 0:
    ax[1, 1].axhline(y=maxdata-3, color='r', linestyle='--', label='-3dB线')
    ax[1, 1].axvline(x=t[x1[0]]*1e6, color='g', linestyle='--', alpha=0.7, label='-3dB起始点')
    ax[1, 1].axvline(x=t[x1[-1]]*1e6, color='g', linestyle='--', alpha=0.7, label='-3dB结束点')

    # 在图中添加IRW信息
    ax[1, 1].text(0.05, 0.05, f'IRW = {IRW*1e6:.2f} μs', transform=ax[1, 1].transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax[1, 1].legend()

plt.tight_layout()
plt.show()
plt.close()











