#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 21:16:47 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)
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
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(freq * 1e-6, spectrum, linewidth=2)
    ax.grid(True)
    ax.set_title(f'当采样率为{a[k]}倍时，线性调频信号的幅频特性')
    ax.set_xlabel('Frequency in MHz')
    ax.set_ylabel('幅度')
    plt.show()
    plt.close()
