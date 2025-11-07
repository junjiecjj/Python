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
# LFM信号匹配滤波和PSLR计算 - Python版本
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

# 寻找第一副瓣和计算PSLR
L = len(sout_dB)
maxdata = np.max(sout_dB)
I = np.argmax(sout_dB)

# 寻找第一副瓣的起始点
j = I
for i in range(I+1, min(I+L, L)):
    if sout_dB[i] > sout_dB[i-1]:
        j = i
        break

# 寻找第一副瓣的最大值
M = j
for k in range(j, min(j+L, L-1)):
    if sout_dB[k] < sout_dB[k-1]:
        M = k-1
        break

PSLR = sout_dB[M]

print(f"PSLR = {PSLR:.2f} dB")
print(f"主瓣峰值 = {maxdata:.2f} dB")
print(f"峰值副瓣比 = {maxdata - PSLR:.2f} dB")

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

# 图4: 匹配滤波输出 (dB) 并标注主瓣和第一副瓣
ax[1, 1].plot(t * 1e6, sout_dB, linewidth=2, label='匹配滤波输出')
ax[1, 1].grid(True)
ax[1, 1].set_xlabel('时间 (μs)')
ax[1, 1].set_ylabel('幅度 (dB)')
ax[1, 1].set_title('匹配滤波输出 (dB)')

# 标记主瓣峰值和第一副瓣
ax[1, 1].axvline(x=t[I]*1e6, color='r', linestyle='--', label='主瓣峰值')
ax[1, 1].axvline(x=t[M]*1e6, color='g', linestyle='--', label='第一副瓣峰值')

# 在图中添加PSLR信息
ax[1, 1].text(0.05, 0.05, f'PSLR = {PSLR:.2f} dB\n峰值副瓣比 = {maxdata - PSLR:.2f} dB', transform=ax[1, 1].transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax[1, 1].legend()

plt.tight_layout()
plt.show()
plt.close()

# 单独绘制详细的匹配滤波结果图，突出显示主瓣和副瓣
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(t * 1e6, sout_dB, linewidth=2)
ax.grid(True)
ax.set_xlabel('时间 (μs)')
ax.set_ylabel('幅度 (dB)')
ax.set_title('LFM信号匹配滤波输出 - 主瓣与副瓣分析')

# 标记关键点
ax.axvline(x=t[I]*1e6, color='r', linestyle='--', linewidth=2, label='主瓣峰值')
ax.axvline(x=t[M]*1e6, color='g', linestyle='--', linewidth=2, label='第一副瓣峰值')
ax.plot(t[I]*1e6, maxdata, 'ro', markersize=8, label=f'主瓣峰值: {maxdata:.2f} dB')
ax.plot(t[M]*1e6, PSLR, 'go', markersize=8, label=f'第一副瓣: {PSLR:.2f} dB')

ax.set_xlim([-0.3, 0.3])
# 添加文本标注
ax.text(0.05, 0.15, f'PSLR = {PSLR:.2f} dB\n峰值副瓣比 = {maxdata - PSLR:.2f} dB', transform=ax.transAxes, fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.legend(loc = 'lower right')
plt.show()
plt.close()












































