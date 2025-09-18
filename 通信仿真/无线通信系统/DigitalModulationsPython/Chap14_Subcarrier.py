#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 17:42:34 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ======================== 绘制时域波形图=======================
Fs = 1000                                # 总的采样率
N = 1024                                 # 总的子载波数
T = N / Fs                               # 信号绘制为一个周期的长度
x = np.arange(0, T, 1/Fs)               # 生成时间向量，用于绘制波形
Numscr = 4                               # 绘制的子载波数量
s_data = 1                               # 初始相位
y = np.zeros((Numscr, len(x)), dtype=complex)  # 初始化存储每个子载波的复数值的矩阵
ini_phase = np.full(len(x), s_data)      # 生成与时间长度相匹配的初始相位向量

for k in range(Numscr):                  # 循环遍历要绘制的子载波数量
    for n in range(len(x)):              # 循环遍历时间序列
        y[k, n] = ini_phase[n] * np.exp(1j * 2 * np.pi * k * n / N)  # 计算每个时间点上每个子载波的复数值

# 绘制时域波形
plt.figure(1, figsize=(10, 6))
for k in range(Numscr):
    plt.plot(x, np.real(y[k, :]), label=f'子载波 {k+1}')

plt.xlabel('时间/s')                      # 设置 X 轴标签为"时间"
plt.ylabel('幅度/V')                      # 设置 Y 轴标签为"幅度"
plt.title('OFDM子载波时域波形')
plt.grid(True)
plt.legend()
plt.tight_layout()

# ======================== 绘制频域波形图=======================
a = 20
y1 = np.zeros((Numscr, a * N), dtype=complex)
y_combined = np.hstack((y, y1))          # 水平拼接两个矩阵

# 生成频率向量
f = np.linspace(-Fs/2, Fs/2, (a+1)*N, endpoint=False)
y_fft = np.zeros((Numscr, (a+1)*N))

for k in range(Numscr):
    # 计算FFT并进行频谱搬移
    y_fft[k, :] = np.real(np.fft.fftshift(np.fft.fft(y_combined[k, :]))) / N

# 绘制频域波形
plt.figure(2, figsize=(10, 6))
colors = ['blue', 'red', 'green', 'orange']
labels = ['子载波 1', '子载波 2', '子载波 3', '子载波 4']

for k in range(Numscr):
    plt.plot(f, y_fft[k, :], color=colors[k], label=labels[k], linewidth=1.5)

plt.grid(True)
plt.xlim([-10, 10])                      # 将 x 轴范围限制在 -10 到 10 之间
plt.xlabel('频率/Hz')
plt.ylabel('幅度/V')
plt.title('OFDM子载波频域波形')
plt.legend()
plt.tight_layout()

plt.show()

# 可选：更详细的频域分析版本
print("频域分析详细信息:")
print(f"采样率 Fs: {Fs} Hz")
print(f"子载波数量: {N}")
print(f"时间长度 T: {T} s")
print(f"时间序列长度: {len(x)}")
print(f"零填充后长度: {len(y_combined[0])}")
print(f"频率分辨率: {Fs/len(y_combined[0]):.4f} Hz")

# 显示每个子载波的中心频率
for k in range(Numscr):
    center_freq = k * Fs / N
    print(f"子载波 {k+1} 中心频率: {center_freq:.2f} Hz")
