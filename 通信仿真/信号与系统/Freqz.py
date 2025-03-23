#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 01:49:04 2025

@author: jack
"""

# https://www.oryoy.com/news/shi-yong-python-shi-xian-pin-yu-fen-xi-shen-ru-li-jie-freqz-han-shu-zai-xin-hao-chu-li-zhong-de-ying.html
import numpy as np
import scipy
import matplotlib.pyplot as plt

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 20        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [4, 3] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22


#%%
# 定义分子和分母系数
b = [1]
a1 = [1, -0.9]
a2 = [1, 0.9]

# 生成频率点
w = np.linspace(0, np.pi, 512)

# 计算频率响应
_, h1 = scipy.signal.freqz(b, a1, w)
_, h2 = scipy.signal.freqz(b, a2, w)

# 绘制频率响应图
f, axs = plt.subplots(1, 1, figsize=(6, 4), )
axs.plot(w / np.pi, abs(h1),  label='alpha = -0.9')
axs.plot(w / np.pi, abs(h2), ':', label='alpha = 0.9')
axs.legend()
axs.set_title('Frequency Response')
axs.set_xlabel('Normalized Frequency')
axs.set_ylabel('Magnitude')
plt.show()
plt.close()


# 在这个例子中，我们定义了两个不同的系统，分别具有不同的分母系数a1和a2。通过freqz函数计算它们的频率响应，并使用matplotlib库绘制出频率响应的幅度图。
# 深入理解freqz的应用
# 1. 系统设计与分析:在系统设计和分析中，freqz可以帮助我们评估不同滤波器的设计效果。例如，设计一个低通滤波器，并使用freqz分析其频率响应。
# 设计一个低通滤波器
b, a = scipy.signal.butter(4, 0.2, 'low')

# 计算频率响应
w, h = scipy.signal.freqz(b, a, worN = 512)

# 绘制频率响应图
f, axs = plt.subplots(1, 1, figsize=(6, 4), )
axs.plot(w / np.pi, abs(h))
axs.set_title('Lowpass Filter Frequency Response')
axs.set_xlabel('Normalized Frequency')
axs.set_ylabel('Magnitude')
plt.grid()
plt.show()
plt.close()

# 2. 信号处理中的应用:在信号处理中，freqz可以用于分析信号经过系统后的频率变化。例如，对一个含有噪声的信号进行滤波，并分析滤波前后的频率响应。
# 生成一个含有噪声的信号
fs = 1000  # 采样频率
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2 * np.pi * 50 * t) + 2 * np.random.randn(fs)

# 设计一个带通滤波器
b, a = scipy.signal.butter(4, [0.1, 0.2], 'band')

# 计算滤波前后的频率响应
w, h = scipy.signal.freqz(b, a, worN = 512)
filtered_signal = scipy.signal.filtfilt(b, a, signal)

# 绘制原始信号和滤波后信号的频谱
f, axs = plt.subplots(2, 1, figsize = (12, 6))
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
axs[0].plot(np.fft.fftfreq(fs, 1/fs), np.abs(np.fft.fft(signal)))
axs[0].set_title('Original Signal Spectrum')
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Magnitude')

# plt.subplot(1, 2, 2)
axs[1].plot(np.fft.fftfreq(fs, 1/fs), np.abs(np.fft.fft(filtered_signal)))
axs[1].set_title('Filtered Signal Spectrum')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Magnitude')

plt.tight_layout()
plt.show()
plt.close()

# 在这个例子中，我们首先生成一个含有噪声的信号，然后设计一个带通滤波器，并使用freqz分析其频率响应。最后，通过FFT分析滤波前后信号的频谱变化。
# 结论
# freqz函数是Python信号处理中一个非常重要的工具，它可以帮助我们深入理解离散系统的频率响应特性。通过本文的介绍和示例，相信读者已经掌握了freqz函数的基本用法及其在系统设计和信号处理中的应用。
# 在实际项目中，灵活运用freqz函数，结合其他信号处理技术，可以有效地解决各种复杂的信号处理问题。







































