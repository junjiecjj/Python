#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:07:16 2025

@author: jack
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300      # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22

#%%  https://blog.csdn.net/qq_43485394/article/details/122655901
### 频域实现脉冲压缩
def rectpuls(t, remove, T):
    Ts = t[1] - t[0]
    fs = 1/Ts

    rect = (t >= -T/2) * (t <= T/2)
    # res = np.zeros(rect.size)
    K = int(remove*fs)
    rect = np.roll(rect, K) # 循环左移

    # t = t + remove
    return rect

### parameters
c = 3e8          # 光速
f0 = 10e9        # 载波
Tp = 10e-6       # 脉冲持续时间
B = 10e6         # 带宽
k = B/Tp         # 调频斜率
fs = 100e6       # 采样频率
R0 = 3000         # 目标距离

# signal generation
N = 1024*4       # 采样点
# n = np.arange(N)
Ts = 1/fs        # 采样间隔
t = np.arange(N)*Ts
f = np.arange(-N/2, N/2) * fs/N
tau_0 = 2*R0/c   # 时延

st = rectpuls(t, Tp/2, Tp) * np.exp(1j * np.pi * k * (t-Tp/2)**2)    #  参考信号
#  回波信号
# secho = rectpuls(t, tau_0+Tp/2, Tp) * np.exp(1j * np.pi * k * (t - tau_0 - Tp/2)**2) * np.exp(-1j * 2 * np.pi * f0 * tau_0)
secho = rectpuls(t, tau_0+Tp/2, Tp) * np.exp(1j * np.pi * (k * (t - tau_0 - Tp/2)**2 + 2 * f0 * tau_0))
#=============== 频域实现脉冲压缩 ================
Xs = scipy.fft.fft(st, N);        # 本地副本的FFT
Xecho = scipy.fft.fft(secho, N);  # 输入信号的FFT
Y = np.conjugate(Xs)*Xecho;       # 乘法器
Y = scipy.fft.fftshift(Y);
y = scipy.fft.ifft(Y, N);          # IFFT

r = t*c/2;
y = np.abs(y)/max(np.abs(y)) + 1e-10;
R0_est = r[np.argmax(y)]

##### plot
fig, axs = plt.subplots(2, 3, figsize = (18, 8), constrained_layout = True)

axs[0,0].plot(t * 1e6, np.real(st), color = 'b', label = '')
axs[0,0].set_xlabel('时间/us',)
axs[0,0].set_ylabel('幅值',)
axs[0,0].set_title("发送信号")

axs[0,1].plot(f/(1e6), np.abs(scipy.fft.fftshift(Xs)), color = 'b', label = '')
axs[0,1].set_xlabel('Frequency/MHz',)
axs[0,1].set_ylabel('幅值',)
axs[0,1].set_title("发送信号频谱" )

axs[0,2].plot(f/(1e6), np.abs(Y), color = 'b', label = '')
axs[0,2].set_xlabel('Frequency/MHz',)
axs[0,2].set_ylabel('幅值',)
axs[0,2].set_title("脉冲压缩结果的频谱" )

axs[1,0].plot(t * 1e6, np.real(secho), color = 'b', label = '')
axs[1,0].set_xlabel('时间/us',)
axs[1,0].set_ylabel('幅值',)
axs[1,0].set_title("回波" )


axs[1,1].plot(f/(1e6), np.abs(scipy.fft.fftshift(Xecho)), color = 'b', label = ' ')
axs[1,1].set_xlabel('Frequency/MHz',)
axs[1,1].set_ylabel('幅值',)
axs[1,1].set_title("回波频谱" )

axs[1,2].plot(r, 20*np.log10(y), color = 'b', label = '')
axs[1,2].set_xlabel('Range/m',)
axs[1,2].set_ylabel('幅值',)
axs[1,2].set_title("脉冲压缩结果" )

print(f"R0 = {R0}, R0_est = {R0_est}")
plt.show()
plt.close()

#=============== 时域实现脉冲压缩 ================
matched_filter = np.conj(st[::-1])  # 发射信号的共轭反转
compressed_signal = np.convolve(secho, matched_filter, mode = 'same')
# 结果归一化
compressed_signal = compressed_signal / np.max(np.abs(compressed_signal) + 1e-10 )
# 结果可视化
fig, axs = plt.subplots(4, 1, figsize = (6, 16), constrained_layout = True)

# 发射信号（实部）
axs[0].plot(t*1e6, np.real(st))
axs[0].set_title("Transmitted LFM Signal (Real Part)")
axs[0].set_xlabel("Time (μs)")
axs[0].set_ylabel("Amplitude")

# 接收回波信号（实部）
axs[1].plot(t*1e6, np.real(secho))
axs[1].set_title("Received Echo Signal (Real Part)")
axs[1].set_xlabel("Time (μs)")
axs[1].set_ylabel("Amplitude")

# matched_filter 信号（实部）
t_match = np.arange(len(matched_filter)) / fs
axs[2].plot(t_match * 1e6, np.real(matched_filter))
axs[2].set_title("matched_filter (Real Part)")
axs[2].set_xlabel("Time (μs)")
axs[2].set_ylabel("Amplitude")

# 脉冲压缩结果（幅度）
tmp = int((min(len(secho), len(matched_filter)) - 1) /2)
t_compressed = (np.arange(len(compressed_signal)) - tmp) / fs
r = t_compressed * c/2;
R0_est1 = r[np.argmax(np.abs(compressed_signal))]
print(f"R0 = {R0}, R0_est1 = {R0_est1}")

axs[3].plot(r, 20 * np.log10(np.abs(compressed_signal) + 1e-10))
axs[3].set_title("Range/m")
axs[3].set_xlabel("Range (m)")
axs[3].set_ylabel("Amplitude (dB)")

plt.show()
plt.close()

















