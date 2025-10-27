#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 20:25:06 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzk0NTQzNDQyMw==&mid=2247502308&idx=4&sn=8eb3ac77f9d8ae2dbfade0011eedb5e4&chksm=c29f97bf19222e341b5832810dacdd7319fe5ef8c8def5905cf43287d77aeafdd819bbc9d16b&mpshare=1&scene=1&srcid=1008SjlJA9Ns6D8TDm5hPxrd&sharer_shareinfo=c18b3490bcb876724f662116bc2573ba&sharer_shareinfo_first=c18b3490bcb876724f662116bc2573ba&exportkey=n_ChQIAhIQAUNkqDURM2ZhmS71nfbByBKfAgIE97dBBAEAAAAAACaSJZJh6cwAAAAOpnltbLcz9gKNyK89dVj05WPG4mHIk%2FbazCYhiyl4Pop9R25ap1haGYWITqEOXbNG7OJTkR95%2Fjzic3SxIV84UceXBT74i1SDL9GPtCGvXU06TW2GBSENtWTHALcv%2FbwCt7JylonDS2AroAJgSIUzCA9Dc4nu3Ik5P3hMVSjYOPn57gGPHIWa%2BaTsgBCqyItO%2BW3SPNQVFc%2BaS4T3KkRtRoEFC7ypCYbz9BmvecZMJvR3L9okima6Kw5N9pY1cbMBY1%2FvsLm9Nek%2BFXNxJcsG02lHbHBnyIuY%2Fs5iydGl2S59h9xyPWQCf3aPqdOnEs3%2FBAtfl7I8NgXAVXDUXrVFyblhyNDqSPJA&acctmode=0&pass_ticket=R5QG9DzJhwUKUF2GjUbU7GrXL5r93BVCnZocyPDI9S%2F9owv9ehWFZZLoHLMNn6Xy&wx_header=0#rd


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fft import fft, fftshift
from scipy.fft import next_fast_len
# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 16  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 16


def ambgfun1(x, fs, prf):
    """
        计算信号的模糊函数
        这是一个简化的实现，可能需要根据实际需求调整
    """
    N = len(x)
    # 时延范围
    max_delay = N // 2
    delays = np.arange(-max_delay, max_delay) / fs

    # 多普勒范围
    max_doppler = prf / 2
    dopplers = np.linspace(-max_doppler, max_doppler, 1024)

    # 初始化模糊函数矩阵
    afmag = np.zeros((len(dopplers), len(delays)))

    # 计算模糊函数
    for i, delay in enumerate(delays):
        delay_samples = int(delay * fs)
        for j, doppler in enumerate(dopplers):
            # 时延信号
            if delay_samples >= 0:
                x1 = x[:N-delay_samples]
                x2 = x[delay_samples:] * np.exp(1j * 2 * np.pi * doppler * np.arange(N-delay_samples) / fs)
            else:
                x1 = x[-delay_samples:]
                x2 = x[:N+delay_samples] * np.exp(1j * 2 * np.pi * doppler * np.arange(N+delay_samples) / fs)

            # 计算互相关
            afmag[j, i] = np.abs(np.sum(x1 * np.conj(x2)))

    # 归一化
    afmag = afmag / np.max(afmag)

    return afmag, delays, dopplers

def ambgfun(x, fs, prf = 1000):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    Nx = len(x)
    Ex = np.sum(np.abs(x) ** 2)

    # Auto-ambiguity: N = 2*Nx - 1 delays
    N_delay = 2 * Nx - 1

    # Number of Doppler frequencies: M = 2^ceil(log2(N))
    M_doppler = 2 ** int(np.ceil(np.log2(N_delay)))

    # Auto-ambiguity delay vector
    delay = np.arange(1-Nx, Nx) / fs

    # Create Doppler frequency vector
    doppler = np.linspace(-fs/2, fs/2 - fs/M_doppler, int(M_doppler))

    # Initialize ambiguity function matrix
    afmag = np.zeros((len(doppler), len(delay)), dtype=complex)

    # Compute ambiguity function
    for i, fd in enumerate(doppler):
        for j, tau in enumerate(delay):
            # Convert delay to samples
            tau_samples = int(round(tau * fs))

            # Auto-ambiguity function
            if 0 <= tau_samples < Nx:
                # Positive delay
                u1 = x[:Nx - tau_samples] * np.exp(1j * 2 * np.pi * fd *  np.arange(Nx - tau_samples) / fs)
                u2 = x[tau_samples:]
                afmag[i, j] = np.dot(u1, np.conj(u2))
            elif tau_samples < 0 and tau_samples > -Nx:
                # Negative delay
                tau_samples_abs = abs(tau_samples)
                u1 = x[tau_samples_abs:] * np.exp(1j * 2 * np.pi * fd *  np.arange(Nx - tau_samples_abs) / fs)
                u2 = x[:Nx - tau_samples_abs]

                afmag[i, j] = np.dot(u1, np.conj(u2))
    afmag = np.abs(afmag) / Ex
    return afmag, delay, doppler


# 主程序
# Hfm参数
fc = 12.5e3
fs = 5 * fc
ts = 1 / fs
N = 450
T = (1 / fs) * N

B = 6000
fmax = fc / (1 - B / (2 * fc))
fmin = fc / (1 + B / (2 * fc))
bw = fmax - fmin
t0 = fc * N * ts / B
K = N * ts * fmax * fmin / B
# W = B * N * ts
t = (np.arange(-(N-1)/2, (N-1)/2 + 1) * ts).reshape(-1, 1)
x0 = np.exp(-1j * (2 * np.pi * K * np.log(1 - t / t0)))
m = bw / T
v = 9
c = 1500
D = 1 + 2 * v / c
ft = fc / (1 - (m / fc) * t)
ftr = fc * D / (1 - (m / fc) * D * t)

# 图1: 频率随时间变化
fig, ax = plt.subplots( figsize = (8, 6))
ax.plot(t, ft, label='Transmitted')
ax.plot(t, ftr, label='Received')
ax.grid(True)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.legend()
plt.show()
plt.close()


# 接收信号
tr = (np.arange(-(N-1)/2, (N-1)/2 + 1) + 50) * ts
xr = np.exp(-1j * (2 * np.pi * K * np.log(1 - tr / t0)))

prf = 100
L = int(round((1 / prf) / ts))

# 创建发射信号
x = np.zeros(L, dtype=complex)
x[:N] = x0.flatten()

# 计算模糊函数
afmag, delay, doppler = ambgfun(x, fs, prf)
ambgu = afmag * (afmag > 0.5)

# 图2: 模糊函数3D图
X, Y = np.meshgrid(delay, doppler)

fig = plt.figure(figsize = (8, 8) , constrained_layout = True)
ax = fig.add_subplot(111, projection='3d')
cbar = ax.plot_surface(X*1e3, Y, afmag, rstride = 2, cstride = 2, cmap = plt.get_cmap('jet'))
ax.grid(False)
ax.set_proj_type('ortho')

# 3D坐标区的背景设置为白色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(X.min()*1e3, X.max()*1e3)
ax.set_ylim(Y.min(), Y.max())
ax.set_xlabel('Delay (ms)')
ax.set_ylabel('Doppler Shift (hertz)')

plt.show()
plt.close()

# 图3: 模糊函数等高线图
fig, ax = plt.subplots( figsize = (8, 6))
ax.contour(delay*1e3, doppler, ambgu, levels = 10, cmap = plt.get_cmap('jet'))
ax.set_xlim([-7, 7])
ax.set_ylim([-7600, 7600])
ax.set_xlabel('Delay (ms)')
ax.set_ylabel('Doppler Shift (hertz)')

ax.grid(True)
plt.show()
plt.close()

# 提取零多普勒和零时延剖面
# 注意：这里索引可能需要根据实际数组大小调整
afmag_T0 = afmag[len(doppler)//2, :]  # 零多普勒剖面
afmag_f0 = afmag[:, len(delay)//2]    # 零时延剖面

# 图4: 距离模糊函数
fig, ax = plt.subplots( figsize = (8, 6))
ax.plot(delay * 1000, afmag_T0)
ax.set_xlabel('Delay (ms)')
ax.set_ylabel('Amplitude')
ax.grid(True)
ax.set_xlim([-2, 2])
ax.set_ylim([0, 1])
# ax.axhline(y=0.7, color='r', linestyle='--', linewidth=1.2)
# ax.axvline(x=-0.44/bw * 1000, color='r', linestyle='--', linewidth=1.2)
plt.show()
plt.close()

# 图5: 速度模糊函数
fig, ax = plt.subplots( figsize = (8, 6))
ax.plot(doppler, afmag_f0)
ax.set_xlabel('Doppler Shift (hertz)')
ax.set_ylabel('Amplitude')
ax.set_xlim([-3/T, 3/T])
ax.set_ylim([0, 1])
ax.grid(True)
# ax.axhline(y=0.7, color='r', linestyle='--', linewidth=1.2)
# ax.axvline(x=-0.44/T, color='r', linestyle='--', linewidth=1.2)
plt.show()
plt.close()

# 图6: 信号时域和频域表示
tx = np.arange(L) * ts * 1000
fig, ax = plt.subplots(2,1, figsize = (8, 6))

# 时域图
ax[0].plot(t.flatten() * 1000, np.real(x0.flatten()))
ax[0].set_ylim([-1.1, 1.1])
ax[0].set_xlabel('Time (ms)')
ax[0].set_ylabel('Amplitude')
ax[0].grid(True)

# 频域图
nfft = int(2 ** np.ceil(np.log2(N)))
fre = np.arange(0, nfft/2 + 1) * fs / nfft / 1000
fftr = fft(x[:N], nfft)
ax[1].plot(fre, np.abs(fftr[:int(nfft/2) + 1]))
ax[1].set_xlabel('Frequency (kHz)')
ax[1].set_ylabel('Amplitude')
ax[1].grid(True)

plt.tight_layout()
plt.show()
plt.close()


