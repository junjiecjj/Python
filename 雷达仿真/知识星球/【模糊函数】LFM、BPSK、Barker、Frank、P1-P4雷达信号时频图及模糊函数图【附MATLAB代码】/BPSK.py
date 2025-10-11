#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 20:44:43 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import scipy
from scipy.fft import fft, fftshift
import scipy.signal as signal

# 二进制序列
s = np.array([1, 1, 0, 1, 0, 0, 1, 0])
f = 50e3  # 采样频率/载波信号频率
PRF = 1e3
N = len(s)

# 调制
t = np.linspace(0, 2*np.pi, 100)
cp1 = 2*s - 1  # 双极性非归零码

bit = []
cp = []
mod = []

for n in range(N):
    bit.extend([s[n]] * 100)
    cp.extend([cp1[n]] * 100)
    c = np.cos(f * t / (2*np.pi*50e3) * 2*np.pi)  # 调整频率比例
    mod.extend(c)

bit = np.array(bit)
cp = np.array(cp)
mod = np.array(mod)
bpsk_mod = cp * mod

# 时域图
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(bit, 'b', linewidth=1.5)
plt.grid(True)
plt.title('Binary Signal')
plt.axis([0, 100*len(s), -2, 2])

plt.subplot(2, 1, 2)
plt.plot(bpsk_mod, 'r', linewidth=1.5)
plt.grid(True)
plt.title('BPSK modulation')
plt.axis([0, 100*len(s), -2, 2])
plt.xlabel('1            1            0            1            0            0            1            0')

plt.tight_layout()

# 频域图
plt.figure(figsize=(10, 6))
nfft = 100 * N  # 做FFT的点数就是信号的点数
Z = fftshift(fft(bpsk_mod, nfft))  # 对信号做FFT
fr = np.arange(0, nfft/2) / nfft * f  # fft的频率范围
plt.plot(fr * 1e-3, np.abs(Z[int(nfft/2):]), '.-')  # 画fft出来的频谱图
plt.xlabel('频率（kHz）')
plt.ylabel('幅度')
plt.title('BPSK信号频域图')
plt.grid(True)

# 自相关图
z1 = np.correlate(bpsk_mod, bpsk_mod, mode='full')
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(z1)), z1)
plt.xlabel('采样点')
plt.ylabel('幅度')
plt.title('BPSK信号自相关域图')
plt.grid(True)

# 时频域图
plt.figure(figsize=(12, 8))
nsc = int(100 * N / 10)  # 做STFT的窗长
nov = int(nsc * 0.9)  # 两个窗长的覆盖部分的长度
nff = max(256, 2**int(np.ceil(np.log2(nsc))))  # 做STFT的点数

f_stft, t_stft, Sxx = spectrogram(bpsk_mod, fs=f, window=scipy.signal.windows.hamming(nsc), noverlap=nov, nfft=nff, return_onesided=False)
Sxx = fftshift(Sxx, axes=0)
f_stft = fftshift(f_stft)

plt.pcolormesh(t_stft * 1000, f_stft * 1e-3, 10 * np.log10(np.abs(Sxx)), shading='gouraud')
plt.colorbar(label='强度 (dB)')
plt.ylabel('频率 (kHz)')
plt.xlabel('时间 (ms)')
plt.title('BPSK信号时频域图')

# 模糊函数 - 简化实现
def ambgfun(x, fs, PRF):
    """
    简化的模糊函数计算
    """
    N = len(x)
    # 时延范围
    max_delay = int(N / 2)
    delays = np.arange(-max_delay, max_delay) / fs

    # 多普勒范围
    max_doppler = PRF / 2
    dopplers = np.linspace(-max_doppler, max_doppler, 256)

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

# 计算模糊函数
afmag, delay, doppler = ambgfun(bpsk_mod, f, PRF)

# 绘制模糊函数
plt.figure(figsize=(12, 8))
X, Y = np.meshgrid(delay * 1e6, doppler / 1e3)
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, afmag, cmap='viridis', linewidth=0, antialiased=True)
ax.set_xlabel('Delay τ (μs)')
ax.set_ylabel('Doppler f_d (kHz)')
ax.set_zlabel('幅度')
plt.title('BPSK信号模糊函数')
plt.colorbar(surf)

plt.tight_layout()
plt.show()
