#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 20:30:50 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzg3ODkwOTgyMw==&mid=2247488206&idx=3&sn=824dd28beffd2d81a2c018ec5ebb9528&chksm=cf0dd2def87a5bc8a4c85aef5689876c998610fdb4a51954dc2154a01c3a9731bd5acbe8fb3e&cur_album_id=3692626176607780876&scene=189#wechat_redirect

理论计算的模糊函数

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
利用FFT计算巴克码序列的波形、频谱及模糊函数
"""
Barker_code = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
T = 1e-6
N = len(Barker_code)
tau = N * T
samp_num = len(Barker_code) * 10
n = np.ceil(np.log(samp_num) / np.log(2))
nfft = int(2 ** n)

# 生成巴克码波形
u = np.zeros(nfft)
u[0:samp_num] = np.kron(Barker_code, np.ones(10))
delay = np.linspace(-tau, tau, nfft)

# 1. 巴克码波形图
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(delay * 1e6 + N, u)
ax.grid(True)
ax.set_xlim([0, 14])
ax.set_ylim([-1.5, 1.5])
ax.set_xlabel('t/us')
ax.set_ylabel('u(t)')
ax.set_title('巴克码波形图')
plt.show()
plt.close()

# 2. 巴克码频谱图
sampling_interval = tau / nfft
freqlimit = 0.5 / sampling_interval
f = np.linspace(-freqlimit, freqlimit, nfft)
freq = np.fft.fft(u, nfft)
vfft = freq.copy()
freq = np.abs(freq) / np.max(np.abs(freq))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(f * 1e-6, np.fft.fftshift(freq))
ax.grid(True)
ax.set_xlim([-6, 6])
ax.set_ylim([0, 1])
ax.set_xlabel('f/MHz')
ax.set_ylabel('|U(f)|')
ax.set_title('巴克码频谱图')
plt.show()
plt.close()

# 3. 计算模糊函数
freq_del = 12 / tau / 100
freq1 = np.arange(-6/tau, 6/tau + freq_del, freq_del)
amf = np.zeros((len(freq1), nfft))

for k in range(len(freq1)):
    sp = u * np.exp(1j * 2 * np.pi * freq1[k] * delay)
    ufft = np.fft.fft(sp, nfft)
    prod = ufft * np.conj(vfft)
    amf[k, :] = np.fft.fftshift(np.abs(np.fft.ifft(prod)))

amf = amf / np.max(amf)
m, n = np.where(amf == 1.0)
m, n = m[0], n[0]  # 取第一个最大值位置

# 3D模糊图
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(delay*1e6, freq1*1e-6)
cbar = ax.plot_surface(X, Y, amf, rstride=2, cstride=2, cmap=plt.get_cmap('jet'))
# plt.colorbar(cbar)
ax.set_xlabel('t/us')
ax.set_ylabel('fd/MHz')
ax.set_zlabel('幅值')
ax.set_title('巴克码信号的模糊图')
ax.grid(False)
ax.set_proj_type('ortho')
plt.show()
plt.close()

# 模糊度图
fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(delay*1e6, freq1*1e-6, amf, 1, colors='blue')
ax.grid(True)
ax.set_xlim([-1, 1])
ax.set_ylim([-0.1, 0.1])
ax.set_xlabel('t/us')
ax.set_ylabel('fd/MHz')
ax.set_title('巴克码序列的模糊度图')
plt.show()
plt.close()

# 距离模糊图
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(delay*1e6, amf[m, :], 'k')
ax.grid(True)
ax.set_xlim([-20, 20])
ax.set_ylim([0, 1])
ax.set_xlabel('t/us')
ax.set_ylabel('幅值')
ax.set_title('距离模糊图')
plt.show()
plt.close()

# 速度模糊图
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(freq1*1e-6, amf[:, n], 'k')
ax.grid(True)
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_xlabel('fd/MHz')
ax.set_ylabel('幅值')
ax.set_title('速度模糊图')
plt.show()
plt.close()
