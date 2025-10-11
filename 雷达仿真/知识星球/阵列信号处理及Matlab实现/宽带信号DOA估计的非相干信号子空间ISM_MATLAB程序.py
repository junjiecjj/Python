#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 10:31:25 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.linalg import eig

# 参数设置
M = 12                                       # 阵元数
N = 200                                      # 快拍数
ts = 0.01                                    # 时域采样间隔
f0 = 100                                     # 入射信号中心频率
f1 = 80                                      # 入射信号最低频率
f2 = 120                                     # 入射信号最高频率
c = 1500                                     # 声速
wavelength = c / f0                          # 波长
d = wavelength / 2                           # 阵元间距
SNR = 15                                     # 信噪比
b = np.pi / 180
theta1 = 30 * b                              # 入射信号波束角1
theta2 = 0 * b                               # 入射信号波束角2
n = np.arange(ts, N * ts + ts, ts)           # 时间序列
thetas = np.array([theta1, theta2]).reshape(-1, 1)

# 生成线性调频信号
s1 = chirp(n, f0=f1, f1=f2, t1=n[-1], method='linear')  # 生成线性调频信号1
s2 = chirp(n + 0.100, f0=f1, f1=f2, t1=n[-1], method='linear') # 生成线性调频信号2

# 进行FFT变换
sa = np.fft.fft(s1, 2048)
sb = np.fft.fft(s2, 2048)

# ISM算法
P = np.array([0, 1])  # 信号索引
sump = np.zeros(181)

for i in range(N):
    f = 80 + i * 1.0  # 当前频率
    s = np.array([sa[i], sb[i]]).reshape(-1, 1)

    # 构建导向矢量矩阵
    a = np.zeros((M, 2), dtype=complex)
    for m in range(M):
        phase_shift = -2j * np.pi * f * d / c * np.sin(thetas) * m
        a[m, :] = np.exp(phase_shift).flatten()

    # 计算协方差矩阵
    R = a @ (s @ s.conj().T) @ a.conj().T

    # 特征分解
    eigenvalues, eigenvectors = eig(R)

    # 获取特征值和特征向量（按特征值降序排列）
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 选择噪声子空间（去除最大的两个特征值对应的特征向量）
    noise_subspace = eigenvectors[:, 2:]

    k = 0
    p = np.zeros(181, dtype=complex)

    # 角度扫描
    for ii in range(-90, 91):
        alpha = np.sin(ii * b) * d / c
        tao = np.arange(M) * alpha
        A = np.exp(-2j * np.pi * f * tao).reshape(-1, 1)

        # 计算空间谱
        p[k] = A.conj().T @ noise_subspace @ noise_subspace.conj().T @ A
        k += 1

    sump += np.abs(p)

# 平均处理并计算最终的空间谱
pmusic = sump / N
pm = 1 / pmusic

# 绘制结果
theta_esti = np.arange(-90, 91)
plt.figure(figsize=(10, 6))
plt.plot(theta_esti, 20 * np.log10(np.abs(pm)))
plt.xlabel('入射角/度')
plt.ylabel('空间谱/dB')
plt.title('宽带信号DOA估计 - ISM算法')
plt.grid(True)
plt.show()
