#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 21:05:00 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# =============================================================================
# 空时降维3DT方法 - 严格翻译版本
# =============================================================================

# 清屏和初始化
print("空时降维3DT方法")

try:
    # 尝试加载clutter_matrix.mat文件
    mat_data = loadmat('clutter_matrix.mat')
    clutter_matrix = mat_data['clutter_matrix']
    print("成功加载clutter_matrix.mat")
except:
    # 如果文件不存在，创建模拟数据
    print("clutter_matrix.mat文件不存在，创建模拟数据")
    NK = 160  # N*K = 16*10
    L = 1000
    clutter_matrix = np.random.randn(NK, L) + 1j * np.random.randn(NK, L)

# 获取数据维度
NK, L = clutter_matrix.shape
N = 16
K = 10
CNR = 60

# 计算协方差矩阵
Rc = clutter_matrix @ clutter_matrix.conj().T / L
noise_power = np.max(np.abs(Rc)) / (10**(CNR/10))
noise = noise_power * np.eye(N * K)
Rx = Rc + noise
anoise = np.max(np.abs(noise))  # 噪声功率

# 空域降维矩阵
Qs = np.eye(N)
psi0 = np.pi / 2

# 全维STAP
fd = np.arange(-1, 1 + 1/50, 1/50)
inv_Rx = np.linalg.inv(Rx)
IF = np.zeros(len(fd), dtype=complex)

for i in range(len(fd)):
    # 目标方向确定时，Ss固定，但目标doppler频率未知，故每一个fd有一个最优权矢量wopt
    Ss = np.exp(1j * np.pi * np.arange(N).reshape(-1, 1) * np.cos(psi0))
    St = np.exp(1j * np.pi * np.arange(K).reshape(-1, 1) * fd[i])
    S = np.kron(St, Ss)
    wopt = inv_Rx @ S / (S.conj().T @ inv_Rx @ S)
    IF[i] = (np.abs(wopt.conj().T @ S)**2 * (10**(CNR/10) + 1) * anoise /
             (wopt.conj().T @ Rx @ wopt))

# 绘制全维STAP结果
plt.figure(figsize=(10, 6))
plt.plot(fd, 10 * np.log10(np.abs(IF)), linewidth=2)
plt.xlabel('2f_d/f_r')
plt.ylabel('IF/dB')
plt.grid(True)
plt.title('全维STAP与3DT降维方法比较')

# 3DT降维方法
index = 0
IF_3dt = np.zeros(len(fd), dtype=complex)
fd_range = np.arange(-1, 1 + 1/50, 1/50)

for fd_val in fd_range:
    # 时域降维矩阵
    Qt = np.column_stack([
        np.exp(1j * np.pi * np.arange(K).reshape(-1, 1) * (fd_val - 1/K)),
        np.exp(1j * np.pi * np.arange(K).reshape(-1, 1) * fd_val),
        np.exp(1j * np.pi * np.arange(K).reshape(-1, 1) * (fd_val + 1/K))
    ])

    Q = np.kron(Qt, Qs)
    Ry = Q.conj().T @ Rx @ Q
    inv_Ry = np.linalg.inv(Ry)

    Ss = np.exp(1j * np.pi * np.arange(N).reshape(-1, 1) * np.cos(psi0))
    St = np.exp(1j * np.pi * np.arange(K).reshape(-1, 1) * fd_val)

    # 计算增益因子
    g1 = np.exp(1j * np.pi * np.arange(K).reshape(-1, 1) * (fd_val - 1/K)).conj().T @ St / (np.exp(1j * np.pi * np.arange(K).reshape(-1, 1) * fd_val).conj().T @ St)

    g2 = np.exp(1j * np.pi * np.arange(K).reshape(-1, 1) * (fd_val + 1/K)).conj().T @ St / (np.exp(1j * np.pi * np.arange(K).reshape(-1, 1) * fd_val).conj().T @ St)

    # 构建降维导向矢量
    Sy = np.vstack([g1 * Ss, Ss, g2 * Ss])

    # 计算3DT权向量
    W_3dt = inv_Ry @ Sy / (Sy.conj().T @ inv_Ry @ Sy)

    # 计算3DT改善因子
    IF_3dt[index] = (np.abs(W_3dt.conj().T @ Sy)**2 * (10**(CNR/10) + 1) * anoise /
                     (W_3dt.conj().T @ Ry @ W_3dt))

    index += 1

# 绘制3DT结果
plt.plot(fd_range, 10 * np.log10(np.abs(IF_3dt)), 'r', linewidth=2)
plt.legend(['最优', '3DT'])
plt.show()

print("计算完成")
