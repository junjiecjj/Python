#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 10:45:13 2025

@author: jack
"""

import numpy as np

def VV(b):
    m = np.arange(-4, 5)
    v = np.exp(1j * m * b)
    return v

def WW(a, n):
    w = (1/16) * np.exp(1j * 2 * np.pi * a * np.arange(n) / n)
    return w

# 主程序
N = 16
M = int(np.floor(N / 4))  # M=4

m = np.arange(-M, M + 1)  # [-4,-3,-2,-1,0,1,2,3,4]
m1 = np.arange(-1, -M - 1, -1)  # [-1,-2,-3,-4]

# 正确构建cv矩阵
cv_diag = np.concatenate([(1j)**m[:M+1], (1j)**m1[:M]])
cv = np.diag(cv_diag)

# 构建v矩阵
v = []
for in_idx in range(2 * M + 1):
    v.append(WW(in_idx - 4, N).conj())
v = 4 * np.array(v).T
# 计算fe
fe = v @ cv.T.conj()

# 构建W1矩阵
a1 = 2 * np.pi * m / (2 * M + 1)
W1 = []
for in_idx in range(2 * M + 1):
    W1.append(VV(a1[in_idx]))
W1 = (1/3) * np.array(W1)
# 计算Fr
Fr = fe @ W1

# 构造c0矩阵
x = np.array([1, -1, 1, -1, 1, 1, 1, 1, 1])  # 9个元素
c0 = np.diag(x)
x2 = c0 @ W1

# 构造信号
snap = 500  # 快拍数
fs = 1000   # 采样频率
t = np.arange(snap) / fs
M1 = 16     # 阵元数
N1 = 2      # 目标数
f = 30e3    # 频率
R = 1 / (4 * np.sin(np.pi / M1))
snr = 10
alpha = np.array([10, 20])
theta = np.array([15, 35])

# 构造阵列流型矩阵A11
a1_indices = np.arange(M1)
A11 = np.zeros((M1, N1), dtype=complex)
s = np.zeros((N1, snap), dtype=complex)

for ii in range(N1):
    np.random.seed(ii)  # 设置随机种子
    s[ii, :] = np.exp(1j * 2 * np.pi * (f * t + 0.5 * 5 * 2**ii * t**2))

for i in range(N1):
    A11[:, i] = np.exp(1j * 2 * np.pi * np.cos(2 * np.pi * a1_indices / M1 - theta[i] * np.pi / 180) * np.sin(alpha[i] * np.pi / 180))


# 生成接收信号
X0 = A11 @ s

# 添加高斯白噪声 - 修正信噪比计算
signal_power = np.mean(np.abs(X0)**2)
noise_power = signal_power / (10**(snr/10))
noise = np.sqrt(noise_power/2) * (np.random.randn(*X0.shape) + 1j * np.random.randn(*X0.shape))
Y = X0 + noise

# 计算协方差矩阵
RR = Y @ Y.T.conj() / snap
RRR = Fr.T.conj() @ RR @ Fr

# 特征值分解 - 使用实数部分
EVA, EV = np.linalg.eig(RRR.real)

# 对特征值排序
sorted_indices = np.argsort(EVA)[::-1]
EVA = EVA[sorted_indices]
EV = EV[:, sorted_indices]

# 提取信号子空间
E = EV[:, :N1]

# 构造C0矩阵
C0 = np.array([1, -1, 1, -1, 1, 1, 1, 1, 1])  # 9个元素
C00 = np.diag(C0)

# 计算Z1
Z1 = C00 @ W1 @ E

# 分割矩阵
Z11 = Z1[0:7, :]  # 前7行
Z12 = Z1[1:8, :]  # 第2-8行
Z13 = Z1[2:9, :]  # 第3-9行

# 构造E1矩阵
E1 = np.hstack([Z11, Z13])

# 构造T矩阵
T1 = np.array([-3, -2, -1, 0, 1, 2, 3])  # 7个元素
T = (1/np.pi) * np.diag(T1)

# 计算B矩阵
B = np.linalg.pinv(E1) @ T @ Z12
B1 = B[0:2, :]

# 特征值分解
w, p = np.linalg.eig(B1)

# 输出结果
elevation_angles = np.arcsin(np.abs(w)) * 180 / np.pi
azimuth_angles = np.angle(w) * 180 / np.pi

print("俯仰角 (度):", elevation_angles)
print("方位角 (度):", azimuth_angles)





































































