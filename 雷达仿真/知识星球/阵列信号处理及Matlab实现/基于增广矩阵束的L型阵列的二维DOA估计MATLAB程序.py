#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 10:15:51 2025

@author: jack
"""


import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.linalg

def R_hankel(m, Rxy, N, Q):
    """Hankel矩阵生成函数"""
    R1 = []
    R2 = []
    # 注意：Python索引从0开始，MATLAB从1开始
    for mm in range(Q):
        R1.append(Rxy[m, mm])
    for i in range(N - Q + 1):
        R2.append(Rxy[m, i + Q - 1])
    R = scipy.linalg.hankel(R1, R2)
    return R

# ========== 参数设置 (与MATLAB完全一致) ==========
derad = np.pi / 180  # 角度转弧度
radeg = 180 / np.pi  # 弧度转角度
twpi = 2 * np.pi

kelmx = 8   # x阵元数
kelmy = 10  # y阵元数
dd = 0.5    # 阵元间距
iwave = 3   # 信源数
n = 200     # 快拍数
snr = 20    # 信噪比

# 真实角度值
theta1 = np.array([10, 30, 50])  # 方位角
theta2 = np.array([15, 35, 55])  # 俯仰角

# ========== 【关键修正1：导向矢量构建】 ==========
# 注意：dx, dy的生成方式，确保长度与kelmx, kelmy一致
dx = np.arange(0, kelmx) * dd  # 等效于MATLAB的 [0:dd:(kelmx-1)*dd]
dy = np.arange(0, kelmy) * dd
iwave = 3
L = iwave
# 将角度转换为弧度
theta1_rad = theta1 * derad
theta2_rad = theta2 * derad

# 【关键】使用矩阵乘法运算符 @ 和正确的维度变换
Ax = np.exp(-1j * twpi * dx.reshape(-1, 1) @ (np.sin(theta1_rad) * np.cos(theta2_rad)).reshape(1, -1))
Ay = np.exp(-1j * twpi * dy.reshape(-1, 1) @ (np.sin(theta1_rad) * np.sin(theta2_rad)).reshape(1, -1))

# 生成信号
S = np.random.randn(iwave, n)
X0 = Ax @ S
# 添加高斯白噪声
X_power = np.mean(np.abs(X0)**2)
X_noise = np.sqrt(X_power / (10**(snr/10))) * (np.random.randn(*X0.shape) + 1j * np.random.randn(*X0.shape)) / np.sqrt(2)
X = X0 + X_noise

Y0 = Ay @ S
# 添加高斯白噪声
Y_power = np.mean(np.abs(Y0)**2)
Y_noise = np.sqrt(Y_power / (10**(snr/10))) * (np.random.randn(*Y0.shape) + 1j * np.random.randn(*Y0.shape)) / np.sqrt(2)
Y = Y0 + Y_noise

# 计算互相关矩阵
Rxy = X @ Y.conj().T

P = 5
Q = 6
Re = []

for kk in range(kelmx - P + 1):
    Rx = []
    for k in range(P):
        # 注意：MATLAB索引从1开始，Python从0开始
        Rx.append(R_hankel(k + kk, Rxy, kelmy, Q))
    Rx = np.vstack(Rx)
    Re.append(Rx)

Re = np.hstack(Re)

# 奇异值分解
Ue, Se, Ve = np.linalg.svd(Re, full_matrices=False)
Uesx = Ue[:, :L]

Uesx1 = Uesx[: (P - 1) * Q, :]
Uesx2 = Uesx[Q: P * Q, :]

Fx = np.linalg.pinv(Uesx1) @ Uesx2
Dx, EVx = np.linalg.eig(Fx)
EVAx = Dx

# 重构Uesy
Uesy = np.zeros((P * Q, L), dtype=complex)
for im in range(Q):
    indices = np.arange(im, im + Q * (P - 1) + 1, Q)
    Uesy[im * P: (im + 1) * P, :] = Uesx[indices, :]

Uesy1 = Uesy[: (Q - 1) * P, :]
Uesy2 = Uesy[P: P * Q, :]

Fy = np.linalg.pinv(Uesy1) @ Uesy2
Dy, EVy = np.linalg.eig(Fy)
EVAy = Dy

# 组合F矩阵
F = 0.5 * Fx + 0.5 * Fy
D, EV = np.linalg.eig(F)

# 计算配对
P1 = np.linalg.inv(EV) @ EVx  # 修正：使用矩阵求逆
P2 = np.linalg.inv(EV) @ EVy  # 修正：使用矩阵求逆

P1 = np.abs(P1)
P2 = np.abs(P2)
P11 = P1.T
P21 = P2.T

# 找到最大值索引
Px = np.argmax(P11, axis=1)
Py = np.argmax(P21, axis=1)

EVAx = EVAx[Px]
EVAy = EVAy[Py]

# 计算角度估计
theta10 = np.arcsin(np.sqrt((np.angle(EVAx) / np.pi)**2 + (np.angle(EVAy) / np.pi)**2)) * radeg
theta20 = -np.arctan(np.angle(EVAy) / np.angle(EVAx)) * radeg

print("估计的theta1:", theta10)
print("估计的theta2:", theta20)













