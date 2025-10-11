#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 11:10:30 2025

@author: jack
"""

import numpy as np
import scipy.linalg as la

def qq(N):
    """
    对应MATLAB中的qq函数，用于生成特定变换矩阵
    """
    k = N // 2  # fix(N/2) 的Python实现
    I = np.eye(k)
    II = np.fliplr(I)  # 左右翻转

    if N % 2 == 0:
        # 注意使用 1j 而不是 j
        p = np.block([[I, 1j*I],
                      [II, -1j*II]]) / np.sqrt(2)
    else:
        zeros_k1 = np.zeros((k, 1))
        zeros_1k = np.zeros((1, k))
        sqrt2 = np.sqrt(2)
        p = np.block([[I, zeros_k1, 1j*I],
                      [zeros_1k, sqrt2, zeros_1k],
                      [II, zeros_k1, -1j*II]]) / np.sqrt(2)
    return p

# ========== 参数设置 ==========
derad = np.pi / 180
radeg = 180 / np.pi
twpi = 2 * np.pi

kelm = 8       # 阵元数
dd = 0.5       # 阵元间距
# 阵元位置：注意Python的arange用法与MATLAB的:运算符略有不同
d = np.arange(-(kelm-1)/2 * dd, ((kelm-1)/2 * dd) + dd/2, dd)

iwave = 3      # 信源数
theta1 = np.array([10, 20, 30])  # 方位角
theta2 = np.array([20, 25, 15])  # 俯仰角
snr = 20       # 信噪比(dB)
n = 200        # 快拍数

# ========== 方向矩阵构建 ==========
# 注意：使用 .T 进行转置，MATLAB中的 ' 是共轭转置，这里不需要共轭
# 使用 @ 进行矩阵乘法，确保维度正确
A0 = np.exp(1j * twpi * d.reshape(-1, 1) @ (np.sin(theta1 * derad) * np.cos(theta2 * derad)).reshape(1, -1)) / np.sqrt(kelm)

A1 = np.exp(1j * twpi * d.reshape(-1, 1) @ (np.sin(theta1 * derad) * np.sin(theta2 * derad)).reshape(1, -1)) / np.sqrt(kelm)

# ========== 信号生成 ==========
S = np.random.randn(iwave, n)  # 随机信号
X0 = np.zeros((kelm * kelm, n), dtype=complex)  # 初始化接收数据矩阵

# 构建接收数据：注意Python索引从0开始
for im in range(kelm):
    start_row = im * kelm
    end_row = (im + 1) * kelm
    # 使用矩阵乘法替代MATLAB中的 A0*diag(A1(im,:))*S
    X0[start_row:end_row, :] = A0 @ np.diag(A1[im, :]) @ S

# 添加高斯白噪声
def awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

X = awgn(X0, snr)

# ========== 算法核心部分 ==========
L = iwave  # 信号子空间维度

# 构建选择矩阵 J1 和 J2
J1 = np.eye(kelm-1, kelm)  # 从kelm×kelm单位矩阵取前kelm-1行
J2 = np.flipud(np.fliplr(J1))  # 先左右翻转再上下翻转

Q = qq(kelm)  # 获取变换矩阵
# 注意：MATLAB中的 kron(Q', Q') 在Python中是 np.kron(Q.conj().T, Q.conj().T)
Y = np.kron(Q.conj().T, Q.conj().T) @ X

Q0 = qq(kelm-1)
# 计算实数域映射矩阵
K1 = np.real(Q0.conj().T @ J2 @ Q)
K2 = np.imag(Q0.conj().T @ J2 @ Q)

I = np.eye(kelm)
# 构建Kronecker积矩阵
Ku1 = np.kron(I, K1)
Ku2 = np.kron(I, K2)
Kv1 = np.kron(K1, I)
Kv2 = np.kron(K2, I)

# 构建实数域观测矩阵
E = np.hstack([np.real(Y), np.imag(Y)])
Ey = E @ E.conj().T / n

# 特征分解
D, V = np.linalg.eig(Ey)
# 对特征值排序（降序）
idx = np.argsort(D)[::-1]
EVAs = D[idx]
EVs = V[:, idx]

# 取前L个主要特征向量构成信号子空间
Es = EVs[:, :L]

# 矩阵束计算
fiu = np.linalg.pinv(Ku1 @ Es) @ Ku2 @ Es
fiv = np.linalg.pinv(Kv1 @ Es) @ Kv2 @ Es
F = fiu + 1j * fiv

# 特征分解获取DOA估计
DD, VV = np.linalg.eig(F)
EVA = DD

# 角度估计
u = 2 * np.arctan(np.real(EVA)) / np.pi
v = 2 * np.arctan(np.imag(EVA)) / np.pi

# 使用arctan2避免除零错误
theta10 = np.arcsin(np.sqrt(u**2 + v**2)) * radeg
theta20 = np.arctan2(v, u) * radeg  # 使用arctan2更安全

print("估计的方位角theta1:", theta10)
print("估计的俯仰角theta2:", theta20)





