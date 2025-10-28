#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 20:39:45 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# =============================================================================
# 全自由度空时自适应处理 - 修正改善因子计算版本
# =============================================================================

# 开始计时
start_time = time.time()

# 杂波仿真参数
N = 12                        # 阵元个数
M = 10                        # 相干脉冲数
CNR = 30                      # 杂噪比
beta = 1                      # 杂波折叠系数(beta = 2*v*T/d)
sita_a = np.arange(-90, 90.1, 0.9)  # 杂波单元个数
sita = sita_a * np.pi / 180
N_bin = len(sita)

# 目标参数
sita_t = -25                  # 目标DOA
omiga_t = 0.4                 # 目标Doppler
SNR = 0                       # 信噪比

# 空间导向矢量和时间导向矢量
# 空间频率和Dopple频率满足 omiga_d = beta * omiga_s
omiga_s = np.pi * np.sin(sita)
omiga_d = beta * omiga_s

aN = np.zeros((N, N_bin), dtype=complex)
bN = np.zeros((M, N_bin), dtype=complex)

# 计算空间导向矢量
for i in range(N):
    aN[i, :] = np.exp(-1j * i * omiga_s) / np.sqrt(N)

# 计算时间导向矢量
for i in range(M):
    bN[i, :] = np.exp(-1j * i * omiga_d) / np.sqrt(M)

# 目标空时信号
aN_t = np.exp(-1j * np.pi * np.arange(N).reshape(-1, 1) * np.sin(sita_t * np.pi / 180)) / np.sqrt(N)
bN_t = np.exp(-1j * np.pi * np.arange(M).reshape(-1, 1) * omiga_t) / np.sqrt(M)

S_t = np.kron(aN_t, bN_t).flatten()

# 计算杂波协方差矩阵
R = np.zeros((M * N, M * N), dtype=complex)
S_clutter = np.zeros((M * N, N_bin), dtype=complex)

# 服从正态分布的随机幅值，方差为1
np.random.seed(42)  # 设置随机种子以确保结果可重现
ksai = 10**(CNR / 10) * (np.random.randn(N_bin) + 1j * np.random.randn(N_bin)) / np.sqrt(2)

for ii in range(N_bin):
    S_clutter[:, ii] = np.kron(aN[:, ii], bN[:, ii])
    R = R + ksai[ii] * (S_clutter[:, ii].reshape(-1, 1) @ S_clutter[:, ii].reshape(1, -1).conj())

# 干扰协方差矩阵，杂噪比为30dB
R = R + np.eye(M * N)     # CNR = 30dB
inv_R = np.linalg.inv(R)                   # 逆矩阵

# 求特征值谱 - 使用特征值分解
eigenvalues, eigenvectors = np.linalg.eig(R)
eigenvalues_sorted = np.sort(np.real(eigenvalues))[::-1]  # 按降序排列

# 图1: 特征值谱 (2D图)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(10 * np.log10(eigenvalues_sorted), linewidth=2)
ax.grid(True)
ax.set_xlabel('特征值数目')
ax.set_ylabel('特征值(dB)')
ax.set_title('阵元数N=12, 相干脉冲数M=10')
ax.set_xlim([0, 120])
ax.set_ylim([-10, 50])
plt.show()
plt.close()

P_f = np.zeros((N_bin, N_bin), dtype=complex)
P_min_var = np.zeros((N_bin, N_bin), dtype=complex)

# 求杂波谱
for ii in range(N_bin):
    for jj in range(N_bin):
        SS = np.kron(aN[:, ii], bN[:, jj])
        P_f[ii, jj] = SS.conj().T @ R @ SS        # 傅氏谱
        P_min_var[ii, jj] = 1.0 / (SS.conj().T @ inv_R @ SS)

# 图2: 最小方差功率谱 (3D图)
X, Y = np.meshgrid(np.sin(sita), omiga_d/np.pi)
Z = 20 * np.log10(np.abs(P_min_var))

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap=plt.get_cmap('jet'))
ax.grid(False)
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_xlabel('方位余弦')
ax.set_ylabel('归一化Dopple频率')
ax.set_zlabel('功率(dB)')
ax.set_title('阵元数N=12, 相干脉冲数M=10')
plt.show()
plt.close()

# 空时最优权向量
w_opt = inv_R @ S_t / (S_t.conj().T @ inv_R @ S_t)

# 求最优空时响应
res_opt = np.zeros((N_bin, N_bin), dtype=complex)
for ii in range(N_bin):
    for jj in range(N_bin):
        SSS = np.kron(aN[:, ii], bN[:, jj])
        res_opt[ii, jj] = SSS.conj().T @ w_opt

# 图3: 最优空时响应 (3D图)
X, Y = np.meshgrid(omiga_d/np.pi, omiga_d/np.pi)
Z = 10 * np.log10(np.abs(res_opt)**2)

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap=plt.get_cmap('jet'))
ax.grid(False)
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_xlabel('归一化Dopple频率')
ax.set_ylabel('方位余弦')
ax.set_zlabel('功率(dB)')
ax.set_title('阵元数N=12, 相干脉冲数M=10')
plt.show()
plt.close()

# 求最优改善因子
# 修正改善因子计算 - 使用MATLAB注释中的公式
IF = np.zeros((N_bin, N_bin), dtype=complex)
for ii in range(N_bin):
    for jj in range(N_bin):
        SS = np.kron(aN[:, ii], bN[:, jj])
        IF[ii, jj] = (SS.conj().T @ inv_R @ SS) / (SS.conj().T @ SS)

# 图4: 改善因子 (2D图)
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(omiga_d/np.pi, 10 * np.log10(np.abs(IF[100, :])), linewidth=2)
ax.grid(True)
ax.set_xlabel('归一化Dopple频率')
ax.set_ylabel('改善因子(dB)')
ax.set_title('阵元数N=12, 相干脉冲数M=10')
# 修正纵轴范围，与MATLAB一致
ax.set_xlim([-1, 1])
ax.set_ylim([-35, 0])  # 修正为MATLAB源码的范围
plt.show()
plt.close()

end_time = time.time()
print(f"计算完成，耗时: {end_time - start_time:.2f} 秒")

# 打印改善因子的统计信息用于调试
print(f"改善因子范围: {10*np.log10(np.min(np.abs(IF))):.2f} dB 到 {10*np.log10(np.max(np.abs(IF))):.2f} dB")
print(f"改善因子均值: {10*np.log10(np.mean(np.abs(IF))):.2f} dB")
