#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 17:04:44 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)

# 清空图形
plt.close('all')

# 信号数据
M = 40
c = 3e8
N = 300
f = 4.5e6
lambda_val = c / f
SNR = [0, 30, 40]
R_circ = 175
thetaK = np.array([50, 20, 40]) / 180 * np.pi
phiK = np.array([120, 240, 140]) / 180 * np.pi
As = 10 ** (np.array(SNR) / 20)
pi = np.pi
# 构建阵列流型矩阵 A
A = np.zeros((M, len(thetaK)), dtype=complex)
for jj in range(len(thetaK)):
    A[:, jj] = np.exp(1j * 2*pi * R_circ /lambda_val * np.sin(thetaK[jj]) * np.cos(phiK[jj] - 2*pi*np.arange(M)/M))

signal = np.random.randn(len(thetaK), N)
for i in range(len(SNR)):
    signal[i, :] = signal[i, :] * As[i]

noise = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)
xn = A @ signal + noise

dis = 1
theta_scan = np.arange(0, 90, dis) / 180 * np.pi
phi_scan = np.arange(0, 360, dis) / 180 * np.pi



# # MVDR
R = xn @ xn.conj().T / N
expect_n = 0  # Python索引从0开始
noise_n = 1
noise_nn = 2
A_expect = A[:, expect_n]
w0 = np.linalg.inv(R) @ A_expect / (A_expect.conj().T @ np.linalg.inv(R) @ A_expect)

F = np.zeros((len(theta_scan), len(phi_scan)), dtype=complex)

for i in range(len(theta_scan)):
    for j in range(len(phi_scan)):
        a = np.exp(1j * 2 * np.pi / lambda_val * R_circ * np.sin(theta_scan[i]) * np.cos(phi_scan[j] - 2 * np.pi * np.arange(0, M) / M))
        F[i, j] = w0.conj().T @ a

F_abs = 20 * np.log10(np.abs(F))


##### LCMV
R = xn @ xn.conj().T / N

C_matrix = A
matrix_f = np.array([1, 0, 0]).reshape(-1, 1)  # 约束向量
F = np.zeros((len(theta_scan), len(phi_scan)), dtype=complex)

w0 = np.linalg.inv(R) @ C_matrix @ np.linalg.inv(C_matrix.conj().T @ np.linalg.inv(R) @ C_matrix) @ matrix_f

for i in range(len(theta_scan)):
    for j in range(len(phi_scan)):
        a = np.exp(1j * 2 * np.pi / lambda_val * R_circ * np.sin(theta_scan[i]) * np.cos(phi_scan[j] - 2 * np.pi * np.arange(0, M) / M))
        F[i, j] = (w0.conj().T @ a)[0]

F_abs = 20 * np.log10(np.abs(F))


####### # RAB
R = xn @ xn.conj().T / N
# MATLAB: [E_all, lambda_all] = eig(R) 返回特征向量矩阵E_all和特征值对角矩阵lambda_all
# Python: lambda_all, E_all = np.linalg.eig(R) 返回特征值数组lambda_all和特征向量矩阵E_all
eigenvalues, eigvector = np.linalg.eig(R)
idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大

eigvector = eigvector[:, idx][:,::-1]
Es = eigvector[:, :len(thetaK)]

C_matrix = A[:, 0].reshape(-1, 1)
C_matrix = Es @ Es.conj().T @ C_matrix
w0 = np.linalg.inv(R) @ C_matrix / (C_matrix.conj().T @ np.linalg.inv(R) @ C_matrix)

F = np.zeros((len(theta_scan), len(phi_scan)), dtype = complex)

for i in range(len(theta_scan)):
    for j in range(len(phi_scan)):
        a = np.exp(1j * 2 * np.pi / lambda_val * R_circ * np.sin(theta_scan[i]) * np.cos(phi_scan[j] - 2 * np.pi * np.arange(M) / M))
        F[i, j] = (w0.conj().T @ a)[0]

F_abs = 20 * np.log10(np.abs(F))


#### 画图
fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')
PHI, THETA = np.meshgrid(phi_scan * 180 / np.pi, theta_scan * 180 / np.pi)
surf = ax.plot_surface(PHI, THETA, F_abs, cmap='viridis')
fig1.colorbar(surf)
ax.set_xlabel('方位角 φ °')
ax.set_ylabel('俯仰角 θ ° ')
ax.set_zlabel('波束增益 (dB)')
ax.set_title(f'M={M}阵元 环阵CBF三维视图 来波f={f/1e6}MHz')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlim([0, 360])
ax.set_ylim([0, 90])
ax.set_zlim([np.min(F_abs) - 20, np.max(F_abs) + 20])
ax.set_xticks(np.arange(0, 361, 60))
ax.set_yticks(np.arange(0, 91, 15))
z_ticks = np.arange(np.round(np.min(F_abs) / 10) * 10 - 10, 21, 20)
ax.set_zticks(z_ticks)

expect_n = 0
noise_n = 1
noise_nn = 2

fig2 = plt.figure(2, figsize=(12, 12))

# 期望方向俯仰角切片
plt.subplot(3, 2, 1)
phi_idx_expect = np.argmin(np.abs(phi_scan - phiK[expect_n]))
plt.plot(theta_scan * 180 / np.pi, F_abs[:, phi_idx_expect])
plt.axvline(x=thetaK[expect_n] * 180 / np.pi, color='r', label=f'θ={thetaK[expect_n] * 180 / np.pi:.1f}°')
plt.title('期望方向俯仰角切片')
plt.xlim([0, 90])
plt.ylim([np.min(F_abs[:, phi_idx_expect]), np.max(F_abs[:, phi_idx_expect]) + 20])
plt.xticks(np.arange(0, 91, 15))
plt.legend()

# 期望方向方位角切片
plt.subplot(3, 2, 2)
theta_idx_expect = np.argmin(np.abs(theta_scan - thetaK[expect_n]))
plt.plot(phi_scan * 180 / np.pi, F_abs[theta_idx_expect, :])
plt.axvline(x=phiK[expect_n] * 180 / np.pi, color='r', label=f'φ={phiK[expect_n] * 180 / np.pi:.1f}°')
plt.title('期望方向方位角切片')
plt.xlim([0, 360])
plt.ylim([np.min(F_abs[theta_idx_expect, :]), np.max(F_abs[theta_idx_expect, :]) + 20])
plt.xticks(np.arange(0, 361, 60))
plt.legend()

# 干扰方向1俯仰角切片
plt.subplot(3, 2, 3)
phi_idx_noise = np.argmin(np.abs(phi_scan - phiK[noise_n]))
plt.plot(theta_scan * 180 / np.pi, F_abs[:, phi_idx_noise])
plt.axvline(x=thetaK[noise_n] * 180 / np.pi, color='r', label=f'θ={thetaK[noise_n] * 180 / np.pi:.1f}°')
plt.title('干扰方向1俯仰角切片')
plt.xlim([0, 90])
plt.ylim([np.min(F_abs[:, phi_idx_noise]), np.max(F_abs[:, phi_idx_noise]) + 20])
plt.xticks(np.arange(0, 91, 15))
plt.legend()

# 干扰方向1方位角切片
plt.subplot(3, 2, 4)
theta_idx_noise = np.argmin(np.abs(theta_scan - thetaK[noise_n]))
plt.plot(phi_scan * 180 / np.pi, F_abs[theta_idx_noise, :])
plt.axvline(x=phiK[noise_n] * 180 / np.pi, color='r', label=f'φ={phiK[noise_n] * 180 / np.pi:.1f}°')
plt.title('干扰方向1方位角切片')
plt.xlim([0, 360])
plt.ylim([np.min(F_abs[theta_idx_noise, :]), np.max(F_abs[theta_idx_noise, :]) + 20])
plt.xticks(np.arange(0, 361, 60))
plt.legend()

# 干扰方向2俯仰角切片
plt.subplot(3, 2, 5)
phi_idx_noise_nn = np.argmin(np.abs(phi_scan - phiK[noise_nn]))
plt.plot(theta_scan * 180 / np.pi, F_abs[:, phi_idx_noise_nn])
plt.axvline(x=thetaK[noise_nn] * 180 / np.pi, color='r', label=f'θ={thetaK[noise_nn] * 180 / np.pi:.1f}°')
plt.title('干扰方向2俯仰角切片')
plt.xlim([0, 90])
plt.ylim([np.min(F_abs[:, phi_idx_noise_nn]), np.max(F_abs[:, phi_idx_noise_nn]) + 20])
plt.xticks(np.arange(0, 91, 15))
plt.legend()

# 干扰方向2方位角切片
plt.subplot(3, 2, 6)
theta_idx_noise_nn = np.argmin(np.abs(theta_scan - thetaK[noise_nn]))
plt.plot(phi_scan * 180 / np.pi, F_abs[theta_idx_noise_nn, :])
plt.axvline(x=phiK[noise_nn] * 180 / np.pi, color='r', label=f'φ={phiK[noise_nn] * 180 / np.pi:.1f}°')
plt.title('干扰方向2方位角切片')
plt.xlim([0, 360])
plt.ylim([np.min(F_abs[theta_idx_noise_nn, :]), np.max(F_abs[theta_idx_noise_nn, :]) + 20])
plt.xticks(np.arange(0, 361, 60))
plt.legend()

plt.suptitle(f'M={M}阵元 环阵CBF二维切片 来波f={f/1e6}MHz', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.93)

plt.show()















