#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 17:20:12 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzk5MDU0NzkwNw==&mid=2247484037&idx=1&sn=5f93f7eb7e129dc9691e2f49bd1b1057&chksm=c4b17f2e00031f2e0f6d5673e8599227e1dccc15fcb94fd7616d4b5650fb5770b25317dd1970&mpshare=1&scene=1&srcid=0722azbNRXBBkfTLWi6VIEro&sharer_shareinfo=aea65b0f1347803ade1c36b892860373&sharer_shareinfo_first=aea65b0f1347803ade1c36b892860373&exportkey=n_ChQIAhIQ361krEeghmIqxJCFXD1HRhKfAgIE97dBBAEAAAAAAGEfBOVPjxMAAAAOpnltbLcz9gKNyK89dVj0c8KtdB2pnZwhGVPrVHvkeytuxBoI6Vc6Nf%2FG%2FToSj5p76aP2G8XGZnY%2B1zO11qJXRX7UbWxObDRrG%2BNCVYhu74XmvCAgsxVnJC0ppwd%2FpQq5sepcCsAxd%2BEag5FC9CrUNqzEynZLvyl8xMjkoOb9b0GacMQY04%2B5irHed64eCpO4ylidkRJ%2F%2Bgh%2FHTb09I1wvmbVVHkLzp1Xg6j%2BvriTP%2BCzNfPgJQHXnR50FiekHuRHkgE1faMZVnq2UJh7E%2F3dLpxM5Yc3Tah9I4DSDKfgToopEYfL6P9f9iCQ20UHdhgMl9gVUvSxHqP22QjWGLb4Cm8WU%2BDflmJm&acctmode=0&pass_ticket=plRn8aPLvJtt%2BA0G7dQxXziufeRX5Xej4t59RjTBfUcg364MtrJx9rkjkO3mOZoI&wx_header=0#rd


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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


#%% 参数设置
M = 40
c = 3e8
N = 300
f = 4.5e6
lambda_val = c / f
SNR = np.array([0, 30, 40])
R_circ = 175
thetaK = np.array([50, 20, 40]) / 180 * np.pi
phiK = np.array([120, 240, 140]) / 180 * np.pi
As = 10**(SNR / 20)

# 构建阵列流形矩阵A
A = np.zeros((M, len(thetaK)), dtype=complex)
for k in range(len(thetaK)):
    phase = 2 * np.pi / lambda_val * R_circ * np.sin(thetaK[k]) * np.cos(phiK[k] - 2 * np.pi * np.arange(M) / M)
    A[:, k] = np.exp(1j * phase)

# 生成信号和噪声
signal = np.random.randn(len(thetaK), N)
for i in range(len(SNR)):
    signal[i, :] = signal[i, :] * As[i]

noise = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)
xn = A @ signal + noise

# 计算协方差矩阵
R = xn @ xn.conj().T / N

# 扫描设置
dis = 1  # 扫描间隔
theta_scan = np.deg2rad(np.arange(0, 90, dis))
phi_scan = np.deg2rad(np.arange(0, 360, dis))

# 约束最小方差波束形成
C_matrix = A
matrix_f = np.array([[1, 0, 0]]).T  # 约束向量

# 计算权重向量
R_inv = np.linalg.inv(R)
w0 = R_inv @ C_matrix @ np.linalg.inv(C_matrix.conj().T @ R_inv @ C_matrix) @ matrix_f

# 扫描波束形成
F = np.zeros((len(theta_scan), len(phi_scan)), dtype=complex)

for i in range(len(theta_scan)):
    for j in range(len(phi_scan)):
        phase = 2 * np.pi / lambda_val * R_circ * np.sin(theta_scan[i]) * np.cos(phi_scan[j] - 2 * np.pi * np.arange(M) / M)
        a = np.exp(1j * phase)
        F[i, j] = w0.conj().T @ a

F_abs = 20 * np.log10(np.abs(F))

#%%  画图 - 3D视图
fig1 = plt.figure(figsize=(12, 9))
ax1 = fig1.add_subplot(111, projection='3d')

# 创建网格
PHI, THETA = np.meshgrid(np.rad2deg(phi_scan), np.rad2deg(theta_scan))

# 绘制3D曲面
surf = ax1.plot_surface(PHI, THETA, F_abs, cmap=cm.viridis, alpha=0.8)

# 设置标签
ax1.set_xlabel('方位角 φ (°)')
ax1.set_ylabel('俯仰角 θ (°)')
ax1.set_zlabel('波束增益 (dB)')
ax1.set_title(f'M={M}阵元 环阵CBF三维视图 来波f={f/1e6}MHz')

# 添加颜色条
fig1.colorbar(surf, ax=ax1, shrink=0.5, aspect=20)

# 设置坐标轴范围
ax1.set_xlim(0, 360)
ax1.set_ylim(0, 90)
ax1.set_zlim(np.min(F_abs) - 20, np.max(F_abs) + 20)

# 设置刻度
ax1.set_xticks(np.arange(0, 361, 60))
ax1.set_yticks(np.arange(0, 91, 15))
ax1.set_zticks(np.arange(np.round(np.min(F_abs)/10)*10-10, 21, 20))

plt.tight_layout()

# 画图 - 二维切片
fig2, axes = plt.subplots(3, 2, figsize=(15, 12))
expect_n = 0  # 期望信号索引
noise_n = 1   # 干扰1索引
noise_nn = 2  # 干扰2索引

# 期望方向俯仰角切片
phi_idx_expect = np.argmin(np.abs(phi_scan - phiK[expect_n]))
axes[0, 0].plot(np.rad2deg(theta_scan), F_abs[:, phi_idx_expect])
axes[0, 0].axvline(np.rad2deg(thetaK[expect_n]), color='r', linestyle='--', label=f'θ={np.rad2deg(thetaK[expect_n]):.1f}°')
axes[0, 0].set_title('期望方向俯仰角切片')
axes[0, 0].set_xlim(0, 90)
axes[0, 0].set_ylim(np.min(F_abs[:, phi_idx_expect]), np.max(F_abs[:, phi_idx_expect]) + 20)
axes[0, 0].set_xticks(np.arange(0, 91, 15))
axes[0, 0].legend()
axes[0, 0].grid(True)

# 期望方向方位角切片
theta_idx_expect = np.argmin(np.abs(theta_scan - thetaK[expect_n]))
axes[0, 1].plot(np.rad2deg(phi_scan), F_abs[theta_idx_expect, :])
axes[0, 1].axvline(np.rad2deg(phiK[expect_n]), color='r', linestyle='--', label=f'φ={np.rad2deg(phiK[expect_n]):.1f}°')
axes[0, 1].set_title('期望方向方位角切片')
axes[0, 1].set_xlim(0, 360)
axes[0, 1].set_ylim(np.min(F_abs[theta_idx_expect, :]), np.max(F_abs[theta_idx_expect, :]) + 20)
axes[0, 1].set_xticks(np.arange(0, 361, 60))
axes[0, 1].legend()
axes[0, 1].grid(True)

# 干扰方向1俯仰角切片
phi_idx_noise = np.argmin(np.abs(phi_scan - phiK[noise_n]))
axes[1, 0].plot(np.rad2deg(theta_scan), F_abs[:, phi_idx_noise])
axes[1, 0].axvline(np.rad2deg(thetaK[noise_n]), color='r', linestyle='--', label=f'θ={np.rad2deg(thetaK[noise_n]):.1f}°')
axes[1, 0].set_title('干扰方向1俯仰角切片')
axes[1, 0].set_xlim(0, 90)
axes[1, 0].set_ylim(np.min(F_abs[:, phi_idx_noise]), np.max(F_abs[:, phi_idx_noise]) + 20)
axes[1, 0].set_xticks(np.arange(0, 91, 15))
axes[1, 0].legend()
axes[1, 0].grid(True)

# 干扰方向1方位角切片
theta_idx_noise = np.argmin(np.abs(theta_scan - thetaK[noise_n]))
axes[1, 1].plot(np.rad2deg(phi_scan), F_abs[theta_idx_noise, :])
axes[1, 1].axvline(np.rad2deg(phiK[noise_n]), color='r', linestyle='--', label=f'φ={np.rad2deg(phiK[noise_n]):.1f}°')
axes[1, 1].set_title('干扰方向1方位角切片')
axes[1, 1].set_xlim(0, 360)
axes[1, 1].set_ylim(np.min(F_abs[theta_idx_noise, :]), np.max(F_abs[theta_idx_noise, :]) + 20)
axes[1, 1].set_xticks(np.arange(0, 361, 60))
axes[1, 1].legend()
axes[1, 1].grid(True)

# 干扰方向2俯仰角切片
phi_idx_noise_nn = np.argmin(np.abs(phi_scan - phiK[noise_nn]))
axes[2, 0].plot(np.rad2deg(theta_scan), F_abs[:, phi_idx_noise_nn])
axes[2, 0].axvline(np.rad2deg(thetaK[noise_nn]), color='r', linestyle='--', label=f'θ={np.rad2deg(thetaK[noise_nn]):.1f}°')
axes[2, 0].set_title('干扰方向2俯仰角切片')
axes[2, 0].set_xlim(0, 90)
axes[2, 0].set_ylim(np.min(F_abs[:, phi_idx_noise_nn]), np.max(F_abs[:, phi_idx_noise_nn]) + 20)
axes[2, 0].set_xticks(np.arange(0, 91, 15))
axes[2, 0].legend()
axes[2, 0].grid(True)

# 干扰方向2方位角切片
theta_idx_noise_nn = np.argmin(np.abs(theta_scan - thetaK[noise_nn]))
axes[2, 1].plot(np.rad2deg(phi_scan), F_abs[theta_idx_noise_nn, :])
axes[2, 1].axvline(np.rad2deg(phiK[noise_nn]), color='r', linestyle='--', label=f'φ={np.rad2deg(phiK[noise_nn]):.1f}°')
axes[2, 1].set_title('干扰方向2方位角切片')
axes[2, 1].set_xlim(0, 360)
axes[2, 1].set_ylim(np.min(F_abs[theta_idx_noise_nn, :]), np.max(F_abs[theta_idx_noise_nn, :]) + 20)
axes[2, 1].set_xticks(np.arange(0, 361, 60))
axes[2, 1].legend()
axes[2, 1].grid(True)

fig2.suptitle(f'M={M}阵元 环阵CBF二维切片 来波f={f/1e6}MHz', fontsize=16)
plt.tight_layout()
plt.show()

# 打印峰值信息
print("波束形成峰值信息:")
for i, (theta, phi) in enumerate(zip(thetaK, phiK)):
    theta_idx = np.argmin(np.abs(theta_scan - theta))
    phi_idx = np.argmin(np.abs(phi_scan - phi))
    gain = F_abs[theta_idx, phi_idx]
    print(f"目标{i+1}: θ={np.rad2deg(theta):.1f}°, φ={np.rad2deg(phi):.1f}°, 增益={gain:.2f} dB")





















