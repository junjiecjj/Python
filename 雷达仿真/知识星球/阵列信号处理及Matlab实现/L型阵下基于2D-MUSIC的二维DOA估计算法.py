#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 10:07:50 2025

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



# two dimensional DOA estimation using 2D-MUSIC algorithm for L-shaped array
# Developed by xiaofei zhang (南京航空航天大学 电子工程系 张小飞）

twpi = 2 * np.pi
rad = np.pi / 180
deg = 180 / np.pi

kelm = 8
snr = 10
iwave = 3
theta = np.array([10, 30, 50])  # elevation angles
fe = np.array([15, 25, 35])     # azimuth angles
n = 100
dd = 0.5
d = np.arange(0, kelm * dd, dd)
d1 = np.arange(dd, kelm * dd, dd)  # for y-axis array

# Array manifold matrices
Ax = np.exp(-1j * twpi * d.reshape(-1, 1) * (np.sin(theta * rad) * np.cos(fe * rad)))
Ay = np.exp(-1j * twpi * d1.reshape(-1, 1) * (np.sin(theta * rad) * np.sin(fe * rad)))
A = np.vstack((Ax, Ay))

# Signal generation
S = np.random.randn(iwave, n)
X = A @ S

# Add noise
X1 = X + np.random.randn(*X.shape) * 10**(-snr/20)  # simulated AWGN
Rxx = X1 @ X1.conj().T / n

# Eigen decomposition
D, EV = np.linalg.eig(Rxx)
# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(D)[::-1]
EVA = D[idx]
EV = EV[:, idx]

# Noise subspace
Un = EV[:, iwave:]

# 2D MUSIC spectrum
SP = np.zeros((90, 90))
thet = np.zeros(90)
f = np.zeros(90)

for ang1 in range(90):
    for ang2 in range(90):
        thet[ang1] = ang1
        phim1 = thet[ang1] * rad
        f[ang2] = ang2
        phim2 = f[ang2] * rad

        a1 = np.exp(-1j * twpi * d * np.sin(phim1) * np.cos(phim2)).reshape(-1, 1)
        a2 = np.exp(-1j * twpi * d1 * np.sin(phim1) * np.sin(phim2)).reshape(-1, 1)
        a = np.vstack((a1, a2))

        SP[ang1, ang2] = 1 / np.abs(a.conj().T @ Un @ Un.conj().T @ a)

SP = np.abs(SP)
SPmax = np.max(SP)
SP = SP / SPmax

# 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for 3D plotting
THET, F = np.meshgrid(thet, f, indexing='ij')
surf = ax.plot_surface(THET, F, SP, cmap='viridis', linewidth=0, antialiased=True)

ax.set_xlabel('elevation (degree)')
ax.set_ylabel('azimuth (degree)')
ax.set_zlabel('magnitude')
ax.set_title('2D-MUSIC Spectrum for L-shaped Array')

# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# Optional: 2D contour plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(THET, F, SP, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.xlabel('elevation (degree)')
plt.ylabel('azimuth (degree)')
plt.title('2D-MUSIC Spectrum (Contour)')
plt.grid(True, alpha=0.3)
plt.show()






