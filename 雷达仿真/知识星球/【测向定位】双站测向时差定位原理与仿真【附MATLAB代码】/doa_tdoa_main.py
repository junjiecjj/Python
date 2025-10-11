#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:22:47 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv

def doa_tdoa_param(S0, S1, X):
    """计算DOA和TDOA参数"""
    c = 3e8
    r0 = np.sqrt((X[0] - S0[0])**2 + (X[1] - S0[1])**2)
    r1 = np.sqrt((X[0] - S1[0])**2 + (X[1] - S1[1])**2)
    delta_t = (r1 - r0) / c
    tan0 = (X[1] - S0[1]) / (X[0] - S0[0])
    tan1 = (X[1] - S1[1]) / (X[0] - S1[0])
    return delta_t, tan0, tan1

def doa_tdoa_gdop(S0, S1, X, sigma_angle, sigma_S, sigma_t):
    """计算GDOP值"""
    x, y = X[0], X[1]
    x0, y0 = S0[0], S0[1]
    x1, y1 = S1[0], S1[1]
    c = 3e8
    sigma_r = c * sigma_t

    r1 = np.sqrt((x - x1)**2 + (y - y1)**2)
    r0 = np.sqrt((x - x0)**2 + (y - y0)**2)

    sin0 = (y - y0) / r0
    cos0 = (x - x0) / r0

    # 构建矩阵
    C = np.array([
        [-sin0**2 / (y - y0), cos0**2 / (x - x0)],
        [(x - x1)/r1 - (x - x0)/r0, (y - y1)/r1 - (y - y0)/r0]
    ])

    U = np.array([
        [sin0**2 / (y - y0), -cos0**2 / (x - x0)],
        [(x - x0)/r0, (y - y0)/r0]
    ])

    W = np.array([
        [0, 0],
        [(x - x1)/r1, (y - y1)/r1]
    ])

    # 协方差矩阵
    Rv = np.array([
        [sigma_angle**2, 0],
        [0, sigma_r**2]
    ])

    Rs = np.array([
        [sigma_S**2, 0],
        [0, sigma_S**2]
    ])

    # 计算GDOP
    C_pinv = pinv(C)
    Pdx = C_pinv @ (Rv + U @ Rs @ U.T + W @ Rs @ W.T) @ C_pinv.T
    gdop = np.sqrt(Pdx[0, 0] + Pdx[1, 1])

    return gdop

# 第一部分
S0 = np.array([-5, 0]) * 1e3
S1 = np.array([5, 0]) * 1e3
sigma_angle = 3e-3
sigma_S = 5
sigma_t = 20e-9

N = 100
M = 60e3
x = np.linspace(-M, M, N)
y = np.linspace(-M, M, N)
gdop = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        X = [x[i], y[j]]
        gdop[j, i] = doa_tdoa_gdop(S0, S1, X, sigma_angle, sigma_S, sigma_t) / 1000

plt.figure(1, figsize=(10, 8))
contour = plt.contour(x/1000, y/1000, gdop, levels=np.arange(0, 10.5, 0.5))
plt.clabel(contour, inline=True, fontsize=8)
plt.xlabel('x/km')
plt.ylabel('y/km')
plt.title('sigma_a=3mrad, sigma_S=5m, sigma_t=20ns, value of GDOP/km')
plt.grid(True)

# 第二部分
S0 = np.array([-5, 0]) * 1e3
S1 = np.array([5, 0]) * 1e3
sigma_angle = 0
sigma_S = 0
sigma_t = 20e-9

gdop = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        X = [x[i], y[j]]
        gdop[j, i] = doa_tdoa_gdop(S0, S1, X, sigma_angle, sigma_S, sigma_t) / 1000

plt.figure(2, figsize=(10, 8))
contour = plt.contour(x/1000, y/1000, gdop, levels=np.arange(0, 1.1, 0.1))
plt.clabel(contour, inline=True, fontsize=8)
plt.xlabel('x/km')
plt.ylabel('y/km')
plt.title('sigma_a=0, sigma_S=0, sigma_t=20ns, value of GDOP/km')
plt.grid(True)

# 第三部分
S0 = np.array([-5, 0]) * 1e3
S1 = np.array([5, 0]) * 1e3
sigma_angle = 3e-3
sigma_S = 0
sigma_t = 0

gdop = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        X = [x[i], y[j]]
        gdop[j, i] = doa_tdoa_gdop(S0, S1, X, sigma_angle, sigma_S, sigma_t) / 1000

plt.figure(3, figsize=(10, 8))
contour = plt.contour(x/1000, y/1000, gdop, levels=np.arange(0, 5.5, 0.5))
plt.clabel(contour, inline=True, fontsize=8)
plt.xlabel('x/km')
plt.ylabel('y/km')
plt.title('sigma_a=3mrad, sigma_S=0, sigma_t=0, value of GDOP/km')
plt.grid(True)

# 第四部分
S0 = np.array([-5, 0]) * 1e3
S1 = np.array([5, 0]) * 1e3
sigma_angle = 0
sigma_S = 5
sigma_t = 0

gdop = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        X = [x[i], y[j]]
        gdop[j, i] = doa_tdoa_gdop(S0, S1, X, sigma_angle, sigma_S, sigma_t) / 1000

plt.figure(4, figsize=(10, 8))
contour = plt.contour(x/1000, y/1000, gdop, levels=np.arange(0, 1.1, 0.1))
plt.clabel(contour, inline=True, fontsize=8)
plt.xlabel('x/km')
plt.ylabel('y/km')
plt.title('sigma_a=0, sigma_S=5m, sigma_t=0, value of GDOP/km')
plt.grid(True)

plt.tight_layout()
plt.show()
