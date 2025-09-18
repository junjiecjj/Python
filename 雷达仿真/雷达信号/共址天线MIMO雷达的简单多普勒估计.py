#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 17:04:02 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzk0NTQzNDQyMw==&mid=2247506040&idx=1&sn=41144e1086b4c5597d1529f387fdbb7d&chksm=c2d4dc33be084fb9c942d172dfc17304da27ff537d9f054977b390fcd7fe85a1fe983cf7acf1&mpshare=1&scene=1&srcid=0709yyFUQ7m91XhRS0IkjfBS&sharer_shareinfo=4a8c28aff3580b2b0d0244970fa21faa&sharer_shareinfo_first=522b257d74365432cdfb535a18c635d6&exportkey=n_ChQIAhIQXUCTQeKuYzxvdqZaStMpPRKJAgIE97dBBAEAAAAAAKdLDNTCaEkAAAAOpnltbLcz9gKNyK89dVj0OHx%2FXvMMpG3HyM2zG5m9cU1NWHGKQ9ixrf6ISJDJHJzVqE4G2VIl3DNVOpRfaKj2G%2F%2BcVMCX98neCGIJ%2BOpnzWGIkK5BGTn27I%2FeW3udwY%2BcGPUa3NTMuXthR2U1xOgFSxJf8TkELA3ebhk3VCGA5d3Xhum11qBxNYLegIYFHwuk69a%2FFne4pp7EsxyB%2F%2BGkwwnWWK2b9%2FJP6PgJKYLUbc9FaXn5gVkqxiO6JYFc02w2vBBEyS1lf8ndXG18%2BbrzfKRj8%2FteNj2FQH36gjdKBDDTBxner%2FUe%2BOwm06dAcp2noVQ%3D&acctmode=0&pass_ticket=gDle66rSN3fXA9Wgpu8crVg1EA2nkoEUVA672hdzhlGllgrO5fUnzQu0HP31aII6&wx_header=0#rd

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 清理工作区
# clear  # 在Python中不需要，使用Jupyter时可以用 %reset

# 参数设置
nT = 10
nR = 10
L = 181
M = 101
theta = np.linspace(-90, 90, L).reshape(-1, 1) * np.pi / 180  # 转换为列向量
freq = np.linspace(-0.5, 0.5, M).reshape(-1, 1)  # 转换为列向量

# 发射信号
N = 100
# 生成QPSK信号
X = (np.sign(np.random.randn(nT, N)) + 1j * np.sign(np.random.randn(nT, N))) / np.sqrt(2)

# 噪声方差
varn = 1

# 目标1
fd1 = -0.1
thetat1 = -55 * np.pi / 180
d1 = np.exp(1j * 2 * np.pi * fd1 * np.arange(N)).reshape(1, -1)  # 行向量
aT1 = np.exp(1j * np.pi * np.arange(nT) * np.sin(thetat1)).reshape(-1, 1)  # 列向量
aR1 = np.exp(1j * np.pi * np.arange(nR) * np.sin(thetat1)).reshape(-1, 1)  # 列向量
Y = aR1 @ aT1.conj().T @ X * d1  # 使用广播机制

# 目标2
fd2 = 0.3
thetat2 = -55 * np.pi / 180
d2 = np.exp(1j * 2 * np.pi * fd2 * np.arange(N)).reshape(1, -1)
aT2 = np.exp(1j * np.pi * np.arange(nT) * np.sin(thetat2)).reshape(-1, 1)
aR2 = np.exp(1j * np.pi * np.arange(nR) * np.sin(thetat2)).reshape(-1, 1)
Y = Y + aR2 @ aT2.conj().T @ X * d2

# 目标3
fd3 = -0.2
thetat3 = 0 * np.pi / 180
d3 = np.exp(1j * 2 * np.pi * fd3 * np.arange(N)).reshape(1, -1)
aT3 = np.exp(1j * np.pi * np.arange(nT) * np.sin(thetat3)).reshape(-1, 1)
aR3 = np.exp(1j * np.pi * np.arange(nR) * np.sin(thetat3)).reshape(-1, 1)
Y = Y + aR3 @ aT3.conj().T @ X * d3

# 添加噪声
Y = Y + np.sqrt(varn) * (np.random.randn(nR, N) + 1j * np.random.randn(nR, N)) / np.sqrt(2)
y = Y.reshape(-1, 1)  # 向量化

# 检测
P = np.zeros((L, M), dtype = complex)

for i in range(L):
    aT = np.exp(1j * np.pi * np.arange(nT) * np.sin(theta[i])).reshape(-1, 1)  # 列向量
    aR = np.exp(1j * np.pi * np.arange(nR) * np.sin(theta[i])).reshape(-1, 1)  # 列向量
    A = aR @ aT.conj().T  # 阵列响应矩阵

    for k in range(M):
        dk = np.exp(1j * 2 * np.pi * freq[k] * np.arange(N)).reshape(1, -1)  # 行向量
        # 构建匹配滤波器
        vk = (A @ X * dk).reshape(-1, 1)  # 向量化
        # 计算后相关功率
        P[i, k] = np.abs(y.conj().T @ vk + vk.conj().T @ y)[0,0] / (2 * N * nT * nR)

# 3D绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 创建网格
F, T = np.meshgrid(freq, theta * 180 / np.pi)

# 绘制3D曲面
surf = ax.plot_surface(F, T, P, cmap='viridis', edgecolor='none', alpha=0.8)

# 设置标签
ax.set_xlabel('Frequency')
ax.set_ylabel('DOA (degree)')
ax.set_zlabel('Post-Correlation Power')
ax.set_title('2D Joint Angle-Doppler Estimation')

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)

plt.tight_layout()
plt.show()

# 可选：显示峰值位置
max_val = np.max(P)
max_idx = np.unravel_index(np.argmax(P), P.shape)
print(f"Maximum value: {max_val:.4f} at theta = {theta[max_idx[0]][0]*180/np.pi:.2f}°, freq = {freq[max_idx[1]][0]:.3f}")

# 可选：绘制等高线图
plt.figure(figsize=(10, 6))
contour = plt.contourf(freq.flatten(), theta.flatten() * 180 / np.pi, P, 50, cmap='viridis')
plt.colorbar(contour, label='Post-Correlation Power')
plt.xlabel('Frequency')
plt.ylabel('DOA (degree)')
plt.title('2D Joint Angle-Doppler Estimation (Contour)')
plt.grid(True, alpha=0.3)
plt.show()

# MSE计算（如果需要）
# [val, t_ind] = max(max(P, axis=1))
# [val, f_ind] = max(max(P, axis=0))
# t_mse = (thetat - theta[t_ind])**2
# f_mse = (fd - freq[f_ind])**2
