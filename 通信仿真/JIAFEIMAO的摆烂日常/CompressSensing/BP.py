#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:43:44 2024

@author: jack
https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247486265&idx=1&sn=4e6dc3bea7aa30fcf0d7ee78b6dd99f2&chksm=c15b9a17f62c13014c68c2af5c05fcd4b1183d326c246e7cab1ec6cb5a87a25114fb72487956&cur_album_id=3587607448191893505&scene=190#rd

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from numpy.linalg import norm
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
# rcParams['figure.dpi'] = 300

np.random.seed(42)
n, m, k = 100, 50, 10
x_true = np.zeros(n)
x_true[np.random.choice(n, k, replace=False)] = np.random.randn(k) * 10
phi = np.random.randn(m, n)
y = phi @ x_true

# 定义L1范数最小化线性规划求解
def linear_programming(phi, y):
    c = np.ones(n)
    A_eq = np.vstack([phi, -phi])
    b_eq = np.hstack([y, -y])
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(None, None), method='highs')
    return result.x if result.success else np.zeros(n)
# 定义ISTA算法求解
def ista(phi, y, alpha=0.001, max_iter=500):
    x = np.zeros(n)
    for _ in range(max_iter):
        grad = phi.T @ (phi @ x - y)
        x = np.sign(x - alpha * grad) * np.maximum(np.abs(x - alpha * grad) - alpha, 0)
    return x
# 定义FISTA算法求解
def fista(phi, y, alpha=0.001, max_iter=500):
    x = np.zeros(n)
    t = 1
    z = x.copy()
    for _ in range(max_iter):
        x_old = x.copy()
        grad = phi.T @ (phi @ z - y)
        x = np.sign(z - alpha * grad) * np.maximum(np.abs(z - alpha * grad) - alpha, 0)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        z = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new
    return x

x_lp = linear_programming(phi, y)
x_ista = ista(phi, y)
x_fista = fista(phi, y)

plt.figure(figsize=(8, 9), constrained_layout=True)
plt.subplot(3, 1, 1)
plt.plot(x_true, label="真实稀疏信号", color="blue", linewidth=2)
plt.plot(x_lp, linestyle="--", color="red", label="线性规划重建", linewidth=2)
plt.xlabel("信号索引", fontsize=14)
plt.ylabel("幅值", fontsize=14)
plt.title("线性规划重建效果", fontsize=16)
plt.legend(loc="upper right", fontsize=12, frameon=False)
plt.grid(True, linestyle='--', alpha=0.6)
plt.subplot(3, 1, 2)
plt.plot(x_true, label="真实稀疏信号", color="blue", linewidth=2)
plt.plot(x_ista, linestyle="-.", color="red", label="ISTA重建", linewidth=2)
plt.xlabel("信号索引", fontsize=14)
plt.ylabel("幅值", fontsize=14)
plt.title("ISTA 重建效果", fontsize=16)
plt.legend(loc="upper right", fontsize=12, frameon=False)
plt.grid(True, linestyle='--', alpha=0.6)
plt.subplot(3, 1, 3)
plt.plot(x_true, label="真实稀疏信号", color="blue", linewidth=2)
plt.plot(x_fista, linestyle=":", color="red", label="FISTA重建", linewidth=2)
plt.xlabel("信号索引", fontsize=14)
plt.ylabel("幅值", fontsize=14)
plt.title("FISTA 重建效果", fontsize=16)
plt.legend(loc="upper right", fontsize=12, frameon=False)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()






































