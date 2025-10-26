#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 20:12:23 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import chebwin
import scipy
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


def AFmean_lfm_OFDM_single(N_c, T_b, mu, x, y, weight):
    """lfm_ofdm单脉冲模糊函数均值"""
    # 创建网格
    X, Y = np.meshgrid(x, y)

    # 子载波频率间隔
    delta_f = 1 / T_b
    # 子载波频率
    f = np.arange(N_c) / T_b
    # 符号能量
    E = T_b * np.sum(np.abs(weight))

    amf, amt = X.shape

    # 计算模糊函数
    a1 = T_b - np.abs(X * T_b)
    b1 = np.sinc((delta_f * Y + mu * X * T_b) * a1)
    d1 = np.exp(1j * np.pi * (delta_f * Y + mu * X * T_b) * (T_b + X * T_b))

    c1 = np.zeros((amf, amt), dtype=complex)
    for k in range(N_c):
        temp = np.exp(1j * 2 * np.pi * f[k] * X * T_b - 1j * np.pi * mu * (X * T_b)**2)
        c1 += weight[k] * temp

    AF_u = a1 * b1 * c1 * d1
    AF_u = AF_u / E
    AF_u = np.abs(AF_u)

    # 函数内部的绘图 - 第5个图形
    plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, AF_u, cmap='jet', rstride = 5, cstride = 5, edgecolor='none')
    ax.set_title('lfm-OFDM模糊函数', fontsize=14)
    ax.set_xlabel('归一化时延', fontsize=14)
    ax.set_ylabel('归一化频移', fontsize=14)
    plt.grid(True)
    plt.colorbar(surf)

    return AF_u

def AFmean_single_symbol(N_c, T_b, x, y, d):
    """单个OFDM符号的平均模糊函数"""
    # 创建网格
    X, Y = np.meshgrid(x, y)

    # 子载波频率间隔
    delta_f = 1 / T_b
    # 子载波频率
    f = np.arange(N_c) / T_b
    # 符号能量
    E = T_b * np.sum(d)

    # 单周期模糊函数均值
    a1 = T_b - np.abs(X * T_b)
    b1 = np.sinc(delta_f * Y * a1)
    c1 = np.zeros((X.shape[0], X.shape[1]), dtype=complex)
    d1 = np.exp(1j * np.pi * delta_f * Y * (T_b + X * T_b))

    for k in range(N_c):
        temp = np.exp(1j * 2 * np.pi * f[k] * X * T_b)
        c1 += d[k] * temp

    AF_m0 = a1 * b1 * c1 * d1
    AF_m0 = AF_m0 / E
    AF_m1 = np.abs(AF_m0)

    return AF_m1

# 主程序
# 子载波数
N_c = 16
# 脉冲持续时间
T_b = 1e-6
# 归一化时间
x = np.linspace(-1, 1, 32 * N_c)
# 归一化频率
y = np.arange(-5, 5.01, 0.01)  # 确保包含5
# 调频率
mu = 16 / T_b**2
# 符号权重
# w = np.ones(N_c)
w = scipy.signal.windows.chebwin(N_c, 50)

# 创建网格
X, Y = np.meshgrid(x, y)
amf, amt = X.shape

# 计算模糊函数
AF_ofdm = AFmean_single_symbol(N_c, T_b, x, y, w)
AF_ocdm = AFmean_lfm_OFDM_single(N_c, T_b, mu, x, y, w)

# 图形1: OFDM模糊函数3D图
plt.figure()
ax1 = plt.axes(projection='3d')
surf1 = ax1.plot_surface(X, Y, AF_ofdm, cmap='jet',rstride = 5, cstride = 5, edgecolor='none')
ax1.set_xlabel('归一化时延', fontsize=14)
ax1.set_ylabel('归一化频移', fontsize=14)
ax1.set_zlim(0, 1)
ax1.view_init(elev=26, azim=-46)
plt.grid(True)

# 图形2: OCDM模糊函数3D图
plt.figure()
ax2 = plt.axes(projection='3d')
surf2 = ax2.plot_surface(X, Y, AF_ocdm, cmap='jet', rstride = 5, cstride = 5, edgecolor='none')
ax2.set_xlabel('归一化时延', fontsize=14)
ax2.set_ylabel('归一化频移', fontsize=14)
ax2.set_zlim(0, 1)
ax2.view_init(elev=26, azim=-46)
plt.grid(True)

# 图形3: 距离模糊函数比较
# 找到y=0对应的索引
y_zero_idx = np.argmin(np.abs(y))
AC_ofdm = AF_ofdm[y_zero_idx, :]
AC_ocdm = AF_ocdm[y_zero_idx, :]

plt.figure()
plt.plot(x, AC_ofdm, 'k', linewidth=1.5, label='OFDM')
plt.plot(x, AC_ocdm, 'b', linewidth=1.5, label='OCDM')
plt.title('距离模糊函数比较', fontsize=14)
plt.xlabel('归一化时延', fontsize=14)
plt.ylabel('归一化幅度', fontsize=14)
plt.legend()
plt.grid(True)

# 图形4: 速度模糊函数比较
# 找到x=0对应的索引
x_zero_idx = np.argmin(np.abs(x))
DC_ofdm = AF_ofdm[:, x_zero_idx]
DC_ocdm = AF_ocdm[:, x_zero_idx]

plt.figure()
plt.plot(y, DC_ofdm, 'k', linewidth=1.5, label='OFDM')
plt.plot(y, DC_ocdm, 'b', linewidth=1.5, label='OCDM')
plt.title('速度模糊函数比较', fontsize=14)
plt.xlabel('归一化频移', fontsize=14)
plt.ylabel('归一化幅度', fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
