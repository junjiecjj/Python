#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 09:58:35 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
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



# 参数设置
M = 18                                     # 天线数量
L = 100                                    # 采样点数
thetas = 10                                # 信号入射角度
thetai = [-30, 30]                         # 干扰入射角度
n = np.arange(0, M).reshape(-1, 1)        # 天线索引

# 方向矢量
vs = np.exp(-1j * np.pi * n * np.sin(thetas / 180 * np.pi))        # 信号方向矢量
vi = np.exp(-1j * np.pi * n * np.sin(np.array(thetai) / 180 * np.pi))  # 干扰方向矢量

f = 16000                                  # 载波频率
t = np.arange(0, L) / 200
snr = 10                                   # 信噪比
inr = 10                                   # 干噪比

# 构造信号
xs = np.sqrt(10 ** (snr / 10)) * vs * np.exp(1j * 2 * np.pi * f * t)  # 有用信号
xi = np.sqrt(10 ** (inr / 10) / 2) * vi @ (np.random.randn(len(thetai), L) + 1j * np.random.randn(len(thetai), L))  # 干扰信号
noise = (np.random.randn(M, L) + 1j * np.random.randn(M, L)) / np.sqrt(2)  # 噪声

# 接收信号处理
X = xi + noise                            # 含噪接收信号
R = X @ X.conj().T / L                    # 构造协方差矩阵
wop1 = np.linalg.inv(R) @ vs / (vs.conj().T @ np.linalg.inv(R) @ vs)  # 波束形成权重

# 波束图扫描
sita = 48 * np.arange(-1, 1.001, 0.001)   # 扫描方向范围
v = np.exp(-1j * np.pi * n * np.sin(sita / 180 * np.pi))  # 扫描方向矢量
B = np.abs(wop1.conj().T @ v)

# 修正：将B转换为一维数组
B = B.flatten()

# 绘图
plt.figure()
plt.plot(sita, 20 * np.log10(B / np.max(B)), 'k')
plt.title('波束图')
plt.xlabel('角度/degree')
plt.ylabel('波束图/dB')
plt.grid(True)
plt.axis([-48, 48, -50, 0])
plt.show()





























