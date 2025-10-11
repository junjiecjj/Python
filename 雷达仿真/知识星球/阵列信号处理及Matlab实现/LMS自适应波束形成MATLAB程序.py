#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 10:05:41 2025

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

# LMS波束形成的Python仿真程序
# 对应MATLAB代码：南京航空航天大学 电子工程系 张小飞

M = 16                                     # 天线数量
K = 2                                      # 信源数量
theta = np.array([0, 30])                 # 波达方向
d = 0.3                                   # 天线间距
N = 500                                   # 采样点数
Meann = 0                                 # 噪声均值
varn = 1                                  # 噪声方差
SNR = 20                                  # 信噪比
INR = 20                                  # 干扰噪声比

rvar1 = np.sqrt(varn) * 10**(SNR/20)      # 信号功率
rvar2 = np.sqrt(varn) * 10**(INR/20)      # 干扰功率

# 生成源信号
s = np.array([
    rvar1 * np.exp(1j * 2 * np.pi * (50 * 0.001 * np.arange(N))),
    rvar2 * np.exp(1j * 2 * np.pi * (100 * 0.001 * np.arange(N) + np.random.rand()))
])

# 方向矩阵
A = np.exp(-1j * 2 * np.pi * d * np.arange(M).reshape(-1, 1) * np.sin(theta * np.pi / 180))

# 噪声
e_noise = np.sqrt(varn/2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))

# 接收数据
Y = A @ s + e_noise

# LMS算法
L_beam = 200
de = s[0, :]                              # 期望信号
mu = 0.0005                               # 步长
w = np.zeros(M, dtype=complex)            # 权重向量
y = np.zeros(N, dtype=complex)            # 输出信号
e = np.zeros(N, dtype=complex)            # 误差信号

for k in range(N):
    y[k] = w.conj().T @ Y[:, k]           # 预测下一个样本和误差
    e[k] = de[k] - y[k]                   # 误差
    w = w + mu * Y[:, k] * e[k].conj()    # 更新权重矩阵和步长

# 使用LMS方法进行波束形成
beam = np.zeros(L_beam)
for i in range(L_beam):
    a = np.exp(-1j * 2 * np.pi * d * np.arange(M) * np.sin(-np.pi/2 + np.pi * i / L_beam))
    beam[i] = 20 * np.log10(np.abs(w.conj().T @ a))

# 绘图
plt.figure(figsize=(10, 6))
angle = np.linspace(-90, 90-180/L_beam, L_beam)
plt.plot(angle, beam)
plt.grid(True)
plt.xlabel('方向角/degree')
plt.ylabel('幅度响应/dB')
plt.title('LMS波束形成方向图')
plt.show()

plt.figure(figsize=(10, 6))
en = np.abs(e)**2
plt.semilogy(en)
plt.grid(True)
plt.xlabel('迭代次数')
plt.ylabel('MSE')
plt.title('LMS算法收敛曲线')
plt.show()
