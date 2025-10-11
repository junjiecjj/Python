#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 10:06:58 2025

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


# Propagator Method
# 对应MATLAB代码：南京航空航天大学 电子工程系 张小飞

derad = np.pi / 180
radeg = 180 / np.pi
twpi = 2 * np.pi
kelm = 16  # 阵元数量
dd = 0.5  # 阵元间距
d = np.arange(0, kelm * dd, dd)
iwave = 3   # 波源数量
theta = np.array([10, 20, 30])  # 波达方向
pw = np.array([1, 0.8, 0.7]).reshape(-1, 1)  # 功率

nv = np.ones(kelm)        # 归一化噪声方差
snr = 20              # 输入信噪比 (dB)
snr0 = 10**(snr / 10)
n = 200  # 样本数量

# 方向矩阵
A = np.exp(-1j * twpi * d.reshape(-1, 1) * np.sin(theta * derad))
K = len(d)
cr = np.zeros((K, K))
L = len(theta)

# 生成信号
data = np.random.randn(L, n)
data = np.sign(data)
s = np.diag(pw.flatten()) @ data

# 接收信号
received_signal = A @ s
# 添加噪声
cx = received_signal + np.diag(np.sqrt(nv / snr0 / 2)) @ (np.random.randn(K, n) + 1j * np.random.randn(K, n))
Rxx = cx @ cx.conj().T / n

# Propagator Method
G = Rxx[:, :iwave]
H = Rxx[:, iwave:]
P = np.linalg.inv(G.conj().T @ G) @ G.conj().T @ H
Q = np.hstack([P.conj().T, -np.eye(kelm - iwave)])

# 角度扫描
angle_scan = np.zeros(361)
SP = np.zeros(361, dtype=complex)

for iang in range(361):
    angle_scan[iang] = (iang - 181) / 2
    phim = derad * angle_scan[iang]
    a = np.exp(-1j * twpi * d * np.sin(phim)).reshape(-1, 1)
    SP[iang] = 1 / (a.conj().T @ Q.conj().T @ Q @ a)

SP_abs = np.abs(SP)
SPmax = np.max(SP_abs)
SP_db = 10 * np.log10(SP_abs / SPmax)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(angle_scan, SP_db, '-k', linewidth=2)
plt.xlabel('angle (degree)')
plt.ylabel('magnitude (dB)')
plt.axis([-90, 90, -60, 0])
plt.xticks(np.arange(-90, 91, 30))
plt.grid(True)
plt.legend(['Propagator Method'])
plt.title('Propagator Method DOA Estimation')
plt.show()
