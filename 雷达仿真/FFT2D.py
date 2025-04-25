#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 12:04:43 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 12          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 12          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12

def FFTmatrix(N):
     M = np.zeros((N, N), dtype = complex)
     tmp = np.arange(N)
     for n in range(N):
         M[n, :] =  np.exp(-1j*2*np.pi*n*tmp/N)
     return M

M = 12
N = 16
x = np.random.randn(M, N) + 1j * np.random.randn(M, N)

F1 = FFTmatrix(M)
F2 = FFTmatrix(N)
eps = 1e-10
## 1
X_1 = (F2 @ x.T @ F1).T
X = F1 @ x @ F2                                     # 先对列做FFT在对行做FFT
print(np.abs(X_1 - X) < eps)

X1 = np.fft.fft2(x, s = (M, N), axes = (0, 1))      # 先对列做FFT在对行做FFT
X2 = np.fft.fft2(x, s = (N, M), axes = (1, 0))      # 先对行做FFT在对列做FFT
print(np.abs(X-X1) < eps)
print(np.abs(X2-X1) < eps)

## 2
X = (F2 @ x.T @ F1).T
X1 = np.fft.fft2(x, )
eps = 1e-10
print(np.abs(X-X1) < eps)

## 3
X = F2 @ x.T @ F1
X1 = np.fft.fft2(x.T,)
eps = 1e-10
print(np.abs(X-X1) < eps)























































