#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:29:16 2025

@author: jack
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt


# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14                # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16           # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16           # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12          # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12          # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]       # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2                 # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6                # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'    # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'          # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'            # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12

#%% 实现与MATLAB cconv完全一致的圆卷积


def cconv(a, b, n=None):
    """
    实现与MATLAB cconv完全一致的圆卷积
    参数:
        a, b: 输入复数数组
        n: 输出长度 (None表示默认长度len(a)+len(b)-1)
    返回:
        圆卷积结果 (复数数组)
    """
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    # 默认输出长度
    if n is None:
        n = len(a) + len(b) - 1
    # 线性卷积
    linear_conv = np.convolve(a, b, mode='full')
    # 处理不同n的情况
    if n <= 0:
        return np.array([], dtype=complex)
    result = np.zeros(n, dtype=complex)
    if n <= len(linear_conv):
        # n <= M+N-1: 重叠相加
        for k in range(n):
            # 收集所有k + m*n位置的元素
            idx = np.arange(k, len(linear_conv), n)
            result[k] = np.sum(linear_conv[idx])
    else:
        # n > M+N-1: 补零
        result[:len(linear_conv)] = linear_conv

    return result

def genH(h, Nx,):
    Nh = h.size
    H = np.zeros((Nx+Nh-1, Nx),  dtype= complex )
    h = np.pad(h, (0, Nx - 1))
    for j in range(Nx):
        H[:,j] = np.roll(h, j)
    return H

def convMatrix(h, N):  #
    """
    Construct the convolution matrix of size (L+N-1)x N from the
    input matrix h of size L. (see chapter 1)
    Parameters:
        h : numpy vector of length L
        N : scalar value
    Returns:
        H : convolution matrix of size (L+N-1)xN
    """
    col = np.hstack((h, np.zeros(N-1)))
    row = np.hstack((h[0], np.zeros(N-1)))

    from scipy.linalg import toeplitz
    H = toeplitz(col, row)
    return H


#%%

h = np.array([-0.4878, -1.5351, 0.2355])
S = np.array([-0.0155, 2.5770, 1.9238, -0.0629, -0.8105, 0.6727, -1.5924, -0.8007])
s = np.fft.ifft(S) # IFFT
N = s.size
L = h.size

H = convMatrix(h, N)
y = H @ s

cir_s_h = cconv(h, s, N)

lenCP = L # - 1
Acp = np.block([[np.zeros((lenCP, N-lenCP)), np.eye(lenCP)], [np.eye(N)]])

s_cp = Acp @ s                    # add CP

H_cp = convMatrix(h, s_cp.size)
y_cp = H_cp @ s_cp                #  pass freq selected channel

y_remo_cp = y_cp[lenCP:lenCP + N] # receiver, remove cp


H_cp1 = convMatrix(h, s_cp.size)[lenCP:lenCP + N, :]
y_remo_cp1 = H_cp1 @ s_cp        #  pass freq selected channel + remove cp










#%%









#%%



























