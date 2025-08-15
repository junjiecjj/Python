#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 17:23:38 2025

@author: jack
"""
import scipy
import numpy as np
# import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12

#%%
# 生成 循环矩阵
def CirculantMatric(gen, row):
     if type(gen) == list:
          col = len(gen)
     elif type(gen) == np.ndarray:
          col = gen.size
     row = col

     mat = np.zeros((row, col), dtype = gen.dtype)
     mat[:, 0] = gen
     for i in range(1, row):
          mat[:,i] = np.roll(gen, i)
     return mat

def circularConvolve(h, s, N):
    if h.size < N:
        h = np.hstack((h, np.zeros(N-h.size)))
    col = N
    row = s.size
    H = np.zeros((row, col), dtype = s.dtype)
    H[:, 0] = h
    for i in range(1, row):
          H[:,i] = np.roll(h, i)
    res = H @ s
    return res

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

# # generateVec =  [1+1j, 2+2j, 3+3j, 4+1j ]
# generateVec =  [1 , 2  , 3 , 4  ]
# X = np.array(generateVec)
# L = len(generateVec)
# A = CirculantMatric(X, L)

# a = np.array([2, 1, 2, 1])
# b = np.array([1, 2, 3, 4])
# # c = cconv(a,b,4)
# # c1 = circularConvolve(a,b,4)

h = np.array([-0.4878, -1.5351, 0.2355])
s = np.array([-0.0155, 2.5770, 1.9238, -0.0629, -0.8105, 0.6727, -1.5924, -0.8007])
N =  s.size

lin_s_h = scipy.signal.convolve(h, s)
cir_s_h = cconv(h, s, N)
print(f"lin_s_h = \n    {lin_s_h}\ncir_s_h = \n    {cir_s_h}")

Ncp = h.size - 1 # 换成其他的也行
s_cp = np.hstack((s[-Ncp:], s))
lin_scp = scipy.signal.convolve(h, s_cp)
r = lin_scp[Ncp:Ncp+N]
print(f" r = \n    {r}\n cir_s_h = \n    {cir_s_h}")

# 14.2.3 Verifying DFT property
R = scipy.fft.fft(r, N)
H = scipy.fft.fft(h, N)
S = scipy.fft.fft(s, N)

print(f"R = {R}")
print(f"H*S = {H*S}")

r1 = scipy.fft.ifft(S*H)
print(f"r1 = \n{r1}\ncir_s_h = \n{cir_s_h}")

#%%

import numpy as np

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

# 测试用例
a = np.array([1+1j, 2-2j, 3+0.7j])
b = np.array([2-1j, 1-2j, 4+1.7j, 1.1-9j, 6.8-4.4j, 6.7+4j])

# 对应MATLAB的输出
c = cconv(a, b)
c1 = cconv(a, b, 5)
c2 = cconv(a, b, 10)
c3 = cconv(a, b, 2)

print(f"\n\nc = {c}")
print(f"c1 = {c1}")
print(f"c2 = {c2}")
print(f"c3 = {c3}")


# 对应MATLAB的输出
C = cconv(b, a)
C1 = cconv(b, a, 5)
C2 = cconv(b, a, 10)
C3 = cconv(b, a, 2)

print(f"\n\nC = {C}")
print(f"C1 = {C1}")
print(f"C2 = {C2}")
print(f"C3 = {C3}")
















