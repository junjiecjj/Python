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
# from matplotlib.font_manager import FontProperties
# import commpy

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

# # generateVec =  [1+1j, 2+2j, 3+3j, 4+1j ]
# generateVec =  [1 , 2  , 3 , 4  ]
# X = np.array(generateVec)
# L = len(generateVec)
# A = CirculantMatric(X, L)
N = 8
h = np.array([-0.4878, -1.5351, 0.2355])
s = np.array([-0.0155, 2.5770, 1.9238, -0.0629, -0.8105, 0.6727, -1.5924, -0.8007])

lin_s_h = scipy.signal.convolve(h, s)
cir_s_h = circularConvolve(h, s, N)

Ncp = 2
s_cp = np.hstack((s[-Ncp:], s))
lin_scp = scipy.signal.convolve(h, s_cp)
r = lin_scp[Ncp:Ncp+N]
print(f"lin_scp = \n{lin_scp[Ncp:Ncp+N]}\ncir_s_h = \n{cir_s_h}")


R = scipy.fft.fft(r, N)
H = scipy.fft.fft(h, N)
S = scipy.fft.fft(s, N)

r1 = scipy.fft.ifft(S*H)
print(f"r1 = \n{r1}\ncir_s_h = \n{cir_s_h}")

#%%

import numpy as np
from scipy.fft import fft, ifft

# 1. 使用FFT的频域方法（最推荐）
def circular_convolve(x, h):
    """使用FFT计算圆卷积"""
    return ifft(fft(x) * fft(h)).real  # 取实数部分消除浮点误差

# 2. 手动实现时域圆卷积

def circular_convolve_direct(x, h):
    """时域直接计算圆卷积"""
    N = len(x)
    y = np.zeros(N)
    for n in range(N):
        for k in range(N):
            y[n] += x[k] * h[(n - k) % N]
    return y

# 3. 使用scipy.signal的circulant矩阵
from scipy.linalg import circulant
def circular_convolve_matrix(x, h):
    """使用循环矩阵计算圆卷积"""
    return circulant(h).T @ x



def isac_circular_convolution():
    # OFDM参数
    N = 64  # 子载波数
    cp_len = 16  # 循环前缀长度

    # 生成信号
    x = np.random.randn(N) + 1j*np.random.randn(N)  # 随机QAM符号
    h = np.zeros(N, dtype=complex)  # 信道脉冲响应
    h[[0, 10, 20]] = [1.0, 0.6, 0.3]  # 直射路径+多径

    # 添加循环前缀
    x_cp = np.concatenate([x[-cp_len:], x])

    # 通过信道（线性卷积）
    y_linear = np.convolve(x_cp, h, mode='same')[:N+cp_len]

    # 去除CP后计算圆卷积
    y_no_cp = y_linear[cp_len:cp_len+N]
    y_circular = circular_convolve(x, h)

    # 可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title("线性卷积输出")
    plt.plot(np.abs(y_no_cp))

    plt.subplot(122)
    plt.title("圆卷积输出")
    plt.plot(np.abs(y_circular))
    plt.tight_layout()
    plt.show()

isac_circular_convolution()

