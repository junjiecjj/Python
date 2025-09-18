#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 13:33:56 2025

@author: jack
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import commpy

from Tools import freqDomainView

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
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

#%%
def srrcFunction(beta, L, span, Tsym = 1):
    # Function for generating rectangular pulse for the given inputs
    # L - oversampling factor (number of samples per symbol)
    # span - filter span in symbol durations
    # Returns the output pulse p(t) that spans the discrete-time base -span:1/L:span. Also returns the filter delay.
    t = np.arange(-span*Tsym/2, span*Tsym/2 + 0.5/L, Tsym/L)
    A = np.sin(np.pi*t*(1-beta)/Tsym) + 4*beta*t/Tsym * np.cos(np.pi*t*(1+beta)/Tsym)
    B = np.pi*t/Tsym * (1-(4*beta*t/Tsym)**2)
    p = 1/np.sqrt(Tsym) * A/B
    p[np.argwhere(np.isnan(p))] = 1
    p[np.argwhere(np.isinf(p))] = beta/(np.sqrt(2*Tsym)) * ((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
    filtDelay = (len(p)-1)/2
    p = p / np.sqrt(np.sum(np.power(p, 2))) # Power normalize.
    return p, t, filtDelay

# 产生傅里叶矩阵
def FFTmatrix(L ):
     mat = np.zeros((L, L), dtype = complex)
     for i in range(L):
          for j in range(L):
               mat[i,j] = 1.0*np.exp(-1j*2.0*np.pi*i*j/L) / (np.sqrt(L)*1.0)
     return mat

# 生成 循环矩阵
def CirculantMatric(gen, row):
     if type(gen) == list:
         col = len(gen)
     elif type(gen) == np.ndarray:
         col = gen.size
     row = col
     mat = np.zeros((row, col), np.complex128)
     mat[0, :] = gen
     for i in range(1, row):
         mat[i, :] = np.roll(gen, i)
     return mat

#%%
def geneJk(N, k):
    flag = 0
    if k < 0:
        flag = 1
        k = -k
    if k == 0:
        Jk = np.eye(N)
    tmp1 = np.zeros((N - k, k))
    tmp2 = np.eye(N - k)
    tmp3 = np.zeros((k,k))
    tmp4 = np.zeros((k, N - k))
    Jk = np.block([[tmp1, tmp2], [tmp3, tmp4]])

    if flag == 1:
        Jk = Jk.T

    return Jk

def geneTildeJk(N, k):
    flag = 0
    if k < 0:
        flag = 1
        k = -k
    if k == 0:
        Jk = np.eye(N)
    tmp1 = np.zeros((N - k, k))
    tmp2 = np.eye(N - k)
    tmp3 = np.eye(k)
    tmp4 = np.zeros((k, N - k))
    Jk = np.block([[tmp1, tmp2], [tmp3, tmp4]])

    if flag == 1:
        Jk = Jk.T

    return Jk

#%% Eq.(9)
# L = 2
N = 8
J3 = geneJk(N, 3)
J_3 = geneJk(N, -3)
J3T = geneJk(N, 3).T
x = np.random.randn(N) + 1j * np.random.randn(N)
# Eq.9
rk = x.conj().T @ J3 @ x
r_k_star = (x.conj().T @ J_3 @ x).conj()

#%% Eq.(12)
N = 8
Jt3 = geneTildeJk(N, 3)
Jt_3 = geneTildeJk(N, -3)
Jt3T = geneTildeJk(N, 3).T

# Eq.(12)
rtk = x.conj().T @ Jt3 @ x
rt_k_star = (x.conj().T @ Jt_3 @ x).conj()

#%% Eq.(21), 离散信号的Wiener-Khinchin theorem定理;
## F @ Rk = np.abs(FN @ x)**2
# 自相关函数
Rk = np.zeros(N, dtype = complex)
for k in range(N):
    Jk =  geneTildeJk(N, k)
    Rk[k] = x.conj().T @ Jk @ x

FN = FFTmatrix(N )
Rk1 = np.sqrt(N) * FN.conj().T @ (np.abs(FN @ x)**2)

#%% Eq.( )







#%% Eq.( )























































































































































































