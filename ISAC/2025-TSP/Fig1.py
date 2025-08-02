#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 13:33:56 2025

@author: jack
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy

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

#%% Eq.(12)(13)(14)
def generateJk(L, N, k):
    if k < 0:
        k = L*N+k
    if k == 0:
        Jk = np.eye(L*N)
    elif k > 0:
        tmp1 = np.zeros((k, L*N-k))
        tmp2 = np.eye(k)
        tmp3 = np.eye(L*N-k)
        tmp4 = np.zeros((L*N - k, k))
        Jk = np.block([[tmp1, tmp2], [tmp3, tmp4]])
    return Jk

L = 2
N = 4

J3 = generateJk(L, N, 3)
J5 = generateJk(L, N, 5)
# J_{q-k}  = J_k @ J_q

J_3 = generateJk(L, N, -3)
J3T = generateJk(L, N, 3).T
J5 = generateJk(L, N, 5)


#%% Eq.(18)

# 产生傅里叶矩阵
def FFTmatrix(row,col):
     mat = np.zeros((row,col),dtype=complex)
     for i in range(row):
          for j in range(col):
               mat[i,j] = 1.0*np.exp(-1j*2.0*np.pi*i*j/row) / (np.sqrt(row)*1.0)
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

generateVec =  [1+1j, 2+2j, 3+3j, 4+1j ]
# generateVec =  [1 , 2  , 3 , 4  ]
X = np.array(generateVec)

N = len(generateVec)
C = CirculantMatric(generateVec, N)

F = FFTmatrix(N, N)
FH = F.T.conjugate() #/ (L * 1.0)

C_hat = np.sqrt(N) * FH @ np.diag(F@C[:,0]) @ F
print(f"C = {C}\nC_hat = {C_hat}")


#%% Eq.(19)(20)
L = 2
N = 4
k = 3
F = FFTmatrix(L*N, L*N)
FH = F.T.conjugate() #/ (L * 1.0)

J_3 = generateJk(L, N, -k)
J5 = generateJk(L, N, L*N-k)

J5_hat = np.sqrt(L*N) * FH @ np.diag(F@J5[:, 0]) @ F      # Eq.(18)
J5_hat1 = np.sqrt(L*N) * FH @ np.diag(F[:, L*N-k]) @ F    # Eq.(19)

delta = np.abs(F[:,k].conj().T - F[:,L*N-k]) # f_{LN−k+1} = f_{k+1}^*




























































































































































































































































































































