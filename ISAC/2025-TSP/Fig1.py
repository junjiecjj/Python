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

#%%
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
    p = p / np.sqrt(np.sum(np.power(p, 2))) # both Add and Delete this line is OK.
    return p, t, filtDelay

#%% Eq.(12)(13)(14)
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
def FFTmatrix(row, col):
     mat = np.zeros((row, col), dtype = complex)
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
J3T = generateJk(L, N, k).T
J5 = generateJk(L, N, L*N-k)

J5_hat = np.sqrt(L*N) * FH @ np.diag(F@J5[:, 0]) @ F      # Eq.(18)
J5_hat1 = np.sqrt(L*N) * FH @ np.diag(F[:, L*N-k]) @ F    # Eq.(19)

delta = np.abs(F[:,k].conj().T - F[:,L*N-k]) # f_{LN−k+1} = f_{k+1}^*

## Eq.(21), (F@a)^H = F^* @ a^*
(F@J5[:,0]).conj() == (F.conj()@J5[:,0].conj())

## Eq.(22)
N = 5
L = 3
x = np.random.randn(N) + 1j * np.random.randn(N)
# L = 4
x_up = np.vstack((x, np.zeros((L-1, x.size))))
x_up = x_up.T.flatten()

FLN = FFTmatrix(L*N, L*N)
FN = FFTmatrix(N, N)

ff1 = FLN @ x_up # Eq.(22) 的sqrt(1/LN)应该是sqrt(1/N)
ff2 = np.sqrt(1/(L)) * np.tile(x @ FN.T, L)

## Eq.(23)
Tsym = 1
span = 6
L = 4
p = srrcFunction(0.35, L, span, Tsym = Tsym)


## Eq.(39)
pi = np.pi
m = 2
N = 12
A = np.exp(1j * 2*pi * m * np.arange(N) / N)
A.sum() # 如果 m 是 N 的整数倍 or 0,和为N; 否则=0;


## Eq.(51)
N = 3
V = (np.random.randn(N) + 1j * np.random.randn(N))[:,None]
S = (np.random.randn(N) + 1j * np.random.randn(N))[:,None]

s1 = np.abs(V.conj().T@S)**2
s2 = V.conj().T @ S @ S.conj().T @ V
tilde_S = (S @ S.conj().T).T.flatten()[:, None]
s3 = np.kron(V.T, V.conj().T) @ tilde_S
s4 = tilde_S.conj().T @ np.kron(V.conj(), V)

## Eq.(58)
N = 4
F = FFTmatrix(N, N)
IN2 =  np.eye(N**2)

i = 0
j = 1
vi = F[:, i][:,None]
vj = F[:, j][:,None]
tmp = np.kron(vi.T, vi.conj().T) @ IN2 @ np.kron(vj.conj(), vj)
## i == j, tmp = 1; i!=j, tmp = 0

## validate Eq.(59)
N = 5
F = FFTmatrix(N, N)
i = 0
j = 1
vi = F[:, i][:,None]
vj = F[:, j][:,None]
kapa = 4
tmp = np.hstack((np.array(kapa-2), np.zeros(N)))
S1 = np.tile(tmp, N-1)
S1 = np.hstack((S1, np.array(kapa -2)))
S1 = np.diag(S1)
t1 = np.kron(vi.T, vi.conj().T) @ S1 @ np.kron(vj.conj(), vj)
t11 = (kapa - 2) * np.linalg.norm(vi*vj)**2
## np.real(t1) == t11

## validate Eq.(60)
N = 3
F = FFTmatrix(N, N)
i = 1
j = 2
vi = F[:, i][:,None]
vj = F[:, j][:,None]
tmp = np.hstack((np.array(1), np.zeros(N)))
c = np.tile(tmp, N-1)
c = np.hstack((c, np.array(1)))[:,None]

S2 = np.tile(np.hstack((c, np.zeros((N**2, N)))), N-1)
S2 = np.hstack((S2, c))

t2 = np.kron(vi.T, vi.conj().T) @ S2 @ np.kron(vj.conj(), vj)
# t2 == 1








































































































































































































