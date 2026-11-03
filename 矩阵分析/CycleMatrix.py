#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:43:25 2024

@author: jack
"""

import numpy as np

# 产生傅里叶矩阵
def FFTmatrix(L, ):
     mat = np.zeros((L, L), dtype = complex)
     lst = np.arange(L)
     for i in range(L):
         mat[i,:] = 1.0*np.exp(-1j*2.0*np.pi*i*lst / L) / (np.sqrt(L)*1.0)
     return mat

# 以行生成 循环矩阵
def CirculantMatricRow(gen, row):
     if type(gen) == list:
          col = len(gen)
     elif type(gen) == np.ndarray:
          col = gen.size
     row = col
     mat = np.zeros((row, col), np.complex128)
     mat[0,:] = gen
     for i in range(1,row):
          mat[i,:] = np.roll(gen,i)
     return mat

# 以行生成 循环矩阵
def CirculantMatricCol(gen, row):
     if type(gen) == list:
          col = len(gen)
     elif type(gen) == np.ndarray:
          col = gen.size
     row = col
     mat = np.zeros((row, col), np.complex128)
     mat[:,0] = gen
     for i in range(1,row):
          mat[:,i] = np.roll(gen,i)
     return mat


# 左下角N-k，右上角k, 维度N
def Jp_Nk_right(N, k):
    Jtilde = np.zeros((N, N))
    Jtilde[k:N, 0:N - k] = np.eye(N - k)
    Jtilde[0:k, N - k:N] = np.eye(k)
    return Jtilde

generateVec =  [1+1j, 2+2j, 3+3j, 4+1j ]
# generateVec =  [1 , 2  , 3 , 4  ]

X = np.array(generateVec)

L = len(generateVec)
C = CirculantMatricCol(generateVec, L)

F = FFTmatrix(L)
FH = F.T.conjugate() # /(L*1.0)

F_FT = F@FH
F_FT = np.around(F_FT,decimals = 2)
print(f"傅里叶矩阵自身与其共轭转置的乘积:\n{F_FT} \n")


F_C_FH = F@C@FH
F_C_FH = np.around(F_C_FH, decimals = 2)
print(f"F_C_FH:\n{F_C_FH}")
print(f"sqrt(L)*F@X = \n{np.sqrt(L) * F@X} ")  # Eq.(11)

print(f"sqrt(L)*FH@C[0,:] = \n{np.sqrt(L) * FH@C[0,:]} ") # Eq.(14)

print(f"sqrt(L)*F@C[0,:] = \n{np.sqrt(L) * F@C[0,:]} ") # Eq.(17)

FH_C_F = FH@C@F
FH_C_F = np.around(FH_C_F, decimals = 2)
print(f"FH_C_F:\n{FH_C_F}")
print(f"F@X = \n{F@X} ")

eigvalue, eigvec = np.linalg.eig(C)
print(f"eigvalue = {eigvalue}")

#%%

# 验证截图公式(20)：Pk = FH @ D @ F
N = 5
k = 2

F = FFTmatrix(N)
FH = F.T.conjugate() # /(L*1.0)

n = np.arange(1, N + 1)
Pk = Jp_Nk_right(N, k)
D = np.diag(np.exp(-1j * 2.0 * np.pi * (n - 1) * k / N))
Pk_hat = FH @ D @ F

print(f"公式(20)左侧 Pk:\n{Pk}\n")
print(f"公式(20)右侧 FH@D@F:\n{np.around(Pk_hat, decimals=10)}\n")
print(f"理论特征值:\n{np.diag(D)}\n")
print(f"F@Pk@FH 的对角线:\n{np.diag(F @ Pk @ FH)}\n")
print(f"公式(20)两侧最大绝对误差: {np.max(np.abs(Pk - Pk_hat)):.3e}")
print(f"公式(20)是否成立: {np.allclose(Pk, Pk_hat, rtol=0, atol=1e-12)}")


























