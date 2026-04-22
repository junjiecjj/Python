#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:43:25 2024

@author: jack
"""

import numpy as np

# 产生傅里叶矩阵
def FFTmatrix(row,col):
     mat = np.zeros((row,col),dtype=complex)
     for i in range(row):
          for j in range(col):
               mat[i,j] = 1.0*np.exp(-1j*2.0*np.pi*i*j/row) # / (np.sqrt(row)*1.0)
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

L = len(generateVec)
A = CirculantMatric(generateVec, L)

F = FFTmatrix(L, L)
FH = F.T.conjugate() / (L * 1.0)

F_FT = F@FH
F_FT = np.around(F_FT, decimals = 2)
print(f"傅里叶矩阵自身与其共轭转置的乘积:\n{F_FT} \n")

# FT_F = FH@F
# FT_F = np.around(FT_F,decimals = 2)
# print(f"傅里叶矩阵的共轭转置与自身乘积:\n{FT_F} \n")

FH_A_F = FH@A@F
FH_A_F = np.around(FH_A_F, decimals = 2)
print(f"FH_A_F:\n{FH_A_F}")
print(f"F@X = \n{F@X} ")
print(f"diag[fft(X)] = \n{np.diag(np.fft.fft(X))} \n")

F_A_FH = F@A@FH
F_A_FH = np.around(F_A_FH, decimals = 2)
print(f"F_A_FH:\n{F_A_FH}")
print(f"F.T.conjugate()@X = \n{F.T.conjugate()@X} ")
print(f"L * diag[ifft(X)] = \n{L * np.diag(np.fft.ifft(X))} \n")
# 从上面可以看出，对于循环矩阵A， F^H*A*F 和 F*A*F^H都是对角矩阵，且值一样只是顺序不同。

B = A.T
F_B_FH = F@B@FH
F_B_FH = np.around(F_B_FH, decimals = 2)
print(f"F_B_FH:\n{F_B_FH}")

X1 = B[0,:]
FH_B_F = FH@B@F
FH_B_F = np.around(FH_B_F, decimals = 2)
print(f"FH_B_F:\n{FH_B_F}")
print(f"F@X1 = \n{F@X1} ")
print(f"diag[fft(X1)] = \n{np.diag(np.fft.fft(X1))} \n")
#  有一个现象，那就是  F^H*B*H 和F^H*A*H都是对角矩阵, 值一样，但是顺序不同。

eigvalue, eigvec = np.linalg.eig(A)
print(f"eigvalue = {eigvalue}")





























