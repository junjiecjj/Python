#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:28:48 2026

@author: jack
"""
import numpy as np


# 实现与MATLAB cconv完全一致的圆卷积
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

#
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


# 产生傅里叶矩阵
def FFTmatrix(L, ):
     mat = np.zeros((L, L), dtype = complex)
     ll = np.arange(L)
     for i in range(L):
         mat[i,:] = 1.0*np.exp(-1j*2.0*np.pi*i*ll/L) / (np.sqrt(L)*1.0)
     return mat


def AcpMat(N, Ncp):
    Acp = np.block([
        [np.zeros((Ncp, N-Ncp)), np.eye(Ncp)],
        [np.eye(N)]
        ])
    return Acp

def ScpMat(N, Ncp, Lh):
    Scp = np.block([
        [np.zeros((N, Ncp)), np.eye(N), np.zeros((N, Lh-1))]
        ])
    return Scp

def RcpMat(N, Ncp):
    Rcp = np.block([
        [np.zeros((N, Ncp)), np.eye(N)]
    ])
    return Rcp


# 下面是OFDM中IFFT -> +cp -> H -> -cp -> FFT的等效过程
h = np.array([-0.4878, -1.5351, 0.2355])
S = np.array([-0.0155, 2.5770, 1.9238, -0.0629])

N = S.size
L = h.size
F = FFTmatrix(N)
FH = F.conj().T

U = FH
s = U @ S                     # IFFT

cir_s_h = cconv(h, s, N)      # circular conv

# Hlin = convMatrix(h, N)
# y = Hlin @ s                # linear conv

Ncp = L - 1
Acp = AcpMat(N, Ncp)
s_cp = Acp @ s                # add CP

Hlin_cp = convMatrix(h, s_cp.size)
y_lincp = Hlin_cp @ s_cp                # pass freq selected channel
y_remo_cp = y_lincp[Ncp:Ncp + N]        # receiver, remove cp, == cir_s_h

Scp = ScpMat(N, Ncp, L)
y_remo_cp1 = Scp @ Hlin_cp @ Acp @ U @ S    #  pass freq selected channel + remove cp

Diag = F @ Scp @ Hlin_cp @ Acp @ U      # Eq.(3): F@T(h)@A@FH is diagonal such that the data is parallelly transmitted over different subcarriers, and thus the ISI is avoided.

Heff = Scp @ Hlin_cp @ Acp
print(f"h = {h}\n Heff = \n{Heff}")     # H --> Hcir, 将拓普利兹矩阵变为循环阵, 到这里，从离散信号角度完美的对应OFDM的理论
