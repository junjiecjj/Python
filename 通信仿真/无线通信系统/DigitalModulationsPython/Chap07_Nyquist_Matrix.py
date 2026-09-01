#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:11:35 2026

@author: jack

Noiseless Nyquist transmission verified entirely by matrix operations.

"""

import numpy as np
from scipy.linalg import toeplitz
np.random.seed(42)

def upsamplingMatrix(N, L):
    QL = np.zeros((L*N, N))
    for n in range(N):
        QL[n*L, n] = 1
    return QL

def convolutionMatrix(h, N):  #
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

    # from scipy.linalg import toeplitz
    H = toeplitz(col, row)
    return H

def downsamplingMatrix(outputLength, inputLength, L, firstSampleIndex):
    DL = np.zeros((outputLength, inputLength))
    sampleIndex = firstSampleIndex+L*np.arange(outputLength)
    if sampleIndex[-1] >= inputLength:
        raise ValueError("The requested downsampling position exceeds the input length.")
    DL[np.arange(outputLength), sampleIndex] = 1
    return DL

def equivalentSymbolRateMatrix(c, N, L, n0):
    """
    根据 G[m,i] = c[n0+(m-i)L] 直接构造符号率端到端矩阵 G。

    Parameters
    ----------
    c : ndarray
        发射脉冲、物理信道和接收滤波器构成的总等效冲激响应：
        c = p_t * h * p_r。
    N : int
        输入符号数量，G的维度为N×N。
    L : int
        上采样倍数。
    n0 : int
        接收端的第一个抽样位置，即符号定时位置。

    Returns
    -------
    G : ndarray
        N×N符号率端到端矩阵。
    """
    c = np.asarray(c).reshape(-1)
    G = np.zeros((N, N), dtype=np.result_type(c.dtype, np.complex128))

    for m in range(N):
        for i in range(N):
            cIndex = n0+(m-i)*L
            if 0 <= cIndex < len(c):
                G[m,i] = c[cIndex]

    return G

# Program 7.8: test SRRCPulse.m: Square-root raised-cosine pulse characteristics
def srrcFunction(beta, L, span):
    # Function for generating rectangular pulse for the given inputs
    # L - oversampling factor (number of samples per symbol)
    # span - filter span in symbol durations
    # Returns the output pulse p(t) that spans the discrete-time base -span:1/L:span. Also returns the filter delay.
    Tsym = 1
    t = np.arange(-span/2, span/2 + 0.5/L, 1/L)
    A = np.sin(np.pi*t*(1-beta)/Tsym) + 4*beta*t/Tsym * np.cos(np.pi*t*(1+beta)/Tsym)
    B = np.pi*t/Tsym * (1-(4*beta*t/Tsym)**2)
    p = 1/np.sqrt(Tsym) * A/B
    # p[np.argwhere(np.isnan(p))] = 1
    p[np.argwhere(np.isnan(p))] = (1+beta*(4/np.pi-1))/np.sqrt(Tsym); # 这个才是准确的，上面的是书上的，不精确
    p[np.argwhere(np.isinf(p))] = beta/(np.sqrt(2*Tsym)) * ((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
    filtDelay = (len(p)-1)/2
    p = p / np.sqrt(np.sum(np.power(p, 2))) # both Add and Delete this line is OK.
    return p, t, filtDelay

# np.set_printoptions(precision=6, suppress=True)

N = 6
L = 4
beta = 0.35
span = 6

# A deterministic complex-symbol vector is used so that the result is reproducible.
s = np.random.randn(N) + 1j * np.random.randn(N)

# This finite rectangular pulse is exactly root-Nyquist under matched filtering:
# its autocorrelation is zero at every nonzero integer multiple of L samples.
p_t, t, filtDelay = srrcFunction(beta, L, span)
p_r = np.conj(p_t[::-1])

# AWGN channel without noise: h[n] = delta[n].
h = np.array([1], dtype=complex)

# 1) L-fold upsampling: s_up = QL@s.
QL = upsamplingMatrix(N, L)
s_up = QL@s

# 2) Transmit pulse shaping: x_t = Pt@s_up.
Pt = convolutionMatrix(p_t, s_up.size)
x_t = Pt@s_up

# 3) Physical channel: r = H@x_t.
H = convolutionMatrix(h, x_t.size)
r = H@x_t

# 4) Receive matched filtering: z = Pr@r.
Pr = convolutionMatrix(p_r, r.size)
z = Pr@r

# 5) Compensate the total filter delay and downsample by L.
firstSampleIndex = len(p_t)-1
DL = downsamplingMatrix(N, z.size, L, firstSampleIndex)
s_hat = DL@z

C1 = Pr@H@Pt
# 通过完整矩阵链得到符号率端到端矩阵
G = DL@Pr@H@Pt@QL

# 总等效冲激响应
c = np.convolve(np.convolve(p_t, h),p_r)
# 通过卷积矩阵得到符号率端到端矩阵
C2 = convolutionMatrix(c,s_up.size)
G_equivalent = DL@C2@QL

# 根据G[m,i]=c[n0+(m-i)L]直接逐元素构造G
G_direct = equivalentSymbolRateMatrix(c, N, L, firstSampleIndex)

# 三种矩阵分别作用于符号向量
s_hat = G@s
s_hat_equivalent = G_equivalent@s
s_hat_direct = G_direct@s

print(f"s = \n{s}")
print("\ns_hat = {s_hat}")

print("\nG = DL@Pr@H@Pt@QL:")
print(G)

print("\nG_equivalent = DL@C2@QL:")
print(G_equivalent)

print("\nG_direct constructed from c[n0+(m-i)L]:")
print(G_direct)

matrixFactorizationError = np.linalg.norm(G-G_equivalent,"fro")
directConstructionError = np.linalg.norm(G-G_direct,"fro")
equivalentDirectError = np.linalg.norm(G_equivalent-G_direct,"fro")
symbolRecoveryError = np.linalg.norm(s_hat-s)
directRecoveryError = np.linalg.norm(s_hat_direct-s)
equivalRecoveryError = np.linalg.norm(s_hat_equivalent-s)

print(f"\n||G-G_equivalent||_F        = {matrixFactorizationError:.3e}")
print(f"||G-G_direct||_F            = {directConstructionError:.3e}")
print(f"||G_equivalent-G_direct||_F = {equivalentDirectError:.3e}")
print(f"||G@s-s||_2                 = {symbolRecoveryError:.3e}")
print(f"||G_direct@s-s||_2          = {directRecoveryError:.3e}")
print(f"||G_equivalent@s-s||_2          = {equivalRecoveryError:.3e}")

assert np.allclose(G,G_equivalent,atol=1e-12)
assert np.allclose(G,G_direct,atol=1e-12)










