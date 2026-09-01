#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 22:49:21 2026

@author: jack

Noiseless verification of the complete CP-OFDM and Nyquist pulse-shaping link:
S -> IFFT -> add CP -> upsample -> pulse shaping -> multipath channel
  -> matched filter -> downsample -> remove CP -> FFT -> equalization -> S_hat
"""

import numpy as np
from scipy.linalg import toeplitz


# 实现与MATLAB cconv完全一致的圆卷积
def cconv(a, b, n=None):
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    if n is None:
        n = len(a)+len(b)-1
    linear_conv = np.convolve(a,b,mode='full')
    if n <= 0:
        return np.array([],dtype=complex)
    result = np.zeros(n,dtype=complex)
    if n <= len(linear_conv):
        for k in range(n):
            idx = np.arange(k,len(linear_conv),n)
            result[k] = np.sum(linear_conv[idx])
    else:
        result[:len(linear_conv)] = linear_conv
    return result


def convMatrix(h, N):
    h = np.asarray(h,dtype=complex).reshape(-1)
    col = np.hstack((h,np.zeros(N-1,dtype=complex)))
    row = np.hstack((h[0],np.zeros(N-1,dtype=complex)))
    H = toeplitz(col,row)
    return H


# 产生傅里叶矩阵
def FFTmatrix(L):
    mat = np.zeros((L,L),dtype=complex)
    ll = np.arange(L)
    for i in range(L):
        mat[i,:] = np.exp(-1j*2*np.pi*i*ll/L)/np.sqrt(L)
    return mat


def AcpMat(N, Ncp):
    Acp = np.block([
        [np.zeros((Ncp,N-Ncp)),np.eye(Ncp)],
        [np.eye(N)]
    ])
    return Acp


def ScpMat(N, Ncp, Leq):
    Scp = np.block([
        [np.zeros((N,Ncp)),np.eye(N),np.zeros((N,Leq-1))]
    ])
    return Scp


def upsamplingMatrix(M, Q):
    GammaQ = np.zeros((Q*M,M))
    for m in range(M):
        GammaQ[m*Q,m] = 1
    return GammaQ


def downsamplingMatrix(outputLength, inputLength, Q, n0):
    DQn0 = np.zeros((outputLength,inputLength))
    sampleIndex = n0+Q*np.arange(outputLength)
    if sampleIndex[-1] >= inputLength:
        raise ValueError('The requested downsampling position exceeds the input length.')
    DQn0[np.arange(outputLength),sampleIndex] = 1
    return DQn0


def srrcFunction(beta, Q, span):
    Tsym = 1
    t = np.arange(-span/2,span/2+0.5/Q,1/Q)
    with np.errstate(divide='ignore',invalid='ignore'):
        A = np.sin(np.pi*t*(1-beta)/Tsym)+4*beta*t/Tsym*np.cos(np.pi*t*(1+beta)/Tsym)
        B = np.pi*t/Tsym*(1-(4*beta*t/Tsym)**2)
        p = A/(np.sqrt(Tsym)*B)
    p[np.isnan(p)] = (1+beta*(4/np.pi-1))/np.sqrt(Tsym)
    p[np.isinf(p)] = beta/np.sqrt(2*Tsym)*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta)))
    filtDelay = (len(p)-1)/2
    p = p/np.sqrt(np.sum(np.abs(p)**2))
    return p,t,filtDelay


np.set_printoptions(precision=6,suppress=True)
np.random.seed(42)

# Parameters
N = 16
Q = 4
beta = 0.35
span = 4
L = 3

# Random frequency-domain symbols
symbolIndex = np.random.randint(0,4,N)
S = np.exp(1j*np.pi/2*symbolIndex)

# OFDM modulation basis U=F^H
F = FFTmatrix(N)
FH = F.conj().T
U = FH
s = U@S

# Transmit pulse and receive matched filter
p_t,t,filtDelay = srrcFunction(beta,Q,span)
p_r = np.conj(p_t[::-1])
Lp = p_t.size
Lr = p_r.size

# Physical complex-valued frequency-selective channel
h = np.random.randn(L)+1j*np.random.randn(L)
h = h/np.linalg.norm(h)

# High-rate equivalent impulse response c=p_r*h*p_t
c = np.convolve(np.convolve(p_t,h),p_r)
Lc = c.size

# Use the first sample of the causal polyphase component as the sampling origin
n0 = 2
h_eq = c[n0::Q]
Leq = h_eq.size

# The CP must cover the memory of the symbol-rate equivalent channel
Ncp = Leq-1
if Ncp > N:
    raise ValueError('Ncp must not exceed N. Increase N or shorten the pulse/channel memory.')
M = N+Ncp

# IFFT -> add CP
Acp = AcpMat(N,Ncp)
x_cp = Acp@s

# Q-fold upsampling
GammaQ = upsamplingMatrix(M,Q)
x_up = GammaQ@x_cp

# Transmit pulse shaping
Pt = convMatrix(p_t,Q*M)
x_t = Pt@x_up

# Physical random multipath channel
H = convMatrix(h,x_t.size)
r = H@x_t

# Receive matched filtering
Pr = convMatrix(p_r,r.size)
z = Pr@r

# Verify p_r*h*p_t=c and Pr@H@Pt=T(c;QM)
C = convMatrix(c,Q*M)
combinedConvolutionMatrixError = np.linalg.norm(Pr@H@Pt-C,'fro')
combinedWaveformError = np.linalg.norm(z - C @ x_up)

# Timing and Q-fold downsampling
outputLength = M + Leq - 1
DQn0 = downsamplingMatrix(outputLength, z.size, Q, n0)
r_d = DQn0@z

# Verify H_eq,lin=D*T(c;QM)*Gamma=T(h_eq;M)
H_eq_lin_chain = DQn0 @ C @ GammaQ
H_eq_lin = convMatrix(h_eq, M)
equivalentChannelMatrixError = np.linalg.norm(H_eq_lin_chain - H_eq_lin,'fro')
downsamplingOutputError = np.linalg.norm(r_d - H_eq_lin @ x_cp)

# Remove CP and retain N useful symbol-rate samples
Scp = ScpMat(N, Ncp, Leq)
y = Scp @ r_d

# Verify that CP converts the symbol-rate linear channel into a circular channel
H_eq_cir = Scp @ H_eq_lin @ Acp
y_equivalent = H_eq_cir @ s
y_cconv = cconv(h_eq, s, N)
cpCircularizationError = np.linalg.norm(y - y_equivalent)
circularConvolutionError = np.linalg.norm(y - y_cconv)

# FFT diagonalization and one-tap frequency-domain equalization
Diag = F@H_eq_cir@FH
offDiagonalPart = Diag - np.diag(np.diag(Diag))
diagonalizationError = np.linalg.norm(offDiagonalPart, 'fro')
Y = F@y
S_hat = Y/np.diag(Diag)
symbolRecoveryError = np.linalg.norm(S_hat-S)

print(f'N = {N}, Q = {Q}, L = {L}, Lp = {Lp}, Lr = {Lr}')
print(f'Lc = {Lc}, n0 = {n0}, Leq = {Leq}, Ncp = {Ncp}, M = {M}')
print(f'\nh =\n{h}')
print(f'\nh_eq =\n{h_eq}')
print(f'\n||Pr@H@Pt-C||_F                         = {combinedConvolutionMatrixError:.3e}')
print(f'||z-C@x_up||_2                           = {combinedWaveformError:.3e}')
print(f'||D@C@Gamma-T(h_eq;M)||_F                = {equivalentChannelMatrixError:.3e}')
print(f'||r_d-T(h_eq;M)@x_cp||_2                 = {downsamplingOutputError:.3e}')
print(f'||y-H_eq_cir@s||_2                       = {cpCircularizationError:.3e}')
print(f'||y-cconv(h_eq,s,N)||_2                  = {circularConvolutionError:.3e}')
print(f'||offdiag(F@H_eq_cir@F^H)||_F            = {diagonalizationError:.3e}')
print(f'||S_hat-S||_2                            = {symbolRecoveryError:.3e}')

assert np.allclose(Pr@H@Pt,C,atol=1e-12)
assert np.allclose(z,C@x_up,atol=1e-12)
assert np.allclose(H_eq_lin_chain,H_eq_lin,atol=1e-12)
assert np.allclose(r_d,H_eq_lin@x_cp,atol=1e-12)
assert np.allclose(y,H_eq_cir@s,atol=1e-12)
assert np.allclose(y,y_cconv,atol=1e-12)
assert np.allclose(Diag,np.diag(np.diag(Diag)),atol=1e-12)
assert np.allclose(S_hat,S,atol=1e-10)

print('\nVerification passed: the complete noiseless CP-OFDM and Nyquist link recovers S exactly.')
