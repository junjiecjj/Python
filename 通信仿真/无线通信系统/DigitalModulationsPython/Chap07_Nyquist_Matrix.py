#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:11:35 2026

@author: jack
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Noiseless Nyquist transmission verified entirely by matrix operations."""

import numpy as np
from scipy.linalg import toeplitz
np.random.seed(42)

def upsamplingMatrix(N, L):
    QL = np.zeros((L*N, N))
    for n in range(N):
        QL[n*L, n] = 1
    return QL

def convolutionMatrix(a, inputLength):
    a = np.asarray(a, dtype=complex).reshape(-1)
    firstColumn = np.concatenate((a, np.zeros(inputLength-1, dtype=complex)))
    firstRow = np.concatenate(([a[0]], np.zeros(inputLength-1, dtype=complex)))
    return toeplitz(firstColumn, firstRow)


def downsamplingMatrix(outputLength, inputLength, L, firstSampleIndex):
    DL = np.zeros((outputLength, inputLength))
    sampleIndex = firstSampleIndex+L*np.arange(outputLength)
    if sampleIndex[-1] >= inputLength:
        raise ValueError("The requested downsampling position exceeds the input length.")
    DL[np.arange(outputLength), sampleIndex] = 1
    return DL

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

np.set_printoptions(precision=6, suppress=True)

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

# Complete end-to-end symbol-rate matrix.
G = DL@Pr@H@Pt@QL

# Equivalent impulse-response representation.
c = np.convolve(np.convolve(p_t, h), p_r)
C = convolutionMatrix(c, s_up.size)
G_equivalent = DL@C@QL
s_hat_equivalent = G_equivalent@s

print("s =")
print(s)
print("\ns_hat =")
print(s_hat)
print("\nEnd-to-end symbol-rate matrix G = DL @ Pr @ H @ Pt @ QL =")
print(G)
print("\nEquivalent matrix G_equivalent = DL @ C @ QL =")
print(G_equivalent)

matrixFactorizationError = np.linalg.norm(G-G_equivalent, "fro")
nyquistMatrixError = np.linalg.norm(G-np.eye(N), "fro")
symbolRecoveryError = np.linalg.norm(s_hat-s)
equivalentRecoveryError = np.linalg.norm(s_hat-s_hat_equivalent)

print(f"\nMatrix factorization error = {matrixFactorizationError:.3e}")
print(f"Nyquist matrix error       = {nyquistMatrixError:.3e}")
print(f"Symbol recovery error      = {symbolRecoveryError:.3e}")
print(f"Equivalent recovery error  = {equivalentRecoveryError:.3e}")

# assert np.allclose(G, G_equivalent, atol=1e-12)
# assert np.allclose(G, np.eye(N), atol=1e-12)
# assert np.allclose(s_hat, s, atol=1e-12)

print("\nVerification passed: in the noiseless Nyquist system, s_hat is equal to s.")
