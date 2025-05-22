#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:50:44 2025

@author: jack

Implementation of Mulyi-user point-to-point MISO simulation based on paper:
"Sohrabi, F., & Yu, W. (2016). Hybrid Digital and Analog Beamforming Design for Large-Scale Antenna Arrays"
Implemented algorithms are 'Algorithm 3' from the paper.

"""

from typing import Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
# from scipy import linalg as spl

#%%
def channelGen(N, M,  K, L, d = 0.5):
    """
    Advanced environment generator with configurable physical variables.
    Parameters
        N : int
            Number of BS antennas.
        M : int
            Number of receiver antennas.
        K: int
            Number of users.
        L : int
            Number of scatterers.
        d : float
            Antenna spacing distance. The default is 0.5.
    Returns:
        H : numpy.ndarray
            Generated environment (channel).
    """
    pi = np.pi
    H = np.zeros((K, M, N), dtype = np.complex128)
    for k in range(K):
        ang_t = (2*np.random.rand(L) - 1) * pi
        ang_r = (2*np.random.rand(L) - 1) * pi
        alpha = (np.random.randn(L) + 1j*np.random.randn(L))/np.sqrt(2.0)
        Hk = np.zeros((M, N), dtype = np.complex128)
        for l in range(L):
            at = np.sqrt(1/N) * np.exp(1j * pi * np.arange(N) * np.sin(ang_t[l]))
            ar = np.sqrt(1/M) * np.exp(1j * pi * np.arange(M) * np.sin(ang_r[l]))[:,None]
            tmp = alpha[l] * np.outer(ar, at)
            Hk += tmp
        H[k,:,:] = Hk
    return (np.sqrt(M*N/L) * H)

#%%
def updateVRF(N, Nrf, Ht, VRF, epsilon = 0.001):
    pi = np.pi
    fVrf_old = N*np.trace(scipy.linalg.pinv(Ht @ VRF @ VRF.conj().T @ Ht.conj().T))
    diff = 1

    it = 0
    fVrf_new = 0
    while diff > epsilon and it < 100:
        it += 1
        for j in range(Nrf):
            VRFj = np.delete(VRF, j, axis = 1)
            Aj = (Ht @  VRFj @ VRFj.conj().T @ Ht.conj().T).astype(np.complex128)
            AjInv = scipy.linalg.pinv(Aj.astype(np.complex128))
            Bj = (Ht.conj().T @ AjInv @ AjInv @ Ht).astype(np.complex128)
            Dj = (Ht.conj().T @ AjInv @ Ht).astype(np.complex128)
            for i in range(N):
                zetaBij = Bj[i,i] + 2 * np.real(np.sum([np.conj(VRF[m, j])*Bj[m,n]*VRF[n, j] for m in range(N) for n in range(N) if m != i and n != i]))
                zetaDij = Dj[i,i] + 2 * np.real(np.sum([np.conj(VRF[m, j])*Dj[m,n]*VRF[n, j] for m in range(N) for n in range(N) if m != i and n != i]))
                etaBij = Bj[i,:]@VRF[:,j] - Bj[i,i] * VRF[i,j]
                etaDij = Dj[i,:]@VRF[:,j] - Dj[i,i] * VRF[i,j]
                cij = (1 + zetaDij) * etaBij - zetaBij * etaDij
                zij = np.imag(2 * np.conj(etaBij) * etaDij)
                # zij = np.imag(2 * etaBij * etaDij)
                tt = np.arcsin(np.imag(cij)/np.abs(cij))
                if np.real(cij) >= 0:
                    phij = tt
                else:
                    phij = pi - tt
                theta1 =    - phij + np.arcsin(zij/np.abs(cij))
                theta2 = pi - phij - np.arcsin(zij/np.abs(cij))
                vfij1 = np.exp(-1j * theta1)
                vfij2 = np.exp(-1j * theta2)
                f1 = N*np.trace(scipy.linalg.pinv(Aj)) - N * (zetaBij + 2*np.real(np.conj(vfij1)*etaBij))/(1 + zetaDij + 2*np.real(np.conj(vfij1)*etaDij))
                f2 = N*np.trace(scipy.linalg.pinv(Aj)) - N * (zetaBij + 2*np.real(np.conj(vfij2)*etaBij))/(1 + zetaDij + 2*np.real(np.conj(vfij2)*etaDij))
                if f1 <= f2:
                    theta = theta1
                else:
                    theta = theta2
                VRF[i,j] = np.exp(-1j * theta)
        fVrf_new = N*np.trace(scipy.linalg.pinv(Ht @ VRF @ VRF.conj().T @ Ht.conj().T))
        diff = np.abs((fVrf_new - fVrf_old)/fVrf_new)
        print(f"      updateVRF it = {it}, fVRFdiff = {diff}")
        fVrf_old = fVrf_new
    return VRF

#%%
def updateP1(Qt, beta, Ps, K, sigma2):
    lamba = 1
    qkk = np.real(np.diag(Qt))
    while 1:
        initpow = 0
        posi = 0
        for k in range(K):
            tmp = beta[k]/lamba - qkk[k]*sigma2
            if tmp > 0:
                initpow += tmp
                posi +=1
        if np.abs(initpow - Ps)/Ps < 0.001:
            break
        if posi > 0:
            lamba += 0.5 * (initpow - Ps)/posi
        else:
            lamba *= 0.25
        print(f" lamba = {lamba}")
    P = np.identity(K)
    for k in range(K):
        P[k,k] = np.max(beta[k]/lamba - qkk[k]*sigma2, 0)/qkk[k]
    return P, initpow, lamba

import cvxpy as cp
# cp.log 以e为底
def updateP(Qt, beta, Ps, K, sigma2):
    lamba = 1
    qkk = np.real(np.diag(Qt))
    x = cp.Variable(shape = K)
    alpha = cp.Parameter(K, nonneg = True)
    alpha.value = beta
    obj = cp.Maximize(cp.sum(cp.multiply(alpha, cp.log(1 + x/sigma2))))
    constraints = [x >= 0, cp.sum(cp.multiply(x, qkk)) - Ps == 0]
    # Solve
    prob = cp.Problem(obj, constraints)
    prob.solve()
    if(prob.status=='optimal'):
        P = np.identity(K)
        np.fill_diagonal(P, x.value)
        Cap = np.sum(beta * np.log2(1 + np.diag(P)/sigma2))
        curpow = (x.value * qkk).sum()
        print(f"      updateP, {prob.status}, {curpow}/{Ps}, {x.value}, {prob.value*np.log2(np.e)}")

    return P, curpow, 1

## 二分法
def updateP2(Qt, beta, Ps, K, sigma2, epsilon = 1e-8):
    lamba = 1
    qkk = np.real(np.diag(Qt))
    rank = beta/(qkk * sigma2)
    cur_lamba = np.sum(beta)/(Ps + sigma2 * qkk.sum())
    if cur_lamba <= rank.min():
        p = beta/(cur_lamba*qkk) - sigma2
        curpow = np.sum(np.maximum(beta/cur_lamba - qkk*sigma2, 0))
    else:
        gap = 1
        max_lamba = rank.max()
        min_lamba = rank.min()
        while gap > epsilon:
            cur_lamba = (max_lamba + min_lamba)/2.0
            curpow = np.sum(np.maximum(beta/cur_lamba - qkk*sigma2, 0))
            if np.abs((curpow - Ps)/Ps) <= epsilon:
                break
            elif curpow > Ps:
                min_lamba = cur_lamba
            elif curpow < Ps:
                max_lamba = cur_lamba
        p = np.maximum(beta/(qkk*cur_lamba) - sigma2, 0)
    P = np.identity(K)
    np.fill_diagonal(P, p)
    Cap = np.sum(beta * np.log2(1 + np.diag(P)/sigma2))
    print(f"      updateP, {curpow}/{Ps}, {cur_lamba}, {p}, {Cap}")
    return P, curpow, cur_lamba

#%% Design of Hybrid Precoders for MU-MISO systems
def alg3(H, beta, Nrf, Ps, sigma2, epsilon = 1e-3):
    pi = np.pi
    K, M, N = H.shape
    H = H.squeeze()
    diffCap = 1
    lastCap = -1
    tmp = np.random.rand(N, Nrf) * 2 * pi
    VRF = np.exp(1j * tmp).astype(np.complex128)
    P = np.identity(K) * Ps/K
    Ht = scipy.linalg.sqrtm(np.linalg.pinv(P.astype(np.complex128))) @ H
    it = 0
    while diffCap > epsilon and it < 2:
        it += 1
        VRF = updateVRF(N, Nrf, Ht, VRF)
        # 生成功率分配矩阵
        VDt = VRF.conj().T @ H.conj().T @ scipy.linalg.inv(H @ VRF @ VRF.conj().T @ H.conj().T)
        Qt = VDt.conj().T @ VRF.conj().T @ VRF @ VDt
        P, sumP, lamba = updateP2(Qt, beta, Ps, K, sigma2)
        Ht = scipy.linalg.sqrtm(np.linalg.pinv(P)) @ H
        Cap = np.sum(beta * np.log2(1 + np.diag(P)/sigma2))
        diffCap = np.abs((Cap-lastCap)/Cap)
        print(f"    Cap it = {it}, CapDiff = {diffCap}/{Cap}")
        lastCap = Cap
    return Cap

#%% Simulation/testing
np.random.seed(42)
# num of BS antennas
N = 64
# num of users
K = 8
# num of receiver antennas
M = 1
# num of data streams per user
d = 1
# num of data streams
Ns = K * d
Nrf = 9
# num of scatterers
L = 15
beta = [1] * K

# stopping (convergence) condition
# epsilon = 1e-4
# These variables must comply with these invariants: Ns <= Ntft <= N, d <= Nrfr <= M
sigma2 = K
# num of iterations for each dB step
num_iters = 1

SNR_dBs = np.arange(-10, 11)
# Generate and average spectral efficiency data for a range of SNR ratios.
arr = np.zeros(SNR_dBs.size, dtype = np.float64 )
for i, snr in enumerate(SNR_dBs):
    Ps = 10**(snr / 10) * sigma2
    print(f"SNR it = {i}/{SNR_dBs.size}")
    for _ in range(num_iters):
        print(f" repead it = {_}/{num_iters}")
        # generated environment - advanced generation
        H = channelGen(N, M, K, L)
        a2 = alg3(H, beta, Nrf, Ps, sigma2,)
        arr[i] += a2
    arr[i] /= num_iters
    print(f"SNR = {snr}(dB)--> {arr[i]}")

#%%
colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (8, 6))
axs.plot(SNR_dBs, arr, color = colors[0], marker = 'o',linestyle='--', lw = 2, label = 'Multi-User Massive MISO',)
axs.set_xlabel("SNR(dB)", fontsize = 16)
axs.set_ylabel("Spectral Efficiency(bits/s/Hz)", fontsize = 16)
plt.grid()
axs.legend(fontsize = 16)
plt.show()
plt.close('all')

# %%

# %%

# %%

# %%

# M = 3
# N = 4
# A = np.arange(M*N).reshape(N, M)
# B = np.arange(N*N).reshape(N, N)
# i = 1
# j = 2

# [A[m, j]*B[m,n]*A[n,j] for m in range(N) for n in range(N) if m != i and n != i]

























































