#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:50:44 2025

@author: jack

Implementation of Multi-user MISO simulation based on paper:
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
            tmp = alpha[l] * np.outer(ar, at.conj())
            Hk += tmp
        H[k,:,:] = Hk
    return (np.sqrt(M*N/L) * H)

#%%
def updateVRF(N, Nrf, Ht, VRF, epsilon = 0.001):
    pi = np.pi
    fVrf_old = np.real(N*np.trace(scipy.linalg.pinv(Ht @ VRF @ VRF.conj().T @ Ht.conj().T)))
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
        fVrf_new = np.real(N*np.trace(scipy.linalg.pinv(Ht @ VRF @ VRF.conj().T @ Ht.conj().T)))
        diff = np.abs((fVrf_new - fVrf_old)/fVrf_new)
        print(f"      updateVRF it = {it}, fVrf_new = {fVrf_new}, fVRFdiff = {diff}")
        fVrf_old = fVrf_new
    return VRF
#%%

import cvxpy as cp
# cp.log 以e为底, updateVRF中it设置为100，
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
        print(f"      updateP, {prob.status}, {curpow}/{Ps}, {x.value}, {prob.value*np.log2(np.e)}/{Cap}")

    return P, curpow, 1

# 这两个updateP的结果几乎一致
def updateP1(Q_tilde, beta, Pt, K, sigma2):
    """
    水注法功率分配函数（使用二分法）
    """
    # 提取Q_tilde的对角线元素（保持原始顺序）
    q_kk = np.diag(np.real(Q_tilde))
    beta = np.real(beta)

    if Pt <= 0:
        p = np.zeros(K)
        lambda_val = np.inf
        P = np.zeros((K, K))
        return P, 0, lambda_val
    # 避免除以0，使用非常小的正数
    lambda_min = 1e-15
    # λ_max: 至少有一个用户有正功率时的最大值, 根据公式：β_k/λ - q_kk*σ² ≥ 0 => λ ≤ β_k/(q_kk*σ²)
    lambda_max = np.max(beta / (q_kk * sigma2))
    # 二分法参数
    max_iter = 10000
    tolerance = 1e-10

    # 二分法搜索
    lambda_val = (lambda_min + lambda_max) / 2  # 初始化lambda为标量
    for iter_num in range(max_iter):
        # 计算当前λ对应的总功率
        total_power = 0
        for k in range(K):
            term = beta[k] / lambda_val - q_kk[k] * sigma2
            if term > 0:
                total_power += term  # 注意：这里计算的是q_kk * p_k
        # 检查收敛条件
        if abs(total_power - Pt) < tolerance:
            break
        # 调整搜索区间
        if total_power < Pt:
            lambda_max = lambda_val  # λ太大，减少功率
        else:
            lambda_min = lambda_val  # λ太小，增加功率
        # 更新lambda
        lambda_val = (lambda_min + lambda_max) / 2
        # 防止lambda过小
        if lambda_val < 1e-15:
            lambda_val = 1e-15
            break
    # 检查是否收敛
    if iter_num == max_iter - 1:
        print('二分法达到最大迭代次数，可能未完全收敛')
    # 计算最终功率分配
    p = np.maximum(beta / lambda_val - q_kk * sigma2, 0) / q_kk
    # 求出P（对角矩阵）
    P = np.diag(p)
    # 验证结果
    curpow = np.sum(p * q_kk)
    Cap = np.sum(beta * np.log2(1 + np.diag(P)/sigma2))
    print(f"      updateP, {curpow}/{Ps}, {lambda_val}, {p}, {Cap}")
    return P, curpow, lambda_val


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
    while diffCap > epsilon and it < 20:
        it += 1
        VRF = updateVRF(N, Nrf, Ht, VRF)
        # 生成功率分配矩阵
        VDt = VRF.conj().T @ H.conj().T @ scipy.linalg.inv(H @ VRF @ VRF.conj().T @ H.conj().T)
        Qt = VDt.conj().T @ VRF.conj().T @ VRF @ VDt
        P, sumP, lamba = updateP1(Qt, beta, Ps, K, sigma2)
        Ht = scipy.linalg.sqrtm(np.linalg.pinv(P)) @ H
        Cap = np.sum(beta * np.log2(1 + np.diag(P)/sigma2))
        diffCap = np.abs((Cap-lastCap)/Cap)
        print(f"    Cap it = {it}, Cap = {Cap}, CapDiff = {diffCap}")
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

























































