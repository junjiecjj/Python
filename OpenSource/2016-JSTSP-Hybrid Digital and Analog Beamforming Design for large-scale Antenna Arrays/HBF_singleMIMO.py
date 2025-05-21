#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:50:44 2025

@author: jack

Implementation of single-user point-to-point MIMO simulation based on paper:
"Sohrabi, F., & Yu, W. (2016). Hybrid Digital and Analog Beamforming Design for Large-Scale Antenna Arrays"
Implemented algorithms are 'Algorithm 1' and 'Algorithm 2' from the paper.
Original code, Damian Filo, 2022

Comments often refer to specific figures and equations from the paper, as "(x)".

"""


from typing import Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
# from scipy import linalg as spl

def alg1(F: np.ndarray, Ns: int, gamma2: float, sigma2: float, epsilon: float = 1e-3) -> np.ndarray:
    """
    The alg1 function is an implementation of 'Algorithm 1' from the paper.
    The algorithm is further implementation of optimization problem used for
    approximating analog precoder matrix Vrf, and the number of rf chains (Nrf) is equal
    to the number of data streams (Ns).

    Parameters
    ----------
    F : numpy.ndarray
        Either F1 or F2 type matrix as explained in the paper near equation (12) and (15).
    Ns : int
        Number of data streams.
    gamma2 : float
        Signal amplitude squared.
    sigma2 : float
        Noise standard deviation squared.
    epsilon : float
        Stopping condition. The default is 1e-3.

    Returns
    -------
    Vrf : numpy.ndarray
        Approximation of analog precoder matrix.
    """
    Nrf = Ns
    # Initialize Vrf
    Vrf = np.ones((F.shape[0], Nrf), dtype=np.complex128)
    last_iter_obj = 0.0
    iter_obj = 0.0
    diff = 1.0
    # Repeat until accuracy of epsilon reached
    while diff >= epsilon:
        for j in range(Nrf):
            # Vrf_j = Vrf
            # Deletion of j-th column
            # Vrf_j[:, j] = 0.0
            Vrf_j = np.delete(Vrf, j, axis = 1)
            # Compute Cj and Gj as per (13)
            Cj = np.identity(Nrf-1) + (gamma2/sigma2) * (Vrf_j.conj().T @ (F @ Vrf_j))
            Gj = (gamma2/sigma2) * F - (gamma2/sigma2)**2 * (F @ Vrf_j @ np.linalg.inv(Cj.astype(np.complex128)) @ Vrf_j.conj().T @ F)
            # Vrf update loop
            for i in range(F.shape[0]):
                eta_ij = 0.0
                # Sum l != i loop
                for l in [x for x in range(F.shape[0]) if x != i]:
                    # print(f"  {Gj[i, l] * Vrf[l, j]}")
                    eta_ij += Gj[i, l] * Vrf[l, j]
                # Value assignment as per (14)
                if eta_ij == 0:
                    Vrf[i, j] = 1
                else:
                    Vrf[i, j] = eta_ij / abs(eta_ij)
        # Save the last result
        last_iter_obj = iter_obj
        # Calculate objective function of (12a)
        iter_obj =  np.log2(np.linalg.det((np.identity(Nrf) + (gamma2/sigma2) * Vrf.conj().T @ F @ Vrf).astype(np.complex128)))
        # Calculate difference of last and current objective function
        diff = abs((iter_obj - last_iter_obj) / iter_obj)
    return Vrf

def alg2(H: np.ndarray, Ns: int, P: float, sigma2: float, epsilon: float = 1e-3) -> float:
    """
    The alg2 function is an implementation of 'Algorithm 2' from the paper.
    The goal of this algorithm is to incorporate the environment and compute receiving signal matrices.
    Using the knowledge gained the algorithm computes the spectral efficiency metric,
    when the number of rf chains (Nrf) is equal to the number of data streams (Ns).

    Parameters
        H : numpy.ndarray
            Environment matrix.
        Ns : int
            Number of data streams.
        P : float
            Broadcast power.
        sigma2 : float
            Noise standard deviation squared.
        epsilon : float
            Stopping condition. The default is 1e-3.

    Returns
        R : float
            Spectral efficiency (bits/s/Hz)
    """
    Nrf = Ns
    gamma = np.sqrt(P / (H.shape[1] * Nrf))

    # Find Vrf using alg1
    F_1 = H.conj().T @ H
    Vrf = alg1(F_1, Ns, gamma**2, sigma2, epsilon)

    # Find Ue and GAMMAe matrices (11)
    Heff = H @ Vrf
    Q = Vrf.conj().T @ Vrf
    # Right singular vectors, TypeError: array type complex256 is unsupported in linalg
    u, s, Ue = np.linalg.svd((Heff @ (scipy.linalg.sqrtm(np.linalg.inv(Q)))).astype(np.complex128))
    # u, s, Ue = np.linalg.svd((Heff @ (np.linalg.inv(Q)**(1/2))).astype(np.complex128))

    # Diagonal matrix of allocated powers to each stream
    GAMMAe = np.identity(Q.shape[0]) * (P/Nrf)**0.5

    # Computing digital precoder matrix (11)
    Vd = (scipy.linalg.sqrtm(np.linalg.inv(Q)) @ Ue @ GAMMAe).astype(np.complex128)
    # Vd = (np.linalg.inv(Q)**(1/2) @ Ue @ GAMMAe).astype(np.complex128)

    # Hybrid precoder matrix (8)
    Vt = Vrf @ Vd

    # Compute analog combiner matrix of receiver (15)
    F_2 = (H @ Vt @ Vt.conj().T @ H.conj().T).astype(np.complex128)
    Wrf = alg1(F_2, Ns, 1/H.shape[0], sigma2, epsilon)

    # Compute the digital combiner matrix of receiver (17)
    J = Wrf.conj().T @ H @ Vrf @ Vd @ Vd.conj().T @ Vrf.conj().T @ H.conj().T @ Wrf  + sigma2 * (Wrf.conj().T @ Wrf)
    Wd = np.linalg.inv(J.astype(np.complex128)) @ (Wrf.conj().T @ (H @ (Vrf @ Vd)))

    # Hybrid combiner matrix (8)
    Wt = Wrf @ Wd

    # Compute the spectral efficiency metric (4)
    R = np.log2(np.linalg.det((np.identity(H.shape[0]) + (1/sigma2) * Wt @ np.linalg.inv((Wt.conj().T @ Wt).astype(np.complex128)) @ Wt.conj().T @ H @ Vt @ Vt.conj().T @ H.conj().T)).astype(np.complex128))

    return R

def channelGen(N, M, L, d = 0.5):
    """
    Advanced environment generator with configurable physical variables.
    Parameters
        N : int
            Number of BS antennas.
        M : int
            Number of receiver antennas.
        L : int
            Number of scatterers.
        d : float
            Antenna spacing distance. The default is 0.5.
    Returns:
        H : numpy.ndarray
            Generated environment (channel).
        Gain : numpy.ndarray
            Complex gain of each path.
        At : numpy.ndarray
            Array response vectors.
    """
    pi = np.pi
    ang_t = (2*np.random.rand(L) - 1) * pi
    ang_r = (2*np.random.rand(L) - 1) * pi
    H = np.zeros((M, N), dtype = np.complex128)
    alpha = (np.random.randn(L ) + 1j*np.random.randn(L ))/np.sqrt(2.0)
    for l in range(L):
        at = np.sqrt(1/N) * np.exp(1j * pi * np.arange(N)* np.sin(ang_t[l]))
        ar = np.sqrt(1/M) * np.exp(1j * pi * np.arange(M)* np.sin(ang_r[l]))[:,None]
        tmp = alpha[l] * np.outer(ar, at.conj())
        H += tmp
    return (np.sqrt(M*N/L) * H)

#%% Simulation/testing
np.random.seed(42)
# num of BS antennas
N = 64
# num of receiver antennas
M = 16
# num of users
K = 1
# num of data streams per user
d = 6
# num of data streams
Ns = K * d
# num of scatterers
L = 15
# stopping (convergence) condition
epsilon = 1e-4
# These variables must comply with these invariants: Ns <= Ntft <= N, d <= Nrfr <= M
sigma2 = 40
# num of iterations for each dB step
num_iters = 1000
# range of dB to graph e.g. -10 to 9 (20 steps)
db_range = 10

SNR_dBs = np.arange(-9, 11)
# Generate and average spectral efficiency data for a range of SNR ratios.
arr = np.zeros(SNR_dBs.size, dtype = np.float64 )
for i, snr in enumerate(SNR_dBs):
    P = 10**(snr / 10) * sigma2
    for _ in range(num_iters):
        # generated environment - advanced generation
        H = channelGen(N, M, L)

        a2 = alg2(H, Ns, P, sigma2, epsilon)
        arr[i] += a2
    arr[i] /= num_iters
    print(f"SNR = {snr}(dB)--> {arr[i]}")

# %%
colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (8, 6))
axs.plot(SNR_dBs, arr, color = colors[0], marker = 'o',linestyle='--', lw = 2, label = 'Single-User Large-Scale MIMO' , )
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














































