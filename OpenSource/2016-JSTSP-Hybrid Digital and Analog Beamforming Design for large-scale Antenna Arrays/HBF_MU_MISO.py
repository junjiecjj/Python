#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:50:44 2025

@author: jack

Implementation of Mulyi-user point-to-point MISO simulation based on paper:
"Sohrabi, F., & Yu, W. (2016). Hybrid Digital and Analog Beamforming Design for Large-Scale Antenna Arrays"
Implemented algorithms are 'Algorithm 3' from the paper.


Comments often refer to specific figures and equations from the paper, as "(x)".

"""

from typing import Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
# from scipy import linalg as spl



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
        Gain : numpy.ndarray
            Complex gain of each path.
        At : numpy.ndarray
            Array response vectors.
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


def updateVRF():
    return

def updateP():
    return



def alg3(H, beta, P, sigma2, epsilon):

    return




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
epsilon = 1e-4
# These variables must comply with these invariants: Ns <= Ntft <= N, d <= Nrfr <= M
sigma2 = 40
# num of iterations for each dB step
num_iters = 100
# range of dB to graph e.g. -10 to 9 (20 steps)
db_range = 10

SNR_dBs = np.arange(-9, 11)
# Generate and average spectral efficiency data for a range of SNR ratios.
arr = np.zeros(SNR_dBs.size, dtype = np.float64 )
for i, snr in enumerate(SNR_dBs):
    P = 10**(snr / 10) * sigma2
    for _ in range(num_iters):
        # generated environment - advanced generation
        H = channelGen(N, M, K, L)

        a2 = alg3(H, beta, P, sigma2, epsilon)
        arr[i] += a2
    arr[i] /= num_iters
    print(f"SNR = {snr}(dB)--> {arr[i]}")

# %%
colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (8, 6))
axs.plot(SNR_dBs, arr, color = colors[0], marker = 'o',linestyle='--', lw = 2, label = 'Multi-User Large-Scale MISO' , )
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



























































