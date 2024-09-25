


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:28:33 2024

@author: jack
"""



import sys
import numpy as np
# import scipy
import cvxpy as cp
import matplotlib.pyplot as plt
# import math
# import matplotlib
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

from Tools import UPA2ULA_Los


from DC_Solver import DC_RIS
from DC_Solver1 import DC_RIS1
from SCA_solver import SCA_RIS
from DC_wo_RIS import DC_woRIS
from Utility import set_random_seed, set_printoption
set_random_seed(1)

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"




C0 = -30                             # dB
C0 = 10**(C0/10.0)                   # 参考距离的路损
d0 = 1

N = 6  # Ap antenna
K = 40 # User number
Nx = 5
Ny = 4
L = Nx * Ny  # RIS antenna

## path loss exponents
alpha_Au = 3.5
alpha_AI = 2.2
alpha_Iu = 2.8

## Rician factor
beta_Au = 0   # dB
beta_AI = 3   # dB
beta_Iu = 3   # dB
beta_Au = 10**(beta_Au/10)
beta_AI = 10**(beta_AI/10)
beta_Iu = 10**(beta_Iu/10)

## AP & RIS location
BS_locate = np.array([[-50, 0, 10]])
RIS_locate = np.array([[0, 0, 10]])

## User Location, Case I
# users_locate_x1 = np.random.rand(int(K/2), 1) * (-20)
# users_locate_x2 = np.random.rand(int(K/2), 1) * 100 + 20
# users_locate_x = np.vstack((users_locate_x1, users_locate_x2))
# users_locate_y = np.random.rand(K, 1) * 20 - 10
# users_locate_z = np.zeros((K, 1))
# users_locate = np.hstack((users_locate_x, users_locate_y, users_locate_z))

## User Location, Case II
users_locate_x = np.random.rand(K, 1) * 20
users_locate_y = np.random.rand(K, 1) * 20 - 10
users_locate_z = np.zeros((K, 1))
users_locate = np.hstack((users_locate_x, users_locate_y, users_locate_z))


## Distance
d_Au = pairwise_distances(users_locate, BS_locate, metric = 'euclidean',)
d_Iu = pairwise_distances(users_locate, RIS_locate, metric = 'euclidean',)
d_AI = pairwise_distances(BS_locate, RIS_locate, metric = 'euclidean',)


## generate path-loss
PL_Au = C0 * (d_Au/d0)**(-alpha_Au)
PL_Iu = C0 * (d_Iu/d0)**(-alpha_Iu)
PL_AI = C0 * (d_AI/d0)**(-alpha_AI)

## User to AP channel, Rayleigh fading

#%% Generate Channel, Method 1
## G: RIS to BS channel

# GLos = UPA2ULA_Los(N = N, Nx = Nx, Ny = Ny, azi_AP = 0, azi_RIS = -np.pi, ele_RIS = 0)
# GNLos = np.sqrt(1/2) * ( np.random.randn(N, L) + 1j * np.random.randn(N, L))
# H_RA = np.sqrt(PL_AI) * (np.sqrt(beta_AI/(1+beta_AI)) * GLos + np.sqrt(1/(1+beta_AI)) * GNLos )

# h_d  = np.sqrt(1/2) * (np.random.randn(N, K) + 1j * np.random.randn(N, K))
# h_d = h_d @ np.diag(np.sqrt(PL_Au.flatten()))

#%% Generate Channel, Method 2

ar = np.exp(1j * np.pi * np.arange(N) * np.sin(np.pi * np.random.rand() - np.pi/2))
at = np.exp(1j * np.pi * np.arange(L) * np.sin(np.pi * np.random.rand() - np.pi/2))
GLos = np.outer(ar, at.conj())
GNLos = np.sqrt(1/2) * ( np.random.randn(N, L) + 1j * np.random.randn(N, L))
h_AI = np.sqrt(PL_AI) * (np.sqrt(beta_AI/(1+beta_AI)) * GLos + np.sqrt(1/(1+beta_AI)) * GNLos )

h_d = np.zeros((N, K), dtype = complex)
h_r = np.zeros((L, K), dtype = complex)
for k in range(K):
    Los = np.exp(1j * np.pi * np.arange(N) * np.sin(np.pi * np.random.rand() - np.pi/2))
    NLos = np.sqrt(1/2) * ( np.random.randn(N,) + 1j * np.random.randn(N,))
    h_d[:, k] = np.sqrt(PL_Au[k]) * (np.sqrt(beta_Au/(1+beta_Au)) * Los + np.sqrt(1/(1+beta_Au)) * NLos )

    Los = np.exp(1j * np.pi * np.arange(L) * np.sin(np.pi * np.random.rand() - np.pi/2))
    NLos = np.sqrt(1/2) * ( np.random.randn(L,) + 1j * np.random.randn(L,))
    h_r[:, k] = np.sqrt(PL_Iu[k]) * (np.sqrt(beta_Iu/(1+beta_Iu)) * Los + np.sqrt(1/(1+beta_Iu)) * NLos )

G = np.zeros([N, L, K], dtype = complex) #  (5, 40, 40)
for k in range(K):
    G[:, :, k] = h_AI @ np.diag(h_r[:,k])


#%%

SNR = 80 # dB, P0/sigma^2 = SNR
SNR = 10**(SNR/10)


rho = 5
epsilon = 1e-3
epsilon_dc = 1e-8
verbose = 2
maxiter = 50
iter_num = 50


## Solver
# f_DC, theta_DC, MSE_DC = DC_RIS(N, L, K, h_d, G, epsilon, epsilon_dc, SNR, maxiter, iter_num, rho, verbose, )
# print(f"MSE_DC = {MSE_DC}, ||f_DC||_2 = {np.linalg.norm(f_DC)}, |theta_DC| = f{np.abs(theta_DC)}")

# f_DC1, theta_DC1, MSE_DC1 = DC_RIS1(N, L, K, h_d, G, epsilon, epsilon_dc, SNR, maxiter, iter_num, rho, verbose)
# print(f"MSE = {MSE_DC1}, \n||f_DC1||_2 = {np.linalg.norm(f_DC1)}, \n|theta_DC1| = {np.abs(theta_DC1)}")

Imax = 10000000000
tau = 2
threshold = 1e-9
f_sca, theta_sca, MSE_sca = SCA_RIS(N, L, K, h_d, G, threshold, SNR, Imax, tau, verbose, RISON = 1)
print(f"MSE = {MSE_sca[-1]}, ||f||_2 = {np.linalg.norm(f_sca)}, |theta| = {np.abs(theta_sca)}")
# MSE_log[t] = MSE

# DC without RIS
# f_woRIS, MSE_wo = DC_woRIS(N, L, K, h_d, G, epsilon, epsilon_dc, SNR, maxiter, iter_num, rho, verbose, )
# print(f"MSE = {MSE_wo[-1]}, ||f||_2 = {np.linalg.norm(f_woRIS)}, ")


P0 = 30 # dBm
P0 = 10**(P0/10.0)/1000

sigmaK2 = -50                        # dBm
sigmaK2 = 10**(sigmaK2/10.0)/1000    # 噪声功率

SNR = 10*np.log10(P0/sigmaK2)











































