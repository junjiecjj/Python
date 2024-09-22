#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 22:29:12 2024

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

from Solver import DC_F, DC_theta
from Utility import set_random_seed, set_printoption
set_random_seed(99999)


filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

BS_locate = np.array([[0, 0, 25]])
RIS_locate = np.array([[50, 50, 40]])

T0 = -30  # dB
T0 = 10**(T0/10)
d0 = 1
alpha_UA = 3.5  # AP 和 User 之间的衰落因子
alpha_UR = 2.8  # RIS 和 User 之间的衰落因子
alpha_RA = 2.2  # RIS 和 AP 之间的衰落因子


K = 16 # users
M = 30 # RIS elements
N = 20 # AP antennas
rho = 5
epsion = 1e-3
epsion_dc = 1e-8

users_locate_x = np.random.rand(K, 1) * 100 - 50
users_locate_y = np.random.rand(K, 1) * 100 + 50
users_locate_z = np.zeros((K, 1))
users_locate = np.hstack((users_locate_x, users_locate_y, users_locate_z))

d_UA = pairwise_distances(users_locate, BS_locate, metric = 'euclidean',)
d_UR = pairwise_distances(users_locate, RIS_locate, metric = 'euclidean',)
d_RA = pairwise_distances(BS_locate, RIS_locate, metric = 'euclidean',)

PL_UA = T0 * (d_UA/d0) **(-alpha_UA)
PL_UR = T0 * (d_UR/d0) **(-alpha_UR)
PL_RA = T0 * (d_RA/d0) **(-alpha_RA)

h_d  = np.sqrt(1/2) * (np.random.randn(N, K) + 1j*np.random.randn(N, K))
h_d = h_d @ np.diag(PL_UA.flatten())

H_UR = np.sqrt(1/2) * (np.random.randn(M, K) + 1j*np.random.randn(M, K))
H_UR = H_UR @ np.diag(PL_UR.flatten())

H_RA = np.sqrt(1/2) * (np.random.randn(N, M) + 1j*np.random.randn(N, M))

G = np.zeros([N, M, K], dtype = complex) #  (5, 40, 40)
for j in range(K):
    G[:, :, j] = H_RA @ np.diag(H_UR[:,j])































































































