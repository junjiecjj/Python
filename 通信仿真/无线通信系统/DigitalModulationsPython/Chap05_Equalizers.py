#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:21:31 2025

@author: jack

《从微积分到5G》
"""

import numpy as np
import commpy
import scipy

from Modulations import modulator


#%% 验证《从微积分到5G》Chap13.Eq(13.1), Page 247

def genH(h, Nx, Nh):
    H = np.zeros((Nx+Nh-1, Nx),  dtype= complex )
    h = np.pad(h, (0, Nx - 1))
    for j in range(Nx):
        H[:,j] = np.roll(h, j)
    return H

def CutFoldAdd(x, L):
    out = np.zeros(L, dtype = x.dtype)
    if x.size % L == 0:
        Int = x.size//L
        for i in range(Int):
            out += x[i*L:(i+1)*L]
    else:
        pad = L - x.size % L
        Int = x.size//L + 1
        for i in range(Int-1):
            out += x[i*L:(i+1)*L]
        out += np.pad(x[(Int-1)*L:Int*L], (0, pad))
    return out

MOD_TYPE = "qam"
Order = 16
modem, Es, bps = modulator(MOD_TYPE, Order)
Constellation = modem.constellation/np.sqrt(Es)
AvgEnergy = np.mean(np.abs(Constellation)**2)

Nh = 4
Nx = 5
h = np.sqrt(1/2) * (np.random.randn(Nh) + 1j * np.random.randn(Nh))
H = genH(h, Nx, Nh)

d = np.random.randint(Order, size = Nx)
x = Constellation[d]
sigma2 = 0.1
z = np.sqrt(sigma2/2) * (np.random.randn(H.shape[0]) + 1j * np.random.randn(H.shape[0]))

## 线卷积
y = H @ x + z


## 把y, h, x切成Nx的长度然后累加起来
h = np.pad(h, (0, Nx - 1))
h_tilde = CutFoldAdd(h, Nx)
y_tilde = CutFoldAdd(y, Nx)
z_tilde = CutFoldAdd(z, Nx)
H_tilde = scipy.linalg.circulant(h_tilde)

## 圆卷积。 y1 == y_tilde 得到验证, 仔细一想，这是肯定成立的， 因为仅仅只是行之间的线性相加，没有不成立的理由。
y1 = H_tilde @ x + z_tilde































































