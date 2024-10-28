#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:38:47 2024

@author: jack
# https://blog.csdn.net/qq_39227541/article/details/119358373
# https://blog.csdn.net/UncleWa/article/details/123780502
https://blog.csdn.net/weixin_43509834/article/details/140092645

"""


import sys
import numpy as np

# azimuth: 方位角
# elevation: 仰角角
# the angles of arrival (AoA), angles of departure (AoD)

def UPA2ULA_Los(N = 4, Nx = 3, Ny = 4, azi_AP = 0, azi_RIS = -np.pi, ele_RIS = 0):

    SteerVecAP = np.exp(-1j * np.pi * np.arange(N) * np.sin(azi_AP))
    A = np.exp(-1j * np.pi * np.arange(Nx) * np.sin(azi_RIS) * np.cos(ele_RIS))
    B = np.exp(-1j * np.pi * np.arange(Ny) * np.sin(ele_RIS) )
    SteerVecRIS = np.kron(A, B)

    GLos = np.outer(SteerVecAP, SteerVecRIS.conj())  # SteerVecAP.reshape(-1, 1) @ (SteerVecRIS.reshape(1, -1))
    return GLos


def User2RISSteerVec( Nx = 3, Ny = 4, azi = 0, ele = 0 ):
    A = np.exp(-1j * np.pi * np.arange(Nx) * np.sin(azi) * np.cos(ele))
    B = np.exp(-1j * np.pi * np.arange(Ny) * np.sin(ele) )
    SteerVecRIS = np.kron(A, B)
    return SteerVecRIS


def Point2UPASteerVec(K, Ny, Nz, RIS_locate, users_locate):
    L = Ny * Nz
    XY = (users_locate - RIS_locate)
    x = XY[:,0]
    y = XY[:,1]
    azi = -np.arctan2(y, x) + np.pi
    x = np.linalg.norm(XY[:,:2], axis = 1, )
    y = XY[:,2]
    ele = np.arctan2(y, x)

    hrLos = np.zeros((L, K), dtype = complex)
    for k in range(K):
        ax = np.exp(-1j * np.pi * np.arange(Ny) * np.sin(azi[k]) * np.cos(ele[k]))
        ay = np.exp(-1j * np.pi * np.arange(Nz) * np.sin(ele[k]) )
        hrLos[:, k] = np.kron(ax, ay)

    return hrLos


def Point2ULASteerVec(N, K, BS_locate, users_locate):
    XY = (users_locate - BS_locate)[:,:2]
    x = XY[:,0]
    y = XY[:,1]
    theta = -np.arctan2(y, x)
    d = np.arange(N)
    stevec = np.exp(1j * np.pi * np.outer(d, np.sin(theta)))
    return stevec

def Generate_hd(N, K, BS_locate, users_locate, beta_Au, PL_Au, sigmaK2):
    # User to AP/RIS channel
    hdLos = Point2ULASteerVec(N, K, BS_locate, users_locate)
    hdNLos = np.sqrt(1/2) * ( np.random.randn(N, K) + 1j * np.random.randn(N, K))
    h_ds = (np.sqrt(beta_Au/(1+beta_Au)) * hdLos + np.sqrt(1/(1+beta_Au)) * hdNLos )
    h_d = h_ds @ np.diag(np.sqrt(PL_Au.flatten()/sigmaK2))
    return h_d

def Generate_hr(K, Ny, Nz, RIS_locate, users_locate, beta_Iu, PL_Iu, sigmaK2):
    L = Ny * Nz
    hrLos = Point2UPASteerVec(K, Ny, Nz, RIS_locate, users_locate)
    hrNLos = np.sqrt(1/2) * ( np.random.randn(L, K) + 1j * np.random.randn(L, K))
    h_rs = (np.sqrt(beta_Iu/(1+beta_Iu)) * hrLos + np.sqrt(1/(1+beta_Iu)) * hrNLos )
    h_r = h_rs @ np.diag(np.sqrt(PL_Iu.flatten()/sigmaK2))
    return h_r

def Generate_hAI(N, Ny, Nz, RIS_locate, users_locate, beta_AI, PL_AI):
    ## G: RIS to BS channel
    L = Ny * Nz
    GLos = UPA2ULA_Los(N = N, Nx = Ny, Ny = Nz, azi_AP = 0, azi_RIS = -np.pi, ele_RIS = 0)
    GNLos = np.sqrt(1/2) * ( np.random.randn(N, L) + 1j * np.random.randn(N, L))
    h_AI = np.sqrt(PL_AI) * (np.sqrt(beta_AI/(1+beta_AI)) * GLos + np.sqrt(1/(1+beta_AI)) * GNLos )
    return h_AI


#%% Generate Channel, Method 1
# ## G: RIS to BS channel
# GLos = UPA2ULA_Los(N = N, Nx = Nx, Ny = Ny, azi_AP = 0, azi_RIS = -np.pi, ele_RIS = 0)
# GNLos = np.sqrt(1/2) * ( np.random.randn(N, L) + 1j * np.random.randn(N, L))
# h_AI = np.sqrt(PL_AI) * (np.sqrt(beta_AI/(1+beta_AI)) * GLos + np.sqrt(1/(1+beta_AI)) * GNLos )

# # User to AP/RIS channel
# hdLos = Point2ULASteerVec(N, K, BS_locate, users_locate)
# hdNLos = np.sqrt(1/2) * ( np.random.randn(N, K) + 1j * np.random.randn(N, K))
# h_ds = (np.sqrt(beta_Au/(1+beta_Au)) * hdLos + np.sqrt(1/(1+beta_Au)) * hdNLos )
# h_d = h_ds @ np.diag(np.sqrt(PL_Au.flatten()/sigmaK2))

# hrLos = Point2UPASteerVec(K, Nx, Ny, RIS_locate, users_locate)
# hrNLos = np.sqrt(1/2) * ( np.random.randn(L, K) + 1j * np.random.randn(L, K))
# h_rs = (np.sqrt(beta_Iu/(1+beta_Iu)) * hrLos + np.sqrt(1/(1+beta_Iu)) * hrNLos )
# h_r = h_rs @ np.diag(np.sqrt(PL_Iu.flatten()/sigmaK2))

# G = np.zeros([N, L, K], dtype = complex) #  (5, 40, 40)
# for k in range(K):
#     G[:, :, k] = h_AI @ np.diag(h_r[:,k])

# G = np.zeros([N, L, K], dtype = complex) #  (5, 40, 40)
# for k in range(K):
#     G[:, :, k] = h_AI @ np.diag(h_r[:,k])

#%% Generate Channel, Method 2
# G: RIS to BS channel
# ar = np.exp(1j * np.pi * np.arange(N) * np.sin(np.pi * np.random.rand() - np.pi/2))
# at = np.exp(1j * np.pi * np.arange(L) * np.sin(np.pi * np.random.rand() - np.pi/2))
# GLos = np.outer(ar, at.conj())
# GNLos = np.sqrt(1/2) * ( np.random.randn(N, L) + 1j * np.random.randn(N, L))
# h_AI = np.sqrt(PL_AI) * (np.sqrt(beta_AI/(1+beta_AI)) * GLos + np.sqrt(1/(1+beta_AI)) * GNLos )

# # User to AP/RIS channel
# h_d = np.zeros((N, K), dtype = complex)
# h_r = np.zeros((L, K), dtype = complex)
# for k in range(K):
#     hdLos = np.exp(1j * np.pi * np.arange(N) * np.sin(np.pi * np.random.rand() - np.pi/2))
#     hdNLos = np.sqrt(1/2) * ( np.random.randn(N,) + 1j * np.random.randn(N,))
#     h_d[:, k] = (np.sqrt(beta_Au/(1+beta_Au)) * hdLos + np.sqrt(1/(1+beta_Au)) * hdNLos )

#     hrLos = np.exp(1j * np.pi * np.arange(L) * np.sin(np.pi * np.random.rand() - np.pi/2))
#     hrNLos = np.sqrt(1/2) * ( np.random.randn(L,) + 1j * np.random.randn(L,))
#     h_r[:, k] = (np.sqrt(beta_Iu/(1+beta_Iu)) * hrLos + np.sqrt(1/(1+beta_Iu)) * hrNLos )
# h_d = h_d @ np.diag(np.sqrt(PL_Au.flatten()/sigmaK2))
# h_r = h_r @ np.diag(np.sqrt(PL_Iu.flatten()/sigmaK2))

# G = np.zeros([N, L, K], dtype = complex) #  (5, 40, 40)
# for k in range(K):
#     G[:, :, k] = h_AI @ np.diag(h_r[:,k])












































