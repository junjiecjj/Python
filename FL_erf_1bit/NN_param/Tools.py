#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:38:47 2024

@author: jack
# https://blog.csdn.net/qq_39227541/article/details/119358373
# https://blog.csdn.net/UncleWa/article/details/123780502


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














































