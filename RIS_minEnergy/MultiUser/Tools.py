#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 01:15:37 2024

@author: jack
"""
import sys
import numpy as np



def ULA2UPA_Los(M = 4, Nx = 3, Ny = 4, azi_AP = 0, ele_AP = 0, azi_RIS = -np.pi, ele_RIS = 0):
    SteerVecAP = np.exp(-1j * np.pi * np.arange(M) * np.sin(azi_AP))
    A = np.exp(-1j * np.pi * np.arange(Nx) * np.sin(azi_RIS) * np.cos(ele_RIS))
    B = np.exp(-1j * np.pi * np.arange(Ny) * np.sin(ele_RIS) )
    SteerVecRIS = np.kron(A, B)

    GLos = SteerVecRIS.reshape(-1, 1) @ (SteerVecAP.reshape(1, -1))
    return GLos


def RIS2UserSteerVec( Nx = 3, Ny = 4, azi = 0, ele = 0 ):
    A = np.exp(-1j * np.pi * np.arange(Nx) * np.sin(azi) * np.cos(ele))
    B = np.exp(-1j * np.pi * np.arange(Ny) * np.sin(ele) )
    SteerVecRIS = np.kron(A, B)
    return SteerVecRIS














