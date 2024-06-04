#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 23:17:11 2024

@author: jack
"""




import sys
import numpy as np
import scipy
import cvxpy as cpy



azi_AP = 0
# ele_AP = 0



M = 4
d1 = np.arange(M)
SteerVecAP = np.exp(-1j * np.pi * d1 * np.sin(azi_AP))


azi_RIS = -np.pi
ele_RIS = 0
Nx = 2
Ny = 3
dx = np.arange(Nx)
dy = np.arange(Ny)
A = np.exp(-1j * np.pi * dx * np.sin(azi_RIS) * np.cos(ele_RIS))
B = np.exp(-1j * np.pi * dy * np.sin(ele_RIS) )
SteerVecRIS = np.kron(A, B)






























