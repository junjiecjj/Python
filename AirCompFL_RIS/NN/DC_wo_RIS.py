#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 22:30:14 2024

@author: jack
"""

import numpy as np
import copy
import cvxpy as cp
import scipy


def DC_F(N, K, h_d, G, theta, rho, epsilon_dc, iter_num, verbose,):
    h = np.zeros([N, K], dtype = complex)
    H = np.zeros([N, N, K], dtype = complex)
    for i in range(K):
        h[:, i] = h_d[:, i] + G[:, :, i] @ theta
        H[:,:, i] = np.outer(h[:, i], h[:, i].conj())
    ## define the optimization problem
    M_var = cp.Variable((N, N), hermitian = True)
    M_partial = cp.Parameter((N, N), hermitian = True)

    tmp = np.random.randn(N, 1) + 1j*np.random.randn(N, 1)
    M = np.outer(tmp, tmp.conj())
    _, v = np.linalg.eigh(M)
    u = v[:, N-1]
    M_partial.value = np.outer(u, u.conj())

    ## constraints
    constraints = [M_var >> 0]
    constraints += [cp.real(cp.trace(M_var@H[:,:,k])) >= 1 for k in range(K)]
    cost = cp.real(cp.trace(M_var)) + rho*cp.real(cp.trace((np.eye(N) - M_partial)@M_var))
    prob = cp.Problem(cp.Minimize(cost), constraints)
    obj_pre = 0

    # iteritively solve:
    for i in range(iter_num):
        prob.solve() # solver = cp.MOSEK
        # obj = np.real(np.trace(M_var.value)) + rho * (np.real(np.trace(M_var.value)) - np.linalg.norm(M_var.value, ord = 2))
        if verbose > 1:
            print(f'   Solving f, Inner iter = {i}, Status = {prob.status}, Value = {prob.value:.3f} ' )
        err = np.abs(prob.value - obj_pre)
        M = copy.deepcopy(M_var.value)
        _, V = np.linalg.eigh(M)
        u = V[:, N-1]
        M_partial.value = copy.deepcopy(np.outer(u, u.conj()))
        obj_pre = prob.value
        if err < epsilon_dc:
            break
    u, _, _ = np.linalg.svd(M, compute_uv = True, hermitian = True)
    f = u[:,0]
    return f # / np.linalg.norm(f)

def DC_woRIS(N, L, K, h_d, G, epsilon, epsilon_dc, SNR, maxiter, iter_num, rho, verbose, ):
    MSE_log = np.zeros(maxiter + 1)
    f = np.random.randn(N, ) + 1j * np.random.randn(N, )
    f = f / np.linalg.norm(f, ord = 2)
    theta = np.zeros(L, dtype = complex)

    h = np.zeros([N, K], dtype = complex)
    for i in range(K):
        h[:, i] = h_d[:, i] + G[:, :, i] @ theta
    MSE_pre = np.linalg.norm(f, ord = 2)**2 / min(np.abs(f.conj()@h)**2) / SNR
    MSE_log[0] = MSE_pre

    for it in range(maxiter):
        print(f"  Outer iter = {it}:")
        f = DC_F(N, K, h_d, G, theta, rho, epsilon_dc, iter_num, verbose,)
        # print(f"  Outer iter = {it}, f = {f}")

        h = np.zeros([N, K], dtype = complex)
        for k in range(K):
            h[:, k] = h_d[:, k] + G[:, :, k] @ theta
        MSE = np.linalg.norm(f, ord = 2)**2 / min(np.abs(f.conj()@h)**2) / SNR
        MSE_log[it + 1] = MSE
        if verbose:
            print(f'  Outer iter = {it}, MSE = {MSE}, ')
        if np.abs(MSE - MSE_pre) < epsilon:
            break
        MSE_pre = MSE
    MSE_log = MSE_log[:it + 2]
    return f, MSE_log







































