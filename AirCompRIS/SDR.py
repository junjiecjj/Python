#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:47:21 2024

@author: jack
"""


import numpy as np
import copy
import cvxpy as cp
import scipy

# Given theta, update M
def SDR_F(N, K, h_d, G, theta, verbose,):
    h = np.zeros([N, K], dtype = complex)
    H = np.zeros([N, N, K], dtype = complex)
    for i in range(K):
        h[:, i] = h_d[:, i] + G[:, :, i] @ theta
        H[:,:, i] = np.outer(h[:, i], h[:, i].conj())
    ## define the optimization problem
    M_var = cp.Variable((N, N), hermitian = True)

    ## constraints
    constraints = [M_var >> 0]
    constraints += [cp.real(cp.trace(M_var@H[:,:,k])) >= 1 for k in range(K)]
    cost = cp.real(cp.trace(M_var))
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    if prob.status == 'optimal':
         print(f'   Solving f, Status = {prob.status}, prob.Value = {prob.value:.3e} ' )
    else:
         print(f'   Solving f infeasible, Status = {prob.status}, prob.Value = {prob.value:.3e} ' )
         exit(-1)
    ### method 1: 高斯随机化过程
    Gmax = 1000  # Gaussian 随机化次数
    max_M = 1e13
    max_f = -1e13
    Sigma, U = np.linalg.eig(M_var.value)
    for i in range(Gmax):
        r = np.sqrt(1/2) * ( np.random.randn(N, 1) + 1j * np.random.randn(N, 1) )
        f = U @ (np.diag(Sigma)**(1/2)) @ r
        f2 = np.real(f.T.conjugate() @ f)
        # print(f"   f, 高斯随机, {i}, {f2}")
        if f2 < max_M:
            max_f = f
            max_M = f2
    optim_f = max_f.flatten()
    return optim_f

# Given M, update theta
def SDR_theta(N, L, K, h_d, G, f,  verbose, ):
    c = np.zeros(K, dtype = complex)
    R = np.zeros((L+1, L+1, K), dtype = complex)
    for k in range(K):
        c[k] = f.conj() @ h_d[:,k]
        akh = (f.conj() @ G[:,:,k])
        A = np.outer(akh.conj(), akh)
        B = (c[k] * akh.conj()).reshape(-1,1)
        C = B.T.conj()
        D = np.array([[0]])
        R[:,:,k] = np.block([[A, B], [C, D]])

    #initial the optimization problem:
    alpha = cp.Variable((K), nonneg = True )
    V_var = cp.Variable((L+1, L+1), hermitian = True)

    constraints = [V_var >> 0, ] # cp.diag(V_var) == 1]
    constraints += [V_var[l, l] == 1 for l in range(L)]
    constraints += [cp.real(cp.trace(R[:, :, k]@V_var)) + np.abs(c[k])**2 >= 1 + alpha[k] for k in range(K)]
    prob = cp.Problem(cp.Minimize(cp.sum(alpha)), constraints)
    prob.solve()

    if prob.status == 'optimal':
         print(f'   Solving theta, Status = {prob.status}, prob.Value = {prob.value:.3e}')
         # print(f"   V_var = {V_var.value.shape}")
    else:
         print(f'   Solving theta infeasible, Status = {prob.status}, prob.Value = {prob.value:.3e}')

    ## method 1: 高斯随机化过程
    Gmax = 1000
    max_F = -1e13
    max_v = -1e13
    Sigma, U = np.linalg.eig(V_var.value,)
    # print(f"   Sigma.shape = {Sigma.shape}, U.shape = {U.shape}")
    for l in range(Gmax):
        r = np.sqrt(1/2) * ( np.random.randn(L+1, 1) + 1j * np.random.randn(L+1, 1) )
        v = U @ (np.diag(Sigma)**(1/2)) @ r
        Vg = v @ (v.T.conjugate())
        alpha_s = 0
        for k in range(K):
            alpha_s += np.real(np.trace(R[:, :, k]@Vg)) + np.abs(c[k])**2 - 1
        # print(f"alpha_s = {np.real(alpha_s)}, max_F = {max_F}")
        if np.real(alpha_s) > max_F:
            max_v = v
            max_F = np.real(alpha_s)
    theta = max_v[0:L]/max_v[L]
    theta = (theta/np.abs(theta)).flatten()

    return theta

def SDR_RIS(N, L, K, h_d, G, epsilon, P0, maxiter,  verbose, ):
    MSE_log = np.zeros(maxiter + 1)
    f = np.random.randn(N, ) + 1j * np.random.randn(N, )
    theta = np.random.randn(L, ) + 1j * np.random.randn(L, )
    theta0 = theta/np.abs(theta)

    h = np.zeros([N, K], dtype = complex)
    for i in range(K):
        h[:, i] = h_d[:, i] + G[:, :, i] @ theta
    MSE_pre = np.linalg.norm(f, ord = 2)**2 / min(np.abs(f.conj()@h)**2) / P0
    MSE_log[0] = MSE_pre

    for it in range(maxiter):
        print(f"  Outer iter = {it}:")
        f = SDR_F(N, K, h_d, G, theta, verbose,)
        # print(f"  Outer iter = {it}, f = {f}")
        theta = SDR_theta(N, L, K, h_d, G, f, verbose, )
        # print(f"  Outer iter = {it}, theta = {theta}")
        h = np.zeros([N, K], dtype = complex)
        for k in range(K):
            h[:, k] = h_d[:, k] + G[:, :, k] @ theta
        MSE = np.linalg.norm(f, ord = 2)**2 / min(np.abs(f.conj()@h)**2) / P0
        MSE_log[it + 1] = MSE
        if verbose:
            print(f'  Outer iter = {it}, MSE = {MSE},  ')
        if np.abs(MSE - MSE_pre) < epsilon:
            break
        MSE_pre = MSE
    MSE_log = MSE_log[:it + 2]
    return f, theta, MSE_log







































