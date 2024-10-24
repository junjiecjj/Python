#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:47:20 2024

@author: jack
"""

import copy
import numpy as np
np.set_printoptions(precision = 6, threshold = 1e3)
import warnings
import cvxpy as cp
import sys


def DC_F(N, K, h_d, G, theta, rho, epsilon_dc, iter_num, verbose,):
    h = np.zeros([N, K], dtype = complex)
    H = np.zeros([N, N, K], dtype = complex)
    for i in range(K):
        h[:,i] = h_d[:,i] + G[:,:,i]@theta
        H[:,:,i] = np.outer(h[:,i], h[:,i].conj())
    M = np.random.randn(N, 1) + 1j*np.random.randn(N, 1)
    M = copy.deepcopy(np.outer(M, M.conj()))
    _, V = np.linalg.eigh(M)
    u = V[:, N-1]

    # define the optimization problem
    M_var = cp.Variable((N, N), complex = True)
    M_partial = cp.Parameter((N, N), hermitian = True)
    M_partial.value = copy.deepcopy(np.outer(u, u.conj()))

    constraints = [M_var >> 0]
    constraints += [cp.real(cp.trace(M_var@H[:,:,k])) >= 1 for k in range(K)]
    cost = cp.real(cp.trace(M_var)) + rho*cp.real(cp.trace((np.eye(N) - M_partial)@M_var))
    prob = cp.Problem(cp.Minimize(cost), constraints)
    obj0 = 0
    # iteritively solve:
    for it in range(iter_num):
        prob.solve()
        if verbose > 1:
            print(f'   Solving f, Inner iter = {it}, Status = {prob.status}, Value = {prob.value} ' )
        if prob.status == 'infeasible' or prob.value is np.inf:
            print(f'   Solving f infeasible, Status = {prob.status}, Value = {prob.value}' )
            break

        err = np.abs(prob.value - obj0)
        M = copy.deepcopy(M_var.value)
        _, V = np.linalg.eigh(M)
        u = V[:, N-1]
        M_partial.value = copy.deepcopy(np.outer(u, u.conj()))
        obj0 = prob.value
        if err < epsilon_dc:
            break
    u, _, _ = np.linalg.svd(M, compute_uv = True, hermitian = True)
    m = u[:,0]
    return m / np.linalg.norm(m)


def DC_theta(N, L, K, h_d, G, f, epsilon_dc, iter_num, verbose,):
    #Compute R,c:
    A = np.zeros([L, K], dtype = complex)
    c = np.zeros([K,], dtype = complex)
    R = np.zeros([L+1, L+1, K], dtype = complex)
    for k in range(K):
        c[k] = f.conj()@h_d[:,k]
        A[:, k] = (f.conj()@G[:,:,k])
        R[0:L, 0:L, k] = np.outer(A[:, k], A[:, k].conj())
        R[0:L, L, k] = A[:, k]*c[k]
        R[L, 0:L, k] = R[0:L, L, k].conj()

    #initial V:
    V = np.random.randn(L+1,1) + 1j*np.random.randn(L+1,1);
    V = V/np.abs(V)
    V = copy.deepcopy(np.outer(V, V.conj()))
    _, v = np.linalg.eigh(V)
    u = v[:,L] # u = np.random.randn(L+1,1) + 1j*np.random.randn(L+1,1);
    #initial other parameters:
    infeasible_check = False
    #initial the optimization problem:
    V_var = cp.Variable((L+1, L+1), hermitian = True)
    V_var.value = V
    V_partial = cp.Parameter((L+1, L+1), hermitian = True)
    V_partial.value = copy.deepcopy(np.outer(u, u.conj()))

    constraints = [V_var >> 0]
    constraints += [V_var[n,n] == 1 for n in range(L)]
    constraints += [cp.real(cp.trace(V_var@R[:,:,k])) + np.abs(c[k])**2 >= 1 for k in range(K)]
    cost = cp.real(cp.trace((np.eye(L+1) - V_partial)@V_var))
    prob = cp.Problem(cp.Minimize(cost), constraints)
    obj0 = 0
    for it in range(iter_num):
        prob.solve()
        if verbose > 1:
            print(f'   Solving theta, iter = {it}, Status = {prob.status}, Value = {prob.value} ' )
        if prob.status == 'infeasible' or prob.value is None:
            infeasible_check = True
            print(f'   Solving theta infeasible, Status = {prob.status}, Value = {prob.value}' )
            break

        err = np.abs(prob.value - obj0)
        V = copy.deepcopy(V_var.value)
        _, v = np.linalg.eigh(V)
        u = v[:,L]
        V_partial.value = copy.deepcopy(np.outer(u,u.conj()))
        obj0 = prob.value
        if err < epsilon_dc:
            break
    u, _, _ = np.linalg.svd(V, compute_uv = True, hermitian = True)
    v_tilde = u[:,0]
    vv = v_tilde[0:L]/v_tilde[L]
    vv = copy.deepcopy(vv/np.abs(vv))
    return vv, infeasible_check

def DC_RIS1(N, L, K, h_d, G, epsilon, epsilon_dc, SNR, maxiter, iter_num, rho, verbose,):
    MSE_log = np.zeros(maxiter + 1)
    f = np.random.randn(N, ) + 1j * np.random.randn(N, )
    f = f / np.linalg.norm(f, ord = 2)
    theta = np.ones(L, dtype = complex)

    h = np.zeros([N, K], dtype = complex)
    for i in range(K):
        h[:, i] = h_d[:, i] + G[:, :, i] @ theta
    MSE_pre = np.linalg.norm(f, ord = 2)**2 / min(np.abs(f.conj()@h)**2) / SNR
    MSE_log[0] = MSE_pre

    infeasible = False
    for it in range(maxiter):
        print(f"  Outer iter = {it}:")
        f = DC_F(N, K, h_d, G, theta, rho, epsilon_dc, iter_num, verbose,)
        # print(f"  Outer iter = {it}, f = {f}")
        theta, infeasible = DC_theta(N, L, K, h_d, G, f, epsilon_dc, iter_num, verbose, )
        # print(f"  Outer iter = {it}, theta = {theta}")
        h = np.zeros([N, K], dtype = complex)
        for k in range(K):
            h[:, k] = h_d[:, k] + G[:, :, k] @ theta
        MSE = np.linalg.norm(f, ord = 2)**2 / min(np.abs(f.conj()@h)**2) / SNR
        MSE_log[it + 1] = MSE
        if verbose:
            print(f'  Outer iter = {it}, MSE = {MSE}, infeasible = {infeasible}')
        if np.abs(MSE - MSE_pre) < epsilon or infeasible == True:
            break
        MSE_pre = MSE
    MSE_log = MSE_log[:it + 2]
    return f, theta, MSE_log




















































