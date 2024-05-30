#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:55:18 2024

@author: jack
"""


import numpy as np
import math
import cvxpy as cp

def SDRsolver(G, hr, hd, N, L = 200):
    Phai = np.diag(hr.flatten()) @ G
    A = Phai @ (Phai.T.conjugate())
    B = Phai @ hd.T.conjugate()
    C = hd @ (Phai.T.conjugate())
    C = np.append(C, 0).reshape(1, -1)
    R = np.concatenate((A, B), axis = 1)
    R = np.concatenate((R, C), axis = 0)

    ## use cvx to solve
    V = cp.Variable((N+1, N+1), hermitian = True)
    obj = cp.Maximize(cp.real(cp.trace(R@V)) + cp.norm(hd, 2)**2)
    # The operator >> denotes matrix inequality.
    constraints = [
        0 << V,
        cp.diag(V) == 1,
        ]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status == 'optimal':
         # print("optimal")
         low_bound = prob.value
         # print(V.value)
    else:
         print("Not optimal")
         exit(-1)

    #%% method 1: 高斯随机化过程
    max_F = -1e13
    max_v = -1e13
    Sigma, U = np.linalg.eig(V.value)
    for i in range(L):
        r = np.sqrt(1/2) * ( np.random.randn(N+1, 1) + 1j * np.random.randn(N+1, 1) )
        v = U @ (np.diag(Sigma)**(1/2)) @ r
        # print(f"v^H @ R @ v = {v.T.conjugate() @ R @ v}, max_F = {max_F}")
        if v.T.conjugate() @ R @ v > max_F:
            max_v = v
            max_F = v.T.conjugate() @ R @ v
    try:
        optim_v = np.exp(1j * np.angle(max_v/max_v[-1]))
    except Exception as e:
        print(f"!!!!!!!!!!!!!!! {e} !!!!!!!!!!!!!!!!!!!!!!")
        # print(f"V = {V.value}")
        print(f"Sigma = {Sigma}")
        # print(f"U = {U}")
        # print(f"v = {v}")
        print(f"v^H @ R @ v = {v.T.conjugate() @ R @ v}, max_F = {max_F}")

    optim_v = optim_v[:-1]

    #%% Method 2
    #%% Method 3

    return low_bound, optim_v



def AU_MRT(hd, hr, G):
    w_aumrt = hd.T.conjugate()/np.linalg.norm(hd, 2)
    varphi0 = np.angle(hd @ w_aumrt)
    v_aumrt = np.exp(1j*(varphi0 - np.angle(np.diag(hr.flatten()) @ G @ w_aumrt)))
    return v_aumrt, w_aumrt


def AI_MRT(hd, hr, G):
    w_aimrt = G[1,:].reshape(-1,1).conjugate()/np.linalg.norm(G[1,:], 2)
    varphi0 = np.angle(hd @ w_aimrt)
    v_aimrt = np.exp(1j*(varphi0 - np.angle(np.diag(hr.flatten()) @ G @ w_aimrt)))
    return v_aimrt, w_aimrt


def AlternatingOptim(hd, hr, G, epsilon, gamma):
    ##  以Ap-user MRT进行初始化
    w = hd.T.conjugate()/np.linalg.norm(hd, 2)
    P_new = 0
    P = 10
    while (np.abs(P - P_new) > epsilon):
        varphi0 = np.angle(hd @ w)
        v = np.exp(1j*(varphi0 - np.angle(np.diag(hr.flatten()) @ G @ w)))
        P = P_new
        P_new = gamma/(np.linalg.norm((v.T @ (np.diag(hr.flatten()) @ G) + hd) @ w, 2)**2)
        w_new = (v.T @ np.diag(hr.flatten()) @ G + hd).T.conjugate()
        w = w_new/np.linalg.norm(w_new, 2)
    return P














































































