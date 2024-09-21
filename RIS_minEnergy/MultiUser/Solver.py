#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:55:18 2024

@author: jack

(1) 如何创建具有多个特殊属性（例如布尔和对称）的变量？
   首先创建具备各种需要属性的变量，然后通过添加等式约束将它们全部设置为相等。

(2) 我能在CVXPY对象上使用NumPy函数吗？
    不行，只能在CVXPY对象上使用CVXPY函数。 如果在CVXPY对象上使用NumPy函数， 可能会出现令人困惑的失败情况。

您可以使用您选择的数值库构造矩阵和向量常量。例如，如果 x 是 CVXPY 表达式 A @ x + b 中的变量，那么 A 和 b 可以是 Numpy 数组、SciPy 稀疏矩阵等。甚至 A 和 b 可以是不同类型的。
目前可以使用以下类型作为常数：
    NumPy的ndarrays（NumPy的多维数组）
    NumPy的matrices（NumPy的矩阵）
    SciPy的sparse matrices（SciPy的稀疏矩阵）


https://github.com/cvxpy/cvxpy/issues/907

https://ask.cvxr.com/t/problem-with-inequality-constraints-in-cvxpy/10953
"""


import numpy as np
import math
import cvxpy as cp

### 对约束不等式的两边同时加了不等式左边的约束.这样方便写约束
def SOCPforW_1(H, Uk, M, gamma):
    a = np.sqrt(1+1/gamma)

    W = cp.Variable((M, Uk), complex = True)
    obj = cp.Minimize(cp.norm(W, 'fro')**2)
    constraints = [ cp.imag(cp.diag(H@W)) == 0,] + [a * cp.real(H[i,:]@W[:,i]) >= cp.norm(cp.hstack([H[i,:]@W, 1.0])) for i in range(Uk)]

    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status == 'optimal':
         # print("optimal")
         pow = 10*np.log10(prob.value * 1000)
         # print(V.value)
    else:
         print("No Optimal")
    return pow, W.value


### 约束不等式两边是严格按照公式，但是约束难写。和上面是等价的
def SOCPforW(H, Uk, M, gamma):
    idxsum = np.zeros((Uk, Uk-1), dtype = int)
    for i in range(Uk):
        a = list(np.arange(Uk))
        a.remove(i)
        idxsum[i, :] = a[:]

    W = cp.Variable((M, Uk), complex = True)
    obj = cp.Minimize(cp.norm(W, 'fro')**2)
    constraints = [ cp.imag(cp.diag(H@W)) == 0,] + [cp.real(H[k,:]@W[:,k]) >= np.sqrt(gamma)*cp.norm(cp.hstack([H[k,:]@W[:, idxsum[k]], 1.0])) for k in range(Uk)]

    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status == 'optimal':
         # print("optimal")
         pow = 10 * np.log10(prob.value * 1000)
         # print(V.value)
    else:
         print("No Optimal")
    return pow,  W.value

#%%
def SDPforV(W, Hr, Hd, G, N, Uk, gamma, L = 1000):
    b = Hd.T.conjugate() @ W   # b_{k,j} in C^{Uk x Uk}
    R = np.empty((Uk, Uk), dtype=object)
    for k in range(Uk):
        for j in range(Uk):
            a_kj = np.diag(Hr[:,k].conjugate()) @ G @ (W[:, j].reshape(-1,1))
            A = a_kj @ (a_kj.T.conjugate())
            B = a_kj * (b[k,j].conjugate())
            C = (a_kj.T.conjugate()) * b[k,j]
            C = np.append(C, 0).reshape(1, -1)
            tmp = np.concatenate((A, B), axis = 1)
            tmp = np.concatenate((tmp, C), axis = 0)
            R[k, j] = tmp

    idxsum = np.zeros((Uk, Uk-1), dtype = int)
    for i in range(Uk):
        a = list(np.arange(Uk))
        a.remove(i)
        idxsum[i, :] = a

    ## Variables to solved.
    V = cp.Variable((N+1, N+1), hermitian = True)
    alpha = cp.Variable((Uk), nonneg = True )
    RV = np.empty((Uk, Uk), dtype = object)
    for k in range(Uk):
        for j in range(Uk):
            RV[k, j] = cp.trace(R[k, j] @ V)

    obj = cp.Maximize(cp.sum(alpha))
    constraints = [ 0 << V, cp.diag(V) == 1,]
    # for k in range(Uk):
    #     constraints.append(cp.real(cp.trace(R[k, k] @ V))  + (np.abs(b[k, k]))**2 >= gamma * cp.real(cp.sum([cp.trace(R[k, i] @ V) for i in range(Uk) if i != k])) + gamma * (cp.sum_squares(b[k, idxsum[k]]) + 1) + alpha[k] )
    #     constraints.append(alpha[i] >= 0)

    ## or
    # constraints += [cp.real(cp.trace(R[k, k] @ V)) + (np.abs(b[k, k]))**2 >= gamma * cp.real(cp.sum([cp.trace(R[k, i] @ V) for i in range(Uk) if i != k])) + gamma * (cp.sum_squares(b[k, idxsum[k]]) + 1) + alpha[k] for k in range(Uk) ]

    ## or
    constraints += [cp.real(RV[k, k]) + (np.abs(b[k, k]))**2 >= gamma * cp.real(np.sum(RV[k, idxsum[k]])) + gamma * (np.linalg.norm(b[k, idxsum[k]],2)**2 + 1) + alpha[k] for k in range(Uk) ]

    prob = cp.Problem(obj, constraints)
    prob.solve() # solver = cp.GUROBI, verbose=True

    if prob.status == 'optimal':
          # print(V.value)
          max_alpha = prob.value
    else:
         print("No Optimal")
         exit(-1)

    ## method 1: 高斯随机化过程
    max_F = -1e13
    max_v = -1e13
    Sigma, U = np.linalg.eig(V.value)
    for l in range(L):
        r = np.sqrt(1/2) * ( np.random.randn(N+1, 1) + 1j * np.random.randn(N+1, 1) )
        v = U @ (np.diag(Sigma)**(1/2)) @ r
        Vg = v @ (v.T.conjugate())
        RVg = np.empty((Uk, Uk), dtype = complex)
        for k in range(Uk):
            for j in range(Uk):
                RVg[k, j] = np.trace(R[k, j] @ Vg)
        alpha_s = 0
        for k in range(Uk):
            alpha_s = alpha_s + RVg[k,k] + (np.abs(b[k,k]))**2 - gamma * np.sum(RVg[k, idxsum[k]]) - gamma * (np.linalg.norm(b[k,idxsum[k]],2)**2 + 1)

        # print(f"alpha_s = {np.real(alpha_s)}, max_F = {max_F}")
        if np.real(alpha_s) > max_F:
            max_v = v
            max_F = np.real(alpha_s)
    optim_v = np.exp(1j * np.angle(max_v/max_v[-1]))
    optim_v = optim_v[:-1]
    return optim_v, max_alpha

#%% Multiuser system: Alternating Optimization Algorithm
def AlternatingOptim(Hr, Hd, G, M, N, Uk, gamma, epsilon = 1e-4, L = 1000):
    P0 = 0
    P_new  = 10
    maxIter = 30
    iternum = 0
    H = np.zeros((Uk, M), dtype = complex)

    ## random init theta
    theta = np.random.rand(N) * np.pi * 2
    Theta = np.diag(np.exp(1j * theta))

    # theta = SDRsolver(Hr, Hd, G, N)
    # Theta = np.diag(theta.flatten().conjugate())
    Pow = []
    while np.abs(P_new - P0) > epsilon and iternum < maxIter:
        iternum += 1
        # print(f"  iternum = {iternum}, eps = {np.abs(P_new - P0)}")
        H = Hr.T.conjugate() @ Theta @ G + Hd.T.conjugate()
        pow_optim, W = SOCPforW(H, Uk, M, gamma)
        Pow.append(pow_optim)
        # print(f"pow = {Pow}, W = {W}")
        P0 = P_new
        P_new = pow_optim
        v, _ = SDPforV(W, Hr, Hd, G, N, Uk, gamma)
        Theta = np.diag(v.flatten().conjugate())

    return iternum, Pow, W, v

#%% Stage 1: find V
def SDRsolver(Hr, Hd, G, N, Uk = 4, L = 200, gamma = 1):
    t = [1/gamma]*Uk
    R = np.empty((Uk,), dtype = object)
    for k in range(Uk):
        Phai = np.diag(Hr[:,k].conjugate().flatten()) @ G
        A = Phai @ (Phai.T.conjugate())
        B = Phai @ (Hd[:,k].reshape(-1,1))
        C = (Hd[:,k].conjugate().reshape(1,-1)) @ (Phai.T.conjugate())
        C = np.append(C, 0).reshape(1, -1)
        tmp = np.concatenate((A, B), axis = 1)
        tmp = np.concatenate((tmp, C), axis = 0)
        R[k] = tmp
    Hd_norm_2 = np.power(np.linalg.norm(Hd, ord = 2,  axis = 0, ), 2)

    ## use cvx to solve
    V = cp.Variable((N+1, N+1), hermitian = True)
    RV = np.empty((Uk,), dtype = object)
    for k in range(Uk):
        RV[k] = cp.trace(R[k] @ V)

    # obj = cp.Maximize(cp.real( cp.sum([t[k]*(RV[k]+Hd_norm_2[k]) for k in range(Uk)]) ))
    ## Or use following obj.
    obj = cp.Maximize(cp.real(np.sum(t * (RV + Hd_norm_2)) ))
    constraints = [
        0 << V,
        cp.diag(V) == 1,
        ]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status == 'optimal':
         low_bound = prob.value
    else:
         print("Not optimal")
         exit(-1)

    ## method 1: 高斯随机化过程
    max_F = -1e13
    max_v = -1e13
    Sigma, U = np.linalg.eig(V.value)
    for i in range(L):
        r = np.sqrt(1/2) * ( np.random.randn(N+1, 1) + 1j * np.random.randn(N+1, 1) )
        v = U @ (np.diag(Sigma)**(1/2)) @ r
        vRv = 0
        vRv = np.sum([t[k] * (v.T.conjugate() @ R[k] @ v + Hd_norm_2[k] ) for k in range(Uk)])
        if np.real(vRv) > max_F:
            max_v = v
            max_F = np.real(vRv)
    optim_v = np.exp(1j * np.angle(max_v/max_v[-1]))
    optim_v = optim_v[:-1]
    return  optim_v


### Multiuser system: Two-Stage Algorithm
def TwoStageAlgorithm(Hr, Hd, G, M, N, Uk, gamma, epsilon, L):
    ## Stage 1: find V
    v = SDRsolver(Hr, Hd, G, N, Uk, L)

    ## Stage 2: find W
    Theta = np.diag(v.flatten().conjugate())
    H = Hr.T.conjugate() @ Theta @ G + Hd.T.conjugate()
    pow_optim, W = SOCPforW(H, Uk, M, gamma)

    return pow_optim, W, v












































