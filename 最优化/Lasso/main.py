#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 21:15:25 2025

@author: jack
"""
## sys lib
import numpy as np
import copy
from matplotlib import pyplot as plt
import scipy
from scipy.stats import norm
from numpy import linalg as LA
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error

## my lib
from LassoOptimizer import generate_data, OMP1, FISTA

def proximal_operator(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

## Page418
def LASSO_admm_primal(H, b, x_true, mu = 0.01, rho = 0.005, maxIter = 1000, tol = 1e-6, ):
    """
    Parameters
    ----------
    H :
    y :
    x_true :
    mu : mu ||x||_1 The default is 0.01.
    rho : 二次罚项的系数, The default is 0.005.
    maxIter :  The default is 1000.
    tol :   The default is 1e-6.

    Returns
    -------
    None.

    """
    # rho = max(np.linalg.eig(H.T@H)[0])
    M, N = H.shape
    Xk = np.zeros(N)
    Zk = np.zeros(N)
    Vk = np.zeros(N)

    cost_history = []
    mse_history = []

    Inverse = scipy.linalg.inv(H.T@H + rho * np.eye(N, N))
    for _ in range(maxIter):
        # update X
        Xk_new = Inverse @ (H.T@b + rho * Zk - Vk)
        ## update Z
        Zk_new =  proximal_operator(Xk_new + Vk/rho, mu/rho)  # np.zeros(N) #
        ## update 拉格朗日乘子
        Vk_new = Vk + rho * (Xk_new - Zk_new)
        if np.linalg.norm(Xk_new - Xk, ord=2) < tol:
            break
        else:
            Xk = Xk_new.copy()
            Zk = Zk_new.copy()
            Vk = Vk_new.copy()
        cost = 0.5 * np.linalg.norm(b - H @ Xk_new) ** 2 + mu * np.sum(np.abs(Xk_new))
        mse = np.linalg.norm(x_true - Xk_new)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)

    return Xk_new, mse_history, cost_history

## Page419
def LASSO_admm_dual(H, b, x_true, mu = 0.01, rho = 0.005, maxIter = 1000, tol = 1e-6, ):
    """
    Parameters
    ----------
    H :
    y :
    x_true :
    mu : mu ||x||_1 The default is 0.01.
    rho : 二次罚项的系数, The default is 0.005.
    maxIter :  The default is 1000.
    tol :   The default is 1e-6.

    Returns
    -------
    None.

    """
    # rho = max(np.linalg.eig(H.T@H)[0])
    M, N = H.shape
    Xk = np.zeros(N)
    # Zk = np.zeros(N)
    Yk = np.zeros(M)
    tau = 1.618
    cost_history = []
    mse_history = []

    Inverse = scipy.linalg.inv(rho * H@H.T + np.eye(M, M))
    for _ in range(maxIter):
        # update Z
        Zk_new = np.minimum(mu, np.maximum(Xk/rho - H.T@Yk, -mu))
        ## update Y
        Yk_new =  Inverse @ (H @ (Xk - rho * Zk_new) - b)
        ## update 拉格朗日乘子 X
        Xk_new = Xk - tau * rho * (H.T @ Yk_new + Zk_new)
        if np.linalg.norm(Xk_new - Xk, ord=2) < tol:
            break
        else:
            Xk = Xk_new.copy()
            # Zk = Zk_new.copy()
            Yk = Yk_new.copy()
        cost = 0.5 * np.linalg.norm(b - H @ Xk_new) ** 2 + mu * np.sum(np.abs(Xk_new))
        mse = np.linalg.norm(x_true - Xk_new)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)

    return Xk_new, mse_history, cost_history

# 近端梯度下降
def ProximalGradientDescent(H, y, x_true, mu = 0.01, maxIter = 1000, tol = 1e-6, ):
    M, N = H.shape

    ## alpha = 0.005
    alpha = 1 / np.real(max(np.linalg.eig(H.T@H)[0]))  ## 固定步长，取为A^T@A的最大特征值的倒数

    Xk = np.zeros(N)

    cost_history = []
    mse_history = []

    for _ in range(maxIter):
        Xk_half = Xk - alpha * (H.T @ (H @ Xk - y))
        # 软门限算子
        Xk_new = proximal_operator(Xk_half, mu * alpha)

        if np.linalg.norm(Xk_new - Xk, ord = 2) < tol:
            break
        else:
            Xk = Xk_new.copy()
        cost = 0.5 * np.linalg.norm(y - H @ Xk_new) ** 2 + mu * np.sum(np.abs(Xk_new))
        mse = np.linalg.norm(x_true - Xk_new)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)

    return Xk_new, mse_history, cost_history

def Huber(dt, delta):
    tmp = copy.deepcopy(dt)
    x = copy.deepcopy(dt)
    x[np.where(np.abs(tmp) < delta)] = x[np.where(np.abs(tmp) < delta)]**2/(2 * delta)
    x[np.where(np.abs(tmp) >= delta)] = np.abs(x[np.where(np.abs(tmp) >= delta)]) - delta/2
    return x
def HuberGrad(dt, delta):
    tmp = copy.deepcopy(dt)
    x = copy.deepcopy(dt)
    x[np.where(np.abs(tmp) < delta)] = x[np.where(np.abs(tmp) < delta)] / delta
    x[np.where(np.abs(tmp) >= delta)] = np.sign(x[np.where(np.abs(tmp) >= delta)])
    return x

## Page221
def GradientDescent(H, b, x_true, mu = 0.01, delta = 0.1, maxIter = 1000, tol = 1e-6):
    M, N = H.shape

    # alpha = 0.005
    L =  np.real(max(np.linalg.eig(H.T@H)[0])) + mu/delta ## 固定步长，取为 A^T@A 的最大特征值的倒数
    alpha = 1 / (1*L)
    Xk = np.zeros(N)

    cost_history = []
    mse_history = []

    for _ in range(maxIter):
        Xk = Xk - alpha * ( H.T @ (H @ Xk - b) +  mu * HuberGrad(Xk, delta))

        # if np.linalg.norm(Xk_new - Xk, ord = 2) < tol:
            # break
        # else:
            # Xk = Xk_new.copy()
        cost = 0.5 * np.linalg.norm(y - H @ Xk) ** 2 + mu * np.sum(np.abs(Xk))
        mse = np.linalg.norm(x_true - Xk)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)

    return Xk, mse_history, cost_history

## Page219, Barzilar-Borwein (BB)
def GradientDescent_wBB(H, b, x_true, mu = 0.01, delta = 0.1, maxIter = 1000, tol = 1e-6):
    M, N = H.shape

    # alpha = 0.005
    L =  np.real(max(np.linalg.eig(H.T@H)[0])) + mu/delta ## 固定步长，取为 A^T@A 的最大特征值的倒数
    alpha = 1 / (2*L)
    Xk = np.zeros(N)

    cost_history = []
    mse_history = []

    for _ in range(maxIter):
        Xk = Xk - alpha * ( H.T @ (H @ Xk - b) +  mu * HuberGrad(Xk, delta))

        # if np.linalg.norm(Xk_new - Xk, ord = 2) < tol:
            # break
        # else:
            # Xk = Xk_new.copy()
        cost = 0.5 * np.linalg.norm(y - H @ Xk) ** 2 + mu * np.sum(np.abs(Xk))
        mse = np.linalg.norm(x_true - Xk)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)

    return Xk, mse_history, cost_history


plt.close('all')
np.random.seed(42)
M, N = 500, 1000
sparsity = 0.03
K = int(N * sparsity)
maxIter = 300
lambda_ = 0.001
rho = 0.01
noise_varDB = 50
noise_var = 10**(-noise_varDB/10)   ##  0.02  # 0.00009 ~ 0.02
H, y, x_true = generate_data(M, N, K, noise_var = noise_var, x_choice = 1)

x_omp, mse_omp, cost_omp = OMP1(H, y, K, x_true, lambda_ = lambda_, maxIter = maxIter,)
x_admm, mse_admm, cost_admm = LASSO_admm_primal(H, y, x_true, mu = lambda_, rho = rho, maxIter = maxIter,)
x_admm_dual, mse_admm_dual, cost_admm_dual = LASSO_admm_dual(H, y, x_true, mu = lambda_, rho = 1/rho, maxIter = maxIter,)
x_prox, mse_prox, cost_prox = ProximalGradientDescent(H, y, x_true, mu = lambda_,  maxIter = maxIter,)
x_grad, mse_grad, cost_grad = GradientDescent(H, y, x_true, mu = lambda_, delta = 0.001, maxIter = maxIter,)

fig, axs = plt.subplots(1, 1, figsize=(12, 8), constrained_layout = True)
# axs.semilogy(mse_amp_tuto, ls = '--', marker = 'o', ms = '12', markevery = 50, label = "AMP_tutorial MSE" )
# axs.semilogy(mse_amp_cs, ls = '-',lw = 3, marker = 'v', ms = '12', markevery = 50, label = "AMPforCS MSE" )
# axs.semilogy(mse_amp_Ku, ls = '--', marker = '^', ms = '12', markevery = 50, label = "AMPKuanCS MSE" )
# # axs.semilogy(mse_amp_3pt, ls = '--', marker = '1', ms = '12', markevery = 50, label = "AMP_3pt MSE" )
# axs.semilogy(mse_amp_bg, ls = '--', marker = '2', ms = '12', markevery = 50, label = "AMP_bg MSE" )
# axs.semilogy(mse_prox, ls = '--', marker = '*', ms = '12', markevery = 50, label = "prox_grad MSE" )
# axs.semilogy(mse_nesterov, ls = '--', marker = '>', ms = '12', markevery = 50, label = "nesterov MSE" )
# axs.semilogy(mse_ista, ls = '--', lw = 3, marker = '<', ms = '12', markevery = 50,  label = "ISTA MSE" )
axs.semilogy(mse_prox, ls = '--', marker = 's', ms = '12', markevery = 50,  label = "prox_grad MSE" )
axs.semilogy(mse_omp, ls = '--', marker = 'p', ms = '12', markevery = 50, label = "OMP" )
axs.semilogy(mse_admm, ls = '--', marker = 'h', ms = '12', markevery = 50, label = "ADMM MSE" )
axs.semilogy(mse_admm_dual, ls = '--', marker = 'd', ms = '12', markevery = 50, label = "ADMM dual MSE" )
axs.semilogy(mse_grad, ls = '--', marker = 'x', ms = '12', markevery = 50, label = "Graddescent MSE" )
axs.legend(fontsize = 22)
axs.set_xlabel('Iteration')
axs.set_ylabel('MSE')
plt.show()
# plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_true, linefmt = 'm--', markerfmt = 'mD',  label="True X", basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
# plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_omp, linefmt = 'k--', markerfmt = 'k^',  label="OMP X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
# plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_admm, linefmt = 'k--', markerfmt = 'k^',  label="ADMM X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_admm_dual, linefmt = 'k--', markerfmt = 'k^',  label="ADMM Dual X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_grad, linefmt = 'k--', markerfmt = 'k^',  label="Grad descent X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()


fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_prox, linefmt = 'c--', markerfmt = 'c^',  label="Prox X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()

# plt.close()
# plt.close('all')














































































































































