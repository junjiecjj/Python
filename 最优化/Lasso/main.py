

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

### 近似点梯度下降法
def ProximalGradientDescent(H, b, x_true, mu = 0.01, maxIter = 1000, tol = 1e-6, ):
    M, N = H.shape

    ## alpha = 0.005
    alpha = 1 / np.real(max(np.linalg.eig(H.T@H)[0]))  ## 固定步长，取为A^T@A的最大特征值的倒数

    Xk = np.zeros(N)

    cost_history = []
    mse_history = []

    for _ in range(maxIter):
        Xk_half = Xk - alpha * (H.T @ (H @ Xk - b))
        # 软门限算子
        Xk_new = proximal_operator(Xk_half, mu * alpha)

        if np.linalg.norm(Xk_new - Xk, ord = 2) < tol:
            break
        else:
            Xk = Xk_new.copy()
        cost = 0.5 * np.linalg.norm(b - H @ Xk_new) ** 2 + mu * np.sum(np.abs(Xk_new))
        mse = np.linalg.norm(x_true - Xk_new)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)

    return Xk_new, mse_history, cost_history

## 梯度下降法: 没有(Barzilai-Borwein)BB步长(即固定步长)，没有连续化策略
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
def GradientDescent(H, b, x_true, mu = 0.01, delta = 0.1, maxIter = 1000, tol = 1e-6):
    M, N = H.shape
    # alpha = 0.005
    L =  np.real(max(np.linalg.eig(H.T@H)[0])) + mu/delta ## 固定步长，
    alpha = 1 / (2*L)
    Xk = np.zeros(N)

    cost_history = []
    mse_history = []

    for _ in range(maxIter):
        Xk = Xk - alpha * ( H.T @ (H @ Xk - b) +  mu * HuberGrad(Xk, delta))

        cost = 0.5 * np.linalg.norm(b - H @ Xk) ** 2 + mu * np.sum(np.abs(Xk))
        mse = np.linalg.norm(x_true - Xk)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)

    return Xk, mse_history, cost_history

## Page221, 梯度下降法: 有非精确线搜索的 BB 步长梯度下降法，没有连续化策略
def GradientDescent_BB(H, b, x_true, mu = 0.01, delta = 0.1, maxIter = 1000,):
    """
    Parameters
    ----------
    H :
    b :
    x_true :
    mu : 原问题对应的正则化系数
    delta : Huber 光滑化参数
    maxIter :  The default is 1000.

    Returns
    -------
    Xk : TYPE
        DESCRIPTION.
    mse_history : TYPE
        DESCRIPTION.
    cost_history : TYPE
        DESCRIPTION.

    """
    M, N = H.shape
    xk = np.zeros(N)
    alpha0 = 0.01 # 步长的初始值
    gtol = 1e-6
    ftol = 1e-8

    r = (H @ xk - b)
    g = H.T @ r +  mu * HuberGrad(xk, delta)
    nrmG = np.linalg.norm(g, ord = 2)
    fsmooth = 0.5 * np.linalg.norm(r) ** 2 + mu * np.sum(Huber(xk, delta)) # 光滑化函数值

    alpha = alpha0 # 起始步长
    gamma = 0.2 # 线搜索条件未满足时步长的衰减率
    ## 线搜索参数
    rhols = 1e-3  # C1
    eta = 0.85 # zhang & Hager
    Q = 1
    Cval = fsmooth

    cost_history = []
    mse_history = []
    cost = 0.5 * np.linalg.norm(r) ** 2 + mu * np.sum(np.abs(xk)) # 原始问题目标函数的值
    mse = np.linalg.norm(x_true - xk)**2 / np.linalg.norm(x_true)**2
    cost_history.append(cost)
    mse_history.append(mse)

    for k in range(maxIter):
        fp = fsmooth
        gp = g.copy()
        xp = xk.copy()
        ## 线搜索准则 (Zhang & Hager) f(xk+αdk)≤Ck+ρα(gk)⊤dk 或进行超过 10 次步长衰减后退出线搜索。 在当前步长不符合线搜索条件的情况下，对当前步长以 η 进行衰减，线搜索次数加一。
        nls = 1
        for _ in range(10):
            xk = xp - alpha * gp
            r = H @ xk - b
            g = H.T @ r +  mu * HuberGrad(xk, delta)
            fsmooth = 0.5 * np.linalg.norm(r) ** 2 + mu * np.sum(Huber(xk, delta))
            if fsmooth < Cval - alpha * rhols * nrmG**2:
                break
            alpha *= gamma
            nls += 1
        nrmG = np.linalg.norm(g, ord = 2)
        cost = 0.5 * np.linalg.norm(r) ** 2 + mu * np.sum(np.abs(xk))
        mse = np.linalg.norm(x_true - xk)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)

        if nrmG < gtol or np.abs(fp - fsmooth) < ftol:
            break
        ## 计算 BB 步长作为下一步迭代的初始步长。
        dx = xk - xp
        dg = g - gp
        dxg = np.abs(dx @ dg)
        if dxg > 0:
            if k %2 == 0:
                alpha = np.abs(dx @ dx)/dxg
            else:
                alpha = dxg/np.abs(dg @ dg)
            ## 保证步长的合理范围
            alpha =  max(1e-12*1.0,  min(alpha, 1e12*1.0))
        Qp = Q
        Q = eta * Qp + 1
        Cval = (eta * Qp * Cval + fsmooth)/Q
    return xk, mse_history, cost_history

## 次梯度解法
def L1_subgrad(H, b, x_true, mu = 0.01, maxIter = 1000, ftol = 1e-6, steptype = 'diminishing2'):
    M, N = H.shape
    alpha0 = 1/np.real(max(np.linalg.eig(H.T@H)[0]))
    # alpha0 = 0.0002
    thres = 1e-4
    xk = np.zeros(N)
    cost_history = []
    mse_history = []
    cost = 0.5 * np.linalg.norm(b - H @ xk) ** 2 + mu * np.sum(np.abs(xk))
    mse = np.linalg.norm(x_true - xk)**2 / np.linalg.norm(x_true)**2
    cost_history.append(cost)
    mse_history.append(mse)

    for k in range(maxIter):
        if steptype == 'fixed':
            alpha = alpha0
        elif steptype == 'diminishing':
            alpha = alpha0 / np.sqrt(k+1)
        elif steptype == 'diminishing2':
            alpha = alpha0 / (k+1)
        tmp = xk.copy()
        tmp[np.where(tmp < thres)] = 0
        xk = xk - alpha * (H.T @ (H @ xk - b) +  mu * np.sign(tmp))
        f_now = 0.5 * np.linalg.norm(b - H @ xk) ** 2 + mu * np.sum(np.abs(xk))
        mse = np.linalg.norm(x_true - xk)**2 / np.linalg.norm(x_true)**2
        cost_history.append(f_now)
        mse_history.append(mse)

        if k > 1 and np.abs(f_now - cost)/np.abs(f_now) < ftol:
            break
        cost = f_now
    return xk, mse_history, cost_history

## 罚函数法解基追踪问题
def BP_penaltyFunctionMethod(H, b, x_true, maxIter = 1000, ftol = 1e-6, steptype = 'fixed'):
    M, N = H.shape
    cost_history = []
    mse_history = []
    x = np.zeros(N)
    Mu = [10, 1, 0.1, 0.01, 0.001]
    maxIt = int(maxIter/len(Mu))
    for mu in Mu:
        x, mse, cost = L1_subgrad(H, b, x_true, mu = mu, maxIter = maxIt, ftol = 1e-6, steptype = steptype)
        mse_history += mse
        cost_history += cost
    return x, mse_history, cost_history

## 基追踪问题的增广拉格朗日函数法(Augmented Lagrange function method)
def BP_ALM(H, b, x_true, mu = 0.01, maxIter = 1000, ftol = 1e-6, steptype = 'diminishing2'):

    return

plt.close('all')
np.random.seed(42)
M, N = 500, 1000
sparsity = 0.03
K = int(N * sparsity)
maxIter = 300
lambda_ = 0.001 # 正则化系数 mu
rho = 0.01  # 罚因子
noise_varDB = 50
noise_var = 10**(-noise_varDB/10)   ##  0.02  # 0.00009 ~ 0.02
H, y, x_true = generate_data(M, N, K, noise_var = noise_var, x_choice = 1)

x_omp, mse_omp, cost_omp = OMP1(H, y, K, x_true, lambda_ = lambda_, maxIter = maxIter,)
x_admm, mse_admm, cost_admm = LASSO_admm_primal(H, y, x_true, mu = lambda_, rho = rho, maxIter = maxIter,)
x_admm_dual, mse_admm_dual, cost_admm_dual = LASSO_admm_dual(H, y, x_true, mu = lambda_, rho = 1/rho, maxIter = maxIter,)
x_prox, mse_prox, cost_prox = ProximalGradientDescent(H, y, x_true, mu = lambda_,  maxIter = maxIter,)
x_grad, mse_grad, cost_grad = GradientDescent(H, y, x_true, mu = lambda_, delta = 0.01, maxIter = maxIter,)
x_gradBB, mse_gradBB, cost_gradBB = GradientDescent_BB(H, y, x_true, mu = lambda_, delta = 0.01, maxIter = maxIter,)

x_sub, mse_sub, cost_sub = L1_subgrad(H, y, x_true, mu = lambda_, maxIter = maxIter, ftol = 1e-6, steptype = 'fixed')
x_pf, mse_pf, cost_pf = BP_penaltyFunctionMethod(H, y, x_true, maxIter = 600, ftol = 1e-6, steptype = 'fixed')

fig, axs = plt.subplots(1, 1, figsize=(12, 8), constrained_layout = True)
# axs.semilogy(mse_amp_tuto, ls = '--', marker = 'o', ms = '12', markevery = 50, label = "AMP_tutorial MSE" )
# axs.semilogy(mse_amp_cs, ls = '-',lw = 3, marker = 'v', ms = '12', markevery = 50, label = "AMPforCS MSE" )
# axs.semilogy(mse_amp_Ku, ls = '--', marker = '^', ms = '12', markevery = 50, label = "AMPKuanCS MSE" )
# # axs.semilogy(mse_amp_3pt, ls = '--', marker = '1', ms = '12', markevery = 50, label = "AMP_3pt MSE" )
# axs.semilogy(mse_amp_bg, ls = '--', marker = '2', ms = '12', markevery = 50, label = "AMP_bg MSE" )

# axs.semilogy(mse_nesterov, ls = '--', marker = '>', ms = '12', markevery = 50, label = "nesterov MSE" )
axs.semilogy(mse_prox, ls = '--', marker = 's', ms = '12', markevery = 50,  label = "prox_grad MSE" )
axs.semilogy(mse_omp, ls = '--', marker = 'p', ms = '12', markevery = 50, label = "OMP" )
axs.semilogy(mse_admm, ls = '--', marker = 'h', ms = '12', markevery = 50, label = "ADMM MSE" )
axs.semilogy(mse_admm_dual, ls = '--', marker = 'd', ms = '12', markevery = 50, label = "ADMM dual MSE" )
axs.semilogy(mse_grad, ls = '--', marker = 'x', ms = '12', markevery = 50, label = "Graddescent MSE" )
axs.semilogy(mse_gradBB, ls = '--', lw = 3, marker = '<', ms = '12', markevery = 50,  label = "Graddescent BB MSE" )
axs.semilogy(mse_sub, ls = '--', marker = '*', ms = '12', markevery = 50, label = "sub_grad MSE" )
axs.semilogy(mse_pf, ls = '--', marker = '2', ms = '12', markevery = 50, label = "BP_penalty MSE" )


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
axs.stem(x_gradBB, linefmt = 'k--', markerfmt = 'k^',  label="Grad BB descent X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_sub, linefmt = 'k--', markerfmt = 'k^',  label="SubGrad descent X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_pf, linefmt = 'k--', markerfmt = 'k^',  label="BP penalty X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_prox, linefmt = 'c--', markerfmt = 'c^',  label="Prox Gradient X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()

# plt.close()
# plt.close('all')














































































































































