#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:18:52 2024

@author: jack
"""


import numpy as np
import copy
from matplotlib import pyplot as plt
from scipy.stats import norm
from numpy import linalg as LA
from scipy.optimize import minimize_scalar

# 初始化输入数据 A 和 b
def generate_data(m, n, rho = 0.1, noise_var = 0.0001):
    A = np.random.randn(m, n) / np.sqrt(m)  # 随机生成一个 m x n 的矩阵
    x = (np.random.rand(n, 1) < rho) * np.random.randn(n, 1) / np.sqrt(rho)
    noise = np.sqrt(noise_var) * np.random.randn(m, 1)  # 降低噪声强度
    b = A @ x + noise
    return A, b, x

# 初始化输入数据 A 和 b
def generate_data1(M, N, K, noise_var = 0.0001, x_choice = 0):
    A = np.random.randn(M, N) / np.sqrt(M)  # 随机生成一个 m x n 的矩阵
    if x_choice == 1:
        x = np.zeros((N,))
        x[np.random.choice(N, K, replace = False)] = np.random.randn(K,)
    elif x_choice == 0:
        x = np.zeros((N,))
        idx = np.random.choice(N, K, replace = False)
        idx1 = np.random.choice(idx, int(K/2), replace = False)
        idx_1 = np.setdiff1d(idx, idx1)
        x[idx1] = 1
        x[idx_1] = -1

    noise = np.sqrt(noise_var) * np.random.randn(M,)
    b = A @ x + noise
    return A, b, x

# OMP 算 法
def OMP(H, y, sparsity):
    y = y.flatten()
    """
    OMP算法的Python实现
        参数：
        A: 测量矩阵，形状为(m, n)
        y: 观测向量，形状为(m,)
        k: 稀疏度，即信号的非零元素个数
    返回： x: 重构的稀疏信号，形状为(n, 1)
    """
    N = H.shape[1]
    residual = y.copy()
    index_set = []
    theta = np.zeros(N)
    for _ in range(sparsity):
        correlations = H.T @ residual
        best_index = np.argmax(np.abs(correlations))
        index_set.append(best_index)
        phi_selected = H[:, index_set]
        theta_selected, _, _, _ = np.linalg.lstsq(phi_selected, y, rcond = None)
        for i, idx in enumerate(index_set):
            theta[idx] = theta_selected[i]
        residual = y - H @ theta
        if np.linalg.norm(residual) < 1e-6:
            break
    return theta

def OMP2(A, y, sparsity):
    y = y.reshape(-1,1)
    """
    OMP算法的Python实现
    参数：
        A: 测量矩阵，形状为(m, n)
        y: 观测向量，形状为(m, 1)
        k: 稀疏度，即信号的非零元素个数
    返回： x: 重构的稀疏信号，形状为(n, 1)
    """
    m, n = A.shape
    residual = y.copy()  # 初始化残差
    support = []  # 初始化支持集合

    for _ in range(sparsity):
        # 计算投影系数
        projections = np.abs(A.T @ residual)
        # 选择最相关的原子
        index = np.argmax(projections)
        support.append(index)
        # 更新估计信号
        x = np.linalg.lstsq(A[:, support], y, rcond = None)[0]
        # 更新残差
        residual = y - A[:, support] @ x
    # 构造稀疏信号
    x_sparse = np.zeros((n, 1))
    x_sparse[support] = x

    return x_sparse

# FISTA算法
def FISTA(H, y, x_true, lambda_ = 0.1, eta = 0.01, max_iter = 1000, tol = 1e-6):
    xhat = np.zeros_like(x_true)
    xhat_old = xhat.copy()
    t = 1
    cost_history = []
    mse_history = []

    for i in range(max_iter):
        # print(f"  {i}: {xhat.shape}")
        gradient = H.T @ (H @ xhat - y)
        xhat_temp = xhat - eta * gradient
        xhat_new = np.sign(xhat_temp) * np.maximum(np.abs(xhat_temp) - eta * lambda_, 0)

        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        xhat = xhat_new + ((t - 1) / t_new) * (xhat_new - xhat_old)

        xhat_old = xhat_new
        t = t_new

        cost = 0.5 * np.linalg.norm(y - H @ xhat) ** 2 + lambda_ * np.sum(np.abs(xhat))
        mse = np.linalg.norm(x_true - xhat)**2 / np.linalg.norm(x_true)**2

        cost_history.append(cost)
        mse_history.append(mse)

        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < tol:
            break
    return xhat, mse_history, cost_history



# from matlab
def AMP_Lasso(H, y, x_true, maxIter = 1000, lambda_ = 0.05, ):
    def damping(x, x_old, mes):
        x = mes * x + (1 - mes) * x_old
        x_old = x
        return x, x_old
    def mean_partial(R, sigma):
        N = len(R)
        tem = np.zeros((N, 1))
        tem[np.abs(R) > sigma] = 1
        mean_o = np.mean(tem)
        return mean_o

    x_true = x_true.flatten() # not must
    y = y.flatten() # not must
    M, N = H.shape
    delta = M/N
    xhat = np.zeros_like(x_true) # ((N, 1))
    gamma = 1
    Onsager = 0
    mes = 0.95
    z_old = 0
    mse_history = []
    cost_history = []
    for i in range(maxIter):
        z = y - H @ xhat + Onsager
        z, z_old = damping(copy.deepcopy(z), copy.deepcopy(z_old), mes)
        r = xhat + H.conjugate().T @ z
        xhat = np.sign(r) * np.maximum(np.abs(r) - lambda_ - gamma, 0)
        temp = mean_partial(xhat + H.conjugate().T @ z, lambda_ + gamma)
        Onsager = 1/delta * z * temp
        gamma = (lambda_ + gamma) / delta * temp

        cost = 0.5 * np.linalg.norm(y - H @ xhat) ** 2 + lambda_ * np.sum(np.abs(xhat))
        mse = np.linalg.norm(x_true - xhat)**2 / np.linalg.norm(x_true)**2

        cost_history.append(cost)
        mse_history.append(mse)
    return xhat, mse_history, cost_history


## AMP algorithm, converges slow,
def AMPforCS(A, y, x_true, max_iter = 100, lamda = 0.1, var_x = 1, epsilon = 0.2, lambda_ = 0.05,):
    def denoise(v, var_x, var_z, epsilon):
        term1 = (1-epsilon)*norm.pdf(v, 0, np.sqrt(var_z))
        term2 = epsilon*norm.pdf(v, 0, np.sqrt(var_x+var_z))
        xW = var_x / (var_x + var_z)*v # Wiener filter
        added_term = term1 + term2
        div_term = np.divide(term2, added_term)
        xhat = np.multiply(div_term, xW) # denoised version, x(t+1)

        # empirical derivative
        Delta = 0.0000000001 # perturbation
        term1_d = (1-epsilon) * norm.pdf((v+Delta), 0, np.sqrt(var_z))
        term2_d = epsilon * norm.pdf((v + Delta), 0, np.sqrt(var_x + var_z))
        xW2 = var_x / (var_x + var_z)*(v + Delta) # Wiener filter

        added_term = term1_d + term2_d
        mul_term = np.multiply(xW2, term2_d)
        xhat2 = np.divide(mul_term, added_term)
        d = (xhat2 - xhat)/Delta
        return xhat, d

    y = y.reshape(-1,1)
    M, N = A.shape
    delta = M/N          # measurement rate
    # initialization
    # mse = np.zeros((max_iter,1)) # store mean square error
    xt = np.zeros((N,1))# estimate of signal
    dt = np.zeros((N,1))# derivative of denoiser
    rt = np.zeros((M,1))# residual

    mse_history = []
    cost_history = []
    for _ in range(max_iter):
        # update residual
        rt = y - A @ xt + 1 / delta * np.mean(dt) * rt
        # compute pseudo-data
        vt = xt + A.T @ rt
        # estimate scalar channel noise variance estimator is due to Montanari
        var_t = np.mean(rt**2)
        # denoising
        xt1, dt = denoise(vt, var_x, var_t, epsilon)
        # damping step
        xt = lamda*xt1 + (1-lamda)*xt

        cost = 0.5 * np.linalg.norm(y - H @ xt) ** 2 + lambda_ * np.sum(np.abs(xt))
        mse = np.linalg.norm(x_true - xt)**2 / np.linalg.norm(x_true)**2

        cost_history.append(cost)
        mse_history.append(mse)
    return xt, mse_history, cost_history

## https://github.com/kuanhsieh/amp_cs
def opt_tuning_param(eps, limit=3):
    '''Find optimal tuning parameter for given sparsity ratio (epsilon) for the AMP algorithm.
    Equation on p8 of "Graphical Models Concepts in Compressed Sensing"
    Inputs
        eps: epsilon = K/N = delta*rho
        limit: Limits the search for optimal tuning parameter alpha
    '''
    def M_pm(alpha, eps):
        return eps*(1+alpha**2) + (1-eps)*(2*(1+alpha**2)*norm.cdf(-alpha)-2*alpha*norm.pdf(alpha))
    res   = minimize_scalar(M_pm, bracket=(0, limit), args = (eps))
    alpha = res.x
    return alpha
def AMPKuanCS(H, y, x_true, alpha, maxIter = 300, lambda_ = 0.05,):
    '''Approximate message passing (AMP) iteration with soft-thresholding denoiser.
    Inputs
        y: measurement vector (length M 1d np.array)
        A: sensing matrix     (M-by-N 2d np.array)
        x: signal estimate    (length N 1d np.array)
        z: residual           (length M 1d np.array)
        alpha: threshold tuning parameter
    Outputs
        x: signal estimate
        z: residual
    Note
        Need to initialise AMP iteration with
        x = np.zeros(N)
        z = y
    '''
    x_true = x_true.flatten() # not must
    y = y.flatten() # not must
    x = np.zeros_like(x_true)
    z = copy.deepcopy(y)
    M = len(y)
    mse_history = []
    cost_history = []

    for i in range(maxIter):
        # Estimate vector
        theta = alpha*np.sqrt(LA.norm(z)**2/M) # alpha*tau
        r = x + H.T @ z
        x =  np.sign(r) * np.maximum(np.abs(r) - theta, 0) #  soft_thresh

        # Calculate residual with the Onsager term
        b = LA.norm(x.flatten(), 0)/M
        z = y - H @ x + b*z

        cost = 0.5 * np.linalg.norm(y - H @ x) ** 2 + lambda_ * np.sum(np.abs(x))
        mse = np.linalg.norm(x_true - x)**2 / np.linalg.norm(x_true)**2

        cost_history.append(cost)
        mse_history.append(mse)
    # L = theta*(1 - b) # The last L is the actual lambda of the LASSO we're minimizing
    return x, mse_history, cost_history, z

## 凸优化算法
def prox_grad(H, y, x_true, stepsize, L, maxIter = 300, lambda_ = 0.05,):
    x_true = x_true.flatten() # not must
    y = y.flatten()  # not must
    x = np.zeros_like(x_true)
    mse_history = []
    cost_history = []
    for i in range(maxIter):
        z = x + stepsize * (H.T @ (y - H @ x))
        x = np.sign(z) * np.maximum(np.abs(z) - stepsize * L, 0) #  soft_thresh

        cost = 0.5 * np.linalg.norm(y - H @ x) ** 2 + lambda_ * np.sum(np.abs(x))
        mse = np.linalg.norm(x_true - x)**2 / np.linalg.norm(x_true)**2

        cost_history.append(cost)
        mse_history.append(mse)
    return x, mse_history, cost_history

## 凸优化算法
def nesterov(H, y, x_true, stepsize, L, maxIter = 300, lambda_ = 0.05,):
    x_true = x_true.flatten() # not must
    y = y.flatten()  # # not must
    x = np.zeros_like(x_true)
    theta = np.zeros_like(x_true)
    mse_history = []
    cost_history = []
    for it in range(maxIter):
        x_prev = x
        # Step 1: take gradient step
        z = theta + stepsize * (H.T @ (y - H @ theta))
        # Step 2: perform element-wise soft-thresholding
        x = np.sign(z) * np.maximum(np.abs(z) - stepsize * L, 0)
        # Step 3: update theta
        theta = x + it * (x - x_prev)/(it + 3)

        cost = 0.5 * np.linalg.norm(y - H @ x) ** 2 + lambda_ * np.sum(np.abs(x))
        mse = np.linalg.norm(x_true - x)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)
    return x, mse_history, cost_history

def OMPKuan(H, y, x_true, maxIter = 1000, lambda_ = 0.05,):
    x_true = x_true.flatten()
    y = y.flatten()
    '''Orthogonal matching pursuit iteration.
    Inputs
        y: measurement vector
        H: sensing matrix
        x_true: true x
    Outputs
        x: updated signal estimate
    '''
    x = np.zeros_like(x_true) # signal estimate
    z = copy.deepcopy(y)      # residual vector
    Omega = []                # index selections
    mse_history = []
    cost_history = []

    for it in range(maxIter):
        n_l = np.argmax(np.abs(H.T @ z) /  LA.norm(H, axis = 0, ))
        Omega.append(n_l)

        H_Omega = H[:, Omega]
        coef_vals, _, _, _ = LA.lstsq(H_Omega, y, rcond=None) # Able to parallelize using LA.lstsq?
        x[Omega] = coef_vals
        z = y - H_Omega @ coef_vals

        cost = 0.5 * np.linalg.norm(y - H @ x) ** 2 + lambda_ * np.sum(np.abs(x))
        mse = np.linalg.norm(x_true - x)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)

    return x, mse_history, cost_history

def CosOMP(H, y, x_true, K_est, maxIter = 300, lambda_ = 0.05,):
    x_true = x_true.flatten()
    y = y.flatten()
    """CoSaMP algorithm iteration.
    See "CoSaMP: Iterative Signal Recovery from Incomplete and Inaccurate Samples" by Needell and Tropp for more details. Notation follows paper.
    Inputs
        y: measurement vector
        H: sensing matrix
        x_true: true x
    Outputs
    """
    mse_history = []
    cost_history = []
    x = np.zeros_like(x_true) # signal estimate
    z = copy.deepcopy(y)      # residual vector

    for i in range(maxIter):
        Omega = np.argsort(np.abs(H.T @ z))[-2*K_est:] # Identify large components
        T = np.union1d(Omega, np.nonzero(x)[0])        # Merge supports
        b = np.zeros_like(x)
        b[T], _, _, _ =  LA.lstsq(H[:,T], y, rcond = None) # Might have to transpose the Phi after slicing...
        b[np.argsort(abs(b))[:-K_est]] = 0 # Prune to obtain next approximation
        x = np.copy(b)
        z = y - H @ x # Update current samples

        cost = 0.5 * np.linalg.norm(y - H @ x) ** 2 + lambda_ * np.sum(np.abs(x))
        mse = np.linalg.norm(x_true - x)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)

    return x, mse_history, cost_history

np.random.seed(42)
m, n = 500, 1000
maxIter = 300
rho = 0.03
lambda_ = 0.05
noise_varDB = 50
noise_var = 10**(-noise_varDB/10)
# noise_var = 0.02  # 0.00009 ~ 0.02

# H, y, x_true = generate_data(m, n, rho = rho, noise_var = noise_var)
# H, y, x_true = generate_data1(m, n, int(rho * n), noise_var = noise_var, x_choice = 1)
H, y, x_true = generate_data1(m, n, int(rho * n), noise_var = noise_var, x_choice = 0)
# 运行 AMP 算法
x_amp, mse_amp, cost_amp = AMP_Lasso(H, y, x_true, maxIter = maxIter, lambda_ = lambda_)
x_amp1, mse_amp1, cost_amp1 = AMPforCS(H, y, x_true, max_iter = 600, lamda = lambda_, epsilon = rho)

alpha_amp = opt_tuning_param(rho)
x_amp2, mse_amp2, cost_amp2, z_amp2 = AMPKuanCS(H, y, x_true, alpha_amp, maxIter = maxIter,  lambda_ =lambda_,)

# 运行 prox_grad / nesterov 算法
L = alpha_amp*LA.norm(z_amp2.flatten())*(1-LA.norm(x_amp2.flatten(), 0)/m)/np.sqrt(m) # Lambda
stepsize = np.real(1/(np.max(LA.eigvals(H.T @ H)))) # Step size

x_prox, mse_prox, cost_prox = prox_grad(H, y, x_true, stepsize, L, maxIter = 500 , lambda_ =lambda_,)
x_nesterov, mse_nesterov, cost_nesterov = nesterov(H, y, x_true, stepsize, L, maxIter = 500 , lambda_ =lambda_,)

# 运行 FISTA 算法
x_fista, mse_fista, cost_fista = FISTA(H, y, x_true, lambda_ = lambda_ ,  )

# 运行 OMP 算法
x_omp = OMP2(H, y, int(n*rho))
x_omp1, mse_omp, cost_omp = OMPKuan(H, y, x_true, maxIter = 300  , lambda_ =lambda_,)
K_est = np.nonzero(x_true)[0].shape[0]
x_omp2, mse_omp2, cost_omp2 = CosOMP(H, y, x_true, K_est, maxIter = 300 , lambda_ =lambda_,)


##
fig, axs = plt.subplots(1, 1, figsize=(12, 8), constrained_layout = True)
axs.semilogy(mse_amp, ls = '--', color = '#FF8C00', label = "AMP MSE" )
axs.semilogy(mse_amp1, ls = '--', color = '#00BFFF', label = "AMP denoise MSE" )
axs.semilogy(mse_amp2, ls = '--', color = '#8B0000', label = "AMP Kuan MSE" )
axs.semilogy(mse_prox, ls = '--', color = '#000000', label = "Prox MSE" )
axs.semilogy(mse_omp, ls = '--', color = '#28a428', label = "OMPKuan MSE" )
axs.semilogy(mse_omp2, ls = '--', color = '#00FF00', label = "CosOMP MSE" )
axs.semilogy(mse_nesterov, ls = '--', color = '#FF0000', label = "nesterov MSE" )
axs.semilogy(mse_fista, ls = '--', color = '#0000FF', label = "FISTA MSE" )
axs.legend(fontsize = 22)
axs.set_xlabel('Iteration')
axs.set_ylabel('MSE')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(12, 8), constrained_layout = True)
axs.semilogy(cost_amp, ls = '--', color = '#FF8C00', label = "AMP Cost" )
axs.semilogy(cost_amp1, ls = '--', color = '#00BFFF', label = "AMP denoise Cost" )
axs.semilogy(cost_amp2, ls = '--', color = '#8B0000', label = "AMP Kuan Cost" )
axs.semilogy(cost_prox, ls = '--', color = '#000000', label = "Prox Cost" )
axs.semilogy(cost_omp, ls = '--', color = '#28a428', label = "OMPKuan Cost" )
axs.semilogy(cost_omp2, ls = '--', color = '#00FF00', label = "CosOMP Cost" )
axs.semilogy(cost_nesterov, ls = '--', color = '#FF0000', label = "nesterov Cost" )
axs.semilogy(cost_fista, ls = '--', color = '#0000FF', label = "FISTA Cost" )
axs.legend(fontsize = 22)
axs.set_xlabel('Iteration')
axs.set_ylabel('Cost')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_true, linefmt = 'm--', markerfmt = 'mD',  label="True X", basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_omp, linefmt = 'g--', markerfmt = 'g*',  label="OMP X" , basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_omp1, linefmt = 'g--', markerfmt = 'g*',  label="OMP Kuan X" , basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_omp2, linefmt = 'g--', markerfmt = 'g*',  label="CosOMP Kuan X" , basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_fista, linefmt = 'c--', markerfmt = 'cv', label="FISTA X" , basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_amp, linefmt = 'b--', markerfmt = 'b^',  label="AMP X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_amp1, linefmt = 'b--', markerfmt = 'b^',  label="AMP denoise X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_amp2, linefmt = 'b--', markerfmt = 'b^',  label="AMP Kuan X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_prox, linefmt = 'k--', markerfmt = 'k^',  label="Prox X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_nesterov, linefmt = 'k--', markerfmt = 'k^',  label="nesterov X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()





























































































