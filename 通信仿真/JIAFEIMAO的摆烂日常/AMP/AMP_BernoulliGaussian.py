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
from sklearn.metrics import mean_squared_error


# 初始化输入数据 A 和 b
def generate_data(M, N, K, noise_var = 0.0001, x_choice = 0, Achoice = 0):
    """
      N: Signal dimension
      M: Number of measurements
      K: Sparsity, number of non-zero entries in signal vector
      A_choice: type of sensing matrix
          0: With iid normal entries N(0,1/M)
          1: With iid entries in {+1/sqrt(M), -1/sqrt(M)} with uniform probability
      x_choice: type of signal
          0: Elements in {+1, 0, -1}, with K +-1's with uniform probability of +1 and -1's
          1: K non-zero entries drawn from the standard normal distribution
    """
    if Achoice == 0:
        A = np.random.randn(M, N) / np.sqrt(M)  # 随机生成一个 m x n 的矩阵
    elif Achoice == 1:
        A = np.random.rand(M,N)
        A = (A<0.5)/np.sqrt(M)
        A[np.logical_not(A)] = -1/np.sqrt(M)
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

# OMP 算法
def OMP(H, y, K, x_true, lambda_ = 0.05, maxIter = 1000, tol = 1e-6, ):
    y = y.flatten()
    """
    OMP算法的Python实现
        参数：
        A: 测量矩阵，形状为(m, n)
        y: 观测向量，形状为(m,)
        k: 稀疏度，即信号的非零元素个数
    返回： x: 重构的稀疏信号，形状为(n, 1)
    """
    M, N = H.shape
    residual = y.copy()
    support = []
    x = np.zeros(N)
    cost_history = []
    mse_history = []

    for _ in range(maxIter):
        best_index = np.argmax(np.abs(H.T @ residual))
        support.append(best_index)
        theta_selected = np.linalg.lstsq(H[:, support], y, rcond = None)[0]
        for i, idx in enumerate(support):
            x[idx] = theta_selected[i]
        residual = y - H @ x
        # if np.linalg.norm(residual) < 1e-6:
            # break
        cost = 0.5 * np.linalg.norm(y - H @ x) ** 2 + lambda_ * np.sum(np.abs(x))
        mse = np.linalg.norm(x_true - x)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)
    return x, mse_history, cost_history

def OMP1(H, y, K, x_true, lambda_ = 0.05, maxIter = 1000, tol = 1e-6, ):
    y = y.flatten()
    """
    OMP算法的Python实现
    参数：
        A: 测量矩阵，形状为(m, n)
        y: 观测向量，形状为(m, 1)
        k: 稀疏度，即信号的非零元素个数
    返回： x: 重构的稀疏信号，形状为(n, 1)
    """
    M, N = H.shape
    residual = y.copy()  # 初始化残差
    support = []  # 初始化支持集合
    cost_history = []
    mse_history = []

    for _ in range(maxIter):
        # 计算投影系数
        projections = np.abs(H.T @ residual)
        # 选择最相关的原子
        index = np.argmax(projections)
        support.append(index)
        # 更新估计信号
        x = np.linalg.lstsq(H[:, support], y, rcond = None)[0]
        # 更新残差
        residual = y - H[:, support] @ x

        # 构造稀疏信号
        x_sparse = np.zeros((N,))
        x_sparse[support] = x

        cost = 0.5 * np.linalg.norm(y - H @ x_sparse) ** 2 + lambda_ * np.sum(np.abs(x_sparse))
        mse = np.linalg.norm(x_true - x_sparse)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)
    return x_sparse, mse_history, cost_history

# ISTA算法
def ISTA(H, y, x_true, lambda_ = 0.05, eta = 0.1, maxIter = 1000, tol = 1e-8,  ):
    xhat = np.zeros(H.shape[1])
    cost_history = []
    mse_history = []

    for i in range(maxIter):
        gradient = H.T @ (H @ xhat - y)
        beta_temp = xhat - eta * gradient
        xhat = np.sign(beta_temp) * np.maximum(np.abs(beta_temp) - eta * lambda_, 0)

        cost = 0.5 * np.linalg.norm(y - H @ xhat) ** 2 + lambda_ * np.sum(np.abs(xhat))
        mse = np.linalg.norm(x_true - xhat)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)
        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < tol:
            break
    return xhat, mse_history, cost_history

# FISTA算法
def FISTA(H, y, x_true, lambda_ = 0.1, eta = 0.01, maxIter = 1000, tol = 1e-6, ):
    xhat = np.zeros_like(x_true)
    xhat_old = xhat.copy()
    t = 1
    cost_history = []
    mse_history = []

    for i in range(maxIter):
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
def AMP_tutorial(H, y, x_true, lambda_ = 0.05, maxIter = 1000, ):
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
def AMPforCS(H, y, x_true, lambda_ = 0.05, maxIter = 100, var_x = 1, sparsity = 0.1, ):
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
    M, N = H.shape
    delta = M/N          # measurement rate

    ## initialization
    xhat = np.zeros((N,1))# estimate of signal
    dt = np.zeros((N,1))# derivative of denoiser
    rt = np.zeros((M,1))# residual

    mse_history = []
    cost_history = []
    for _ in range(maxIter):
        # update residual
        rt = y - H @ xhat + 1 / delta * np.mean(dt) * rt
        # compute pseudo-data
        vt = xhat + H.T @ rt
        # estimate scalar channel noise variance estimator is due to Montanari
        var_t = np.mean(rt**2)
        # denoising
        xt1, dt = denoise(vt, var_x, var_t, sparsity)
        # damping step
        xhat = lambda_*xt1 + (1-lambda_)*xhat

        cost = 0.5 * np.linalg.norm(y - H @ xhat) ** 2 + lambda_ * np.sum(np.abs(xhat))
        mse = np.linalg.norm(x_true - xhat)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)
    return xhat, mse_history, cost_history


### AMP with Bayes-optimal denoiser for different signals
def AMP_3pt(H, y, x_true, eps, c = 1, maxIter = 300, lambda_ = 0.05, ):
    '''Approximate message passing (AMP) iteration with Bayes-optimal (MMSE) denoiser for signals with iid entries drawn from the 3-point distribution: probability (1-eps) equal to 0 and probability eps/2 equal to each +c and -c.
    Inputs
        y: measurement vector (length M 1d np.array)
        H: sensing matrix     (M-by-N 2d np.array)
        x: signal estimate    (length N 1d np.array)
        z: residual           (length M 1d np.array)
        eps: sparsity ratio (fraction of non-zero entries)
        c: the values of the non-zero entries of the signal equal to +c and -c
    Outputs:
    Note
        Need to initialise AMP iteration with x = np.zeros(N), z = y
    '''
    M, N = np.shape(H)
    x = np.zeros_like(x_true)
    z = copy.deepcopy(y)
    mse_history = []
    cost_history = []

    for i in range(maxIter):
        tau = np.sqrt(np.mean(z**2)) # Estimate of effective noise std deviation
        # Estimate vector
        s = x + H.T @ z # Effective (noisy) observation of signal x
        u = s*c / tau**2       # Temporary variable
        top = c * eps * np.sinh(u, dtype = np.float128)
        bot = eps * np.cosh(u, dtype = np.float128) + (1 - eps) * np.exp(c**2/(2*tau**2), dtype = np.float128)
        x   = (top / bot).astype(np.float128)
        # Calculate residual with the Onsager term
        eta_der = x * (c/np.tanh(u) - x) / tau**2
        b = (N/M) * np.mean(eta_der)
        z = y - H @ x + b*z

        cost = 0.5 * np.linalg.norm(y - H @ x) ** 2 + lambda_ * np.sum(np.abs(x))
        mse = np.linalg.norm(x_true - x)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)
    return x, mse_history, cost_history

def AMP_bg(H, y, x_true, sparsity, sig_pow = 1, maxIter = 300, lambda_ = 0.05, ):
    '''Approximate message passing (AMP) iteration with Bayes-optimal (MMSE) denoiser for signals with iid entries drawn from the Bernoulli-Gaussian distribution: probability (1-eps) equal to 0 and probability eps drawn from a Gaussian distribution with standard deviation v.
    Inputs
        y: measurement vector (length M 1d np.array)
        A: sensing matrix     (M-by-N 2d np.array)
        x: signal estimate    (length N 1d np.array)
        z: residual           (length M 1d np.array)
        sparsity: sparsity ratio (fraction of non-zero entries)
        v: the standard deviation of the non-zero entries of the signal which
           are drawn from a Gaussian distribution
    Outputs
        x: signal estimate
        z: residual
    Note
        Need to initialise AMP iteration with
        x = np.zeros(N)
        z = y
    '''
    M, N = np.shape(H)
    x = np.zeros_like(x_true)
    z = copy.deepcopy(y)
    mse_history = []
    cost_history = []

    for i in range(maxIter):
        tau = np.sqrt(np.mean(z**2)) # Estimate of effective noise std deviation
        # Estimate vector
        s = x + H.T @ z # Effective (noisy) observation of signal x
        u = sig_pow / tau**2       # Temporary variable
        term1 = 1 + tau**2/sig_pow
        term2 = 1 + (1-sparsity)/sparsity * np.sqrt(1+u) * np.exp(-(s**2/(2*tau**2))*u/(1+u))
        denom = term1 * term2
        x = s / denom
        # Calculate residual with the Onsager term
        eta_der = (1/denom) + (x/tau**2) * (s/(1+tau**2/sig_pow) - x)
        b = (N/M) * np.mean(eta_der)
        z = y - H @ x + b*z

        cost = 0.5 * np.linalg.norm(y - H @ x) ** 2 + lambda_ * np.sum(np.abs(x))
        mse = np.linalg.norm(x_true - x)**2 / np.linalg.norm(x_true)**2
        cost_history.append(cost)
        mse_history.append(mse)
    return x, mse_history, cost_history


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
    res = minimize_scalar(M_pm, bracket=(0, limit), args = (eps))
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
    xhat = np.zeros_like(x_true)
    z = copy.deepcopy(y)
    M = len(y)
    mse_history = []
    cost_history = []

    for i in range(maxIter):
        # Estimate vector
        theta = alpha*np.sqrt(LA.norm(z)**2/M) # alpha*tau
        r = xhat + H.T @ z
        xhat =  np.sign(r) * np.maximum(np.abs(r) - theta, 0) #  soft_thresh

        # Calculate residual with the Onsager term
        b = LA.norm(xhat.flatten(), 0)/M
        z = y - H @ xhat + b*z

        cost = 0.5 * np.linalg.norm(y - H @ xhat) ** 2 + lambda_ * np.sum(np.abs(xhat))
        mse = np.linalg.norm(x_true - xhat)**2 / np.linalg.norm(x_true)**2

        cost_history.append(cost)
        mse_history.append(mse)
    # L = theta*(1 - b) # The last L is the actual lambda of the LASSO we're minimizing
    return xhat, mse_history, cost_history, z

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

def CoSaMP(H, y, x_true, K_est, maxIter = 300, lambda_ = 0.05,):
    x_true = x_true.flatten()
    y = y.flatten()
    """CoSaMP algorithm iteration. 压缩采样匹配追踪.
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
M, N = 500, 1000
sparsity = 0.03
K = int(N * sparsity)
maxIter = 300
lambda_ = 0.05
noise_varDB = 50
noise_var = 10**(-noise_varDB/10)   ##  0.02  # 0.00009 ~ 0.02
H, y, x_true = generate_data(M, N, K, noise_var = noise_var, x_choice = 1)

##>>>>>>>>>>>>>> 运行 AMP 算法
x_amp_tuto, mse_amp_tuto, cost_amp_tuto = AMP_tutorial(H, y, x_true, maxIter = maxIter, lambda_ = lambda_)
x_amp_cs, mse_amp_cs, cost_amp_cs = AMPforCS(H, y, x_true, maxIter = maxIter, lambda_ = lambda_, sparsity = sparsity)

alpha_amp = opt_tuning_param(sparsity)
x_amp_Ku, mse_amp_Ku, cost_amp_Ku, z_amp_Ku = AMPKuanCS(H, y, x_true, alpha_amp, maxIter = maxIter,  lambda_ =lambda_,)
# x_amp_3pt, mse_amp_3pt, cost_amp_3pt = AMP_3pt(H, y, x_true, eps = sparsity, c = 1, maxIter = maxIter, lambda_ = lambda_,)
x_amp_bg, mse_amp_bg, cost_amp_bg = AMP_bg(H, y, x_true, sparsity = sparsity, sig_pow = 1, maxIter = maxIter, lambda_ = lambda_,)

##>>>>>>>>>>>>>> 运行 prox_grad / nesterov 算法
L = alpha_amp*LA.norm(z_amp_Ku.flatten())*(1-LA.norm(x_amp_Ku.flatten(), 0)/M)/np.sqrt(M) # Lambda
stepsize = np.real(1/(np.max(LA.eigvals(H.T @ H)))) # Step size

x_prox, mse_prox, cost_prox = prox_grad(H, y, x_true, stepsize, L, maxIter = maxIter , lambda_ =lambda_,)
x_nesterov, mse_nesterov, cost_nesterov = nesterov(H, y, x_true, stepsize, L, maxIter = maxIter , lambda_ =lambda_,)

##>>>>>>>>>>>>>> 运行 FISTA 算法
x_ista, mse_ista, cost_ista = ISTA(H, y, x_true, lambda_ = lambda_,  maxIter = maxIter,  )
x_fista, mse_fista, cost_fista = FISTA(H, y, x_true, lambda_ = lambda_ , maxIter = maxIter, )

##>>>>>>>>>>>>>> 运行 OMP 算法
x_omp, mse_omp, cost_omp = OMP(H, y, K, x_true, maxIter = maxIter,)
x_omp1, mse_omp1, cost_omp1 = OMP1(H, y, K, x_true, maxIter = maxIter,)
x_omp_Ku, mse_omp_Ku, cost_omp_Ku = OMPKuan(H, y, x_true, maxIter = maxIter  , lambda_ =lambda_,)
x_omp_CoSaMP, mse_omp_CoSaMP, cost_omp_CoSaMP = CoSaMP(H, y, x_true, K, maxIter = maxIter , lambda_ =lambda_,)

##>>>>>>>>>>>>>> Figs
fig, axs = plt.subplots(1, 1, figsize=(12, 8), constrained_layout = True)
axs.semilogy(mse_amp_tuto, ls = '--', marker = 'o', ms = '12', markevery = 50, label = "AMP_tutorial MSE" )
axs.semilogy(mse_amp_cs, ls = '-',lw = 3, marker = 'v', ms = '12', markevery = 50, label = "AMPforCS MSE" )
axs.semilogy(mse_amp_Ku, ls = '--', marker = '^', ms = '12', markevery = 50, label = "AMPKuanCS MSE" )
# axs.semilogy(mse_amp_3pt, ls = '--', marker = '1', ms = '12', markevery = 50, label = "AMP_3pt MSE" )
axs.semilogy(mse_amp_bg, ls = '--', marker = '2', ms = '12', markevery = 50, label = "AMP_bg MSE" )
axs.semilogy(mse_prox, ls = '--', marker = '*', ms = '12', markevery = 50, label = "prox_grad MSE" )
axs.semilogy(mse_nesterov, ls = '--', marker = '>', ms = '12', markevery = 50, label = "nesterov MSE" )
axs.semilogy(mse_ista, ls = '--', lw = 3, marker = '<', ms = '12', markevery = 50,  label = "ISTA MSE" )
axs.semilogy(mse_fista, ls = '--', marker = 's', ms = '12', markevery = 50,  label = "FISTA MSE" )
axs.semilogy(mse_omp, ls = '--', marker = 'p', ms = '12', markevery = 50, label = "OMP" )
axs.semilogy(mse_omp1, ls = '--', marker = 'h', ms = '12', markevery = 50, label = "OMP1 MSE" )
axs.semilogy(mse_omp_Ku, ls = '--', marker = 'd', ms = '12', markevery = 50, label = "OMPKuan MSE" )
axs.semilogy(mse_omp_CoSaMP, ls = '--', marker = 'x', ms = '12', markevery = 50, label = "CoSaMP MSE" )
axs.legend(fontsize = 22)
axs.set_xlabel('Iteration')
axs.set_ylabel('MSE')
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(12, 8), constrained_layout = True)
axs.semilogy(cost_amp_tuto, ls = '--', marker = 'o', ms = '12', markevery = 50, label = "AMP_tutorial Cost" )
axs.semilogy(cost_amp_cs, ls = '--', marker = 'v', ms = '12', markevery = 50, label = "AMPforCS Cost" )
axs.semilogy(cost_amp_Ku, ls = '--', marker = '^', ms = '12', markevery = 50, label = "AMPKuanCS Cost" )
# axs.semilogy(cost_amp_3pt, ls = '--', marker = '1', ms = '12', markevery = 50, label = "AMP_3pt Cost" )
axs.semilogy(cost_amp_bg, ls = '--', marker = '2', ms = '12', markevery = 50, label = "AMP_bg Cost" )
axs.semilogy(cost_prox, ls = '--', marker = '*', ms = '12', markevery = 50, label = "prox_grad Cost" )
axs.semilogy(cost_nesterov, ls = '--', marker = '>', ms = '12', markevery = 50,  label = "nesterov Cost" )
axs.semilogy(cost_ista, ls = '-',lw = 3, marker = '<', ms = '12', markevery = 50,  label = "ISTA Cost" )
axs.semilogy(cost_fista, ls = '--', marker = 's', ms = '12', markevery = 50, label = "FISTA Cost" )
axs.semilogy(cost_omp, ls = '--',marker = 'p', ms = '12', markevery = 50,   label = "OMP Cost" )
axs.semilogy(cost_omp1, ls = '--',marker = 'h', ms = '12', markevery = 50,  label = "OMP1 Cost" )
axs.semilogy(cost_omp_Ku, ls = '--', marker = 'd', ms = '12', markevery = 50, label = "OMPKuan Cost" )
axs.semilogy(cost_omp_CoSaMP, ls = '--',marker = 'x', ms = '12', markevery = 50, label = "CoSaMP Cost" )
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
axs.stem(x_amp_tuto, linefmt = 'g--', markerfmt = 'g*',  label="AMP_tutorial X" , basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_amp_cs, linefmt = 'g--', markerfmt = 'g*',  label="AMPforCS X" , basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_amp_Ku, linefmt = 'g--', markerfmt = 'g*',  label="AMPKuanCS X" , basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()

# fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
# axs.stem(x_amp_3pt, linefmt = 'g--', markerfmt = 'g*',  label="AMP_3pt X" , basefmt='none')
# axs.legend()
# axs.set_xlabel('Index')
# axs.set_ylabel('Value')
# plt.show()
# plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_amp_bg, linefmt = 'g--', markerfmt = 'g*',  label="AMP_bg X" , basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()



fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_prox, linefmt = 'c--', markerfmt = 'cv', label="prox_grad X" , basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_nesterov, linefmt = 'b--', markerfmt = 'b^',  label="nesterov X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_ista, linefmt = 'b--', markerfmt = 'b^',  label="ISTA X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_fista, linefmt = 'b--', markerfmt = 'b^',  label="FISTA X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_omp, linefmt = 'k--', markerfmt = 'k^',  label="OMP X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_omp1, linefmt = 'k--', markerfmt = 'k^',  label="OMP1 X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_omp_Ku, linefmt = 'k--', markerfmt = 'k^',  label="OMPKuan X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_omp_CoSaMP, linefmt = 'k--', markerfmt = 'k^',  label="CoSaMP X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()



























































































