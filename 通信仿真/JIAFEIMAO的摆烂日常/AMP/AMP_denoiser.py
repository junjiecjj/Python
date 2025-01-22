




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
def generate_data(M, N, K, noise_var = 0.0001, x_choice = 0):
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
    return x, mse_history, cost_history, z

def AMP_bg(H, y, x_true, eps, sig_pow = 1, maxIter = 300, lambda_ = 0.05, ):
    '''Approximate message passing (AMP) iteration with Bayes-optimal (MMSE) denoiser for signals with iid entries drawn from the Bernoulli-Gaussian distribution: probability (1-eps) equal to 0 and probability eps drawn from a Gaussian distribution with standard deviation v.
    Inputs
        y: measurement vector (length M 1d np.array)
        A: sensing matrix     (M-by-N 2d np.array)
        x: signal estimate    (length N 1d np.array)
        z: residual           (length M 1d np.array)
        eps: sparsity ratio (fraction of non-zero entries)
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
        term2 = 1 + (1-eps)/eps * np.sqrt(1+u) * np.exp(-(s**2/(2*tau**2))*u/(1+u))
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
    return x, mse_history, cost_history, z


np.random.seed(42)
#%%>>>>>>>>>>>>>>>>>>>>>>  1. Three-point distribution
N     = 1000  # dimension of signal
M     = 500   # num of measurements
K     = 30    # num of non-zero coefficients
lambda_ = 0.05
noise_varDB = 50
sigma = 10**(-noise_varDB/10) # # Noise standard deviation
sigma = 0.02  # 0.00009 ~ 0.02

sparsity = K/N
alpha    = opt_tuning_param(sparsity) # Find optimal alpha
iter_max = 20 # Max num of iterations

H, y, x_true = generate_data(M, N, K, noise_var = sigma, x_choice = 0)

## 原始AMP
x_amp, mse_amp, cost_amp, _ = AMPKuanCS(H, y, x_true, alpha, maxIter = iter_max , lambda_ = lambda_,)
x_amp1, mse_amp1, cost_amp1, _ = AMP_3pt(H, y, x_true, eps = sparsity, c = 1, maxIter = iter_max, lambda_ = lambda_,)

## 收敛性

## 原始信号
fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_true, linefmt = 'm--', markerfmt = 'mD',  label="True X", basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_amp, linefmt = 'm--', markerfmt = 'mD',  label="AMP X", basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_amp1, linefmt = 'm--', markerfmt = 'mD',  label="AMP3pt X", basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()



#%%>>>>>>>>>>>>>>>>>>>>>>  2. Bernoulli-Gaussian distribution

N     = 1000  # dimension of signal
M     = 500   # num of measurements
K     = 30    # num of non-zero coefficients
lambda_ = 0.05
noise_varDB = 50
sigma = 10**(-noise_varDB/10) # # Noise standard deviation
sigma = 0.01  # 0.00009 ~ 0.02

sparsity = K/N
alpha    = opt_tuning_param(sparsity) # Find optimal alpha
iter_max = 20 # Max num of iterations

H, y, x_true = generate_data(M, N, K, noise_var = sigma, x_choice = 1)

## 原始AMP
x_amp, mse_amp, cost_amp, _ = AMPKuanCS(H, y, x_true, alpha, maxIter = iter_max , lambda_ = lambda_,)
x_amp1, mse_amp1, cost_amp1, _ = AMP_bg(H, y, x_true, eps = sparsity, sig_pow = 1, maxIter = iter_max, lambda_ = lambda_,)

## 收敛性

## 原始信号
fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_true, linefmt = 'm--', markerfmt = 'mD',  label="True X", basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_amp, linefmt = 'm--', markerfmt = 'mD',  label="AMP X", basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_amp1, linefmt = 'm--', markerfmt = 'mD',  label="AMPbg X", basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()



























