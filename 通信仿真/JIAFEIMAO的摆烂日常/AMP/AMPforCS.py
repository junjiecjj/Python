#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:04:17 2024
https://blog.csdn.net/qq_44648285/article/details/143363030
@author: jack
"""


import numpy as np
from matplotlib import pyplot as plt
import copy
from scipy.stats import norm

# 初始化输入数据 A 和 b
def generate_data(n, m, sparsity = 0.1, scale_factor = 1000.0, noise_var = 0.00001):
    A = np.random.randn(m, n)/np.sqrt(m)  # 随机生成一个 m x n 的矩阵
    x = np.zeros(n)
    non_zero_indices = np.random.choice(n, size=int(n * sparsity), replace=False)
    x[non_zero_indices] = scale_factor * np.random.randn(len(non_zero_indices))  # 放大信号
    b = A @ x + noise_var * np.random.randn(m)  # 降低噪声强度
    return A, b, x

# OMP 算法
def OMP(phi, y, sparsity):
    """
    OMP算法的Python实现
        参数：
        A: 测量矩阵，形状为(m, n)
        y: 观测向量，形状为(m,)
        k: 稀疏度，即信号的非零元素个数
    返回： x: 重构的稀疏信号，形状为(n, 1)
    """
    N = phi.shape[1]
    residual = y.copy()
    index_set = []
    theta = np.zeros(N)
    for _ in range(sparsity):
        correlations = phi.T @ residual
        best_index = np.argmax(np.abs(correlations))
        index_set.append(best_index)
        phi_selected = phi[:, index_set]
        theta_selected, _, _, _ = np.linalg.lstsq(phi_selected, y, rcond = None)
        for i, idx in enumerate(index_set):
            theta[idx] = theta_selected[i]
        residual = y - phi @ theta
        if np.linalg.norm(residual) < 1e-6:
            break
    return theta

# ISTA算法
def ISTA(X, y, lambda_ = 0.1, eta = 0.001, max_iter = 1000, tol = 1e-6):
    beta = np.zeros(X.shape[1])
    cost_history = []
    mse_history = []

    for i in range(max_iter):
        gradient = X.T @ (X @ beta - y)
        beta_temp = beta - eta * gradient
        beta = np.sign(beta_temp) * np.maximum(np.abs(beta_temp) - eta * lambda_, 0)
        cost = 0.5 * np.linalg.norm(y - X @ beta) ** 2 + lambda_ * np.sum(np.abs(beta))
        # mse = mean_squared_error(true_coef, beta)

        cost_history.append(cost)
        # mse_history.append(mse)

        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < tol:
            break
    return beta, cost_history, mse_history

# FISTA算法
def FISTA(X, y, lambda_ = 0.1, eta = 0.001, max_iter = 1000, tol = 1e-6):
    beta = np.zeros(X.shape[1])
    beta_old = beta.copy()
    t = 1
    cost_history = []
    mse_history = []

    for i in range(max_iter):
        gradient = X.T @ (X @ beta - y)
        beta_temp = beta - eta * gradient
        beta_new = np.sign(beta_temp) * np.maximum(np.abs(beta_temp) - eta * lambda_, 0)

        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        beta = beta_new + ((t - 1) / t_new) * (beta_new - beta_old)

        beta_old = beta_new
        t = t_new

        cost = 0.5 * np.linalg.norm(y - X @ beta) ** 2 + lambda_ * np.sum(np.abs(beta))
        # mse = mean_squared_error(true_coef, beta)

        cost_history.append(cost)
        # mse_history.append(mse)

        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < tol:
            break
    return beta, cost_history, mse_history

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


## AMP algorithm
def AMPforCS(A, y, real_x, max_iter = 100, lamda = 0.1, var_x = 1, epsilon = 0.2 ):
    y = y.reshape(-1,1)
    M, N = A.shape
    delta = M/N          # measurement rate
    # initialization
    mse = np.zeros((max_iter,1)) # store mean square error
    xt = np.zeros((N,1))# estimate of signal
    dt = np.zeros((N,1))# derivative of denoiser
    rt = np.zeros((M,1))# residual
    for iter in range(0, max_iter):
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
        mse[iter] = np.mean((xt - real_x)**2)
    return xt , mse


np.random.seed(42)

# Example usage
m, n = 50, 100
sparsity = 0.05
A, b, x_true = generate_data(n, m, sparsity, scale_factor = 1.0)

# 运行 AMP 算法
x_amp, _ = AMPforCS(A, b, x_true, epsilon = sparsity)

# 运行 OMP 算法
x_omp = OMP(A, b, int(n*sparsity))

#  运行 ISTA 算法
x_ista, _, _ = ISTA(A, b, )

#  运行 FISTA 算法
x_fista, _, _ = FISTA(A, b, )


# 打印恢复的信号
# print(f'Recovered signal: {x_recovered}')
# print(f'True signal: {x_true}')
fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_true, linefmt = 'm--', markerfmt = 'mD',  label="真实系数", basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_omp, linefmt = 'g--', markerfmt = 'g*',  label="OMP估计系数" , basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_ista, linefmt = 'k--', markerfmt = 'kh', label="ISTA估计系数" , basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_fista, linefmt = 'c--', markerfmt = 'cv', label="FISTA估计系数" , basefmt='none')
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('value')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.stem(x_amp, linefmt = 'b--', markerfmt = 'b^',  label="AMP估计系数", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('value')
plt.show()
plt.close()
































































































































