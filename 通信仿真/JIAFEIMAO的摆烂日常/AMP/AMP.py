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
    # y = y.reshape(-1,1)
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
    xhat = np.zeros((H.shape[1] , 1))
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

# 初始化输入数据 A 和 b
def generate_data(m, n, rho = 0.1, noise_var = 0.0001):
    A = np.random.randn(m, n) / np.sqrt(m)  # 随机生成一个 m x n 的矩阵
    x = (np.random.rand(n, 1) < rho) * np.random.randn(n, 1) / np.sqrt(rho)
    noise = np.sqrt(noise_var) * np.random.randn(m, 1)  # 降低噪声强度
    b = A @ x + noise
    return A, b, x

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

def AMP_Lasso(H, y, x_true, maxIter = 1000, lambda_ = 0.05, ):
    M, N = H.shape
    delta = M/N
    xhat = np.zeros((N, 1))
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


## AMP algorithm
def AMPforCS(A, y, x_true, max_iter = 100, lamda = 0.1, var_x = 1, epsilon = 0.2 ):
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


np.random.seed(42)
m, n = 512, 1024
maxIter = 300
rho = 0.05
lambda_ = 0.05
noise_varDB = 50
noise_var = 10**(-noise_varDB/10)

H, y, x_true = generate_data(m, n, rho = rho, noise_var = noise_var)
# 运行 AMP 算法
x_amp, mse_amp, cost_amp = AMP_Lasso(H, y, x_true, maxIter = maxIter, lambda_ = lambda_)
x_amp1, mse_amp1, cost_amp1 = AMPforCS(H, y, x_true, max_iter = 600, lamda = lambda_, epsilon = rho)


#  运行 FISTA 算法
x_fista, mse_fista, cost_fista = FISTA(H, y, x_true, lambda_ = lambda_)

# 运行 OMP 算法
x_omp = OMP2(H, y, int(n*rho))

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.semilogy(mse_amp, ls = '--', color = 'orange', label = "AMP MSE" )
axs.semilogy(mse_amp1, ls = '--', color = 'm', label = "AMP denoise MSE" )
axs.semilogy(mse_fista, ls = '--', color = 'b', label = "FISTA MSE" )
axs.legend()
axs.set_xlabel('Iteration')
axs.set_ylabel('MSE')
plt.show()
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.semilogy(cost_amp, ls = '--', color = 'orange', label = "AMP Cost" )
axs.semilogy(cost_amp1, ls = '--', color = 'm', label = "AMP denoise Cost" )
axs.semilogy(cost_fista, ls = '--', color = 'b', label = "FISTA Cost" )
axs.legend()
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
axs.stem(x_amp, linefmt = 'b--', markerfmt = 'b^',  label="AMP denoise X", basefmt='none' )
axs.legend()
axs.set_xlabel('Index')
axs.set_ylabel('Value')
plt.show()
plt.close()































































































