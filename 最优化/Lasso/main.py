#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 21:15:25 2025

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


def ADMM4lasso(H, y, K, x_true, lambda_ = 0.05, maxIter = 1000, tol = 1e-6, ):
    P_half = 0.01
    c = 0.005
    Xk = np.zeros(XSize)
    Zk = np.zeros(XSize)
    Vk = np.zeros(XSize)

    X_opt_dst_steps = []
    X_dst_steps = []

    while True:
        Xk_new = np.dot( np.linalg.inv(A.T@A + c * np.eye(XSize, XSize)), c*Zk + Vk + np.dot(A.T, b) )

        # 软门限算子
        Zk_new = np.zeros(XSize)
        for i in range(XSize):
            if Xk_new[i] - Vk[i] / c < - P_half / c:
                Zk_new[i] = Xk_new[i] - Vk[i] / c + P_half / c
            elif Xk_new[i] - Vk[i] / c > P_half / c:
                Zk_new[i] = Xk_new[i] - Vk[i] / c - P_half / c

        Vk_new = Vk + c * (Zk_new - Xk_new)

        # print(np.linalg.norm(Xk_new - Xk, ord=2))

        X_dst_steps.append(np.linalg.norm(Xk_new - X, ord=2))
        X_opt_dst_steps.append(Xk_new)
        if np.linalg.norm(Xk_new - Xk, ord=2) < 1e-5:
            break
        else:
            Xk = Xk_new.copy()
            Zk = Zk_new.copy()
            Vk = Vk_new.copy()

    print(Xk)
    print(X)

    X_opt = X_opt_dst_steps[-1]

    for i, data in enumerate(X_opt_dst_steps):
        X_opt_dst_steps[i] = np.linalg.norm(data - X_opt, ord=2)
    plt.title("Distance")
    plt.plot(X_opt_dst_steps, label='X-opt-distance')
    plt.plot(X_dst_steps, label='X-real-distance')
    plt.legend()
    plt.show()
    return



np.random.seed(42)
M, N = 500, 1000
sparsity = 0.03
K = int(N * sparsity)
maxIter = 300
lambda_ = 0.05
noise_varDB = 50
noise_var = 10**(-noise_varDB/10)   ##  0.02  # 0.00009 ~ 0.02
H, y, x_true = generate_data(M, N, K, noise_var = noise_var, x_choice = 1)























































































































































