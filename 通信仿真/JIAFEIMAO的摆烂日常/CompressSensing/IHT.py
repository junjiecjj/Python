#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:35:21 2024

@author: jack

# https://blog.csdn.net/qq_44648285/article/details/143352383

"""


import numpy as np

def hard_thresholding(x, k):
    """保留x中最大的k个元素，其他元素设为零"""
    if k >= len(x):
        return x
    threshold = np.partition(np.abs(x), -k)[-k]
    return np.where(np.abs(x) >= threshold, x, 0)

def iterative_hard_thresholding(A, b, mu=0.001, k=50, T=1000, epsilon=1e-6):
    """
    稀疏迭代阈值算法（IHT）
    参数:
        A: 测量矩阵
        b: 测量向量
        mu: 步长
        k: 稀疏度
        T: 最大迭代次数
        epsilon: 收敛阈值
    返回:
        x: 重建信号
    """
    x = np.zeros(A.shape[1])
    for t in range(T):
        r = b - A @ x
        y = x + mu * (A.T @ r)
        x_new = hard_thresholding(y, k)
        if np.linalg.norm(x_new - x) < epsilon:
            print(f"收敛于迭代次数: {t+1}")
            break
        x = x_new
    return x

# https://blog.csdn.net/qq_51320133/article/details/137675664

def iht1(A, y, K, max_iters = 600, tol = 1e-9):
    """
    Iterative Hard Thresholding (IHT) algorithm.
    Parameters:
    A (numpy.ndarray): Observation matrix (m x n).
    y (numpy.ndarray): Observed signal (m x 1).
    K (int): Target sparsity level.
    max_iters (int, optional): Maximum number of iterations. Default is 100.
    tol (float, optional): Tolerance for stopping criterion (change in signal estimate). Default is 1e-9.
    Returns:
    numpy.ndarray: Estimated sparse signal (n x 1).
    """
    m, n = A.shape
    x = np.zeros(n)  # Initialize signal estimate
    # r = y.copy()  # Initialize residual

    for t in range(max_iters):
        # Step 2: Gradient update
        gradient = A.T @ (y - A @ x)

        # Step 3: Hard thresholding
        threshold = np.partition(np.abs(x + gradient), -K)[-K]
        x_new = np.sign(x + gradient) * np.maximum(np.abs(x + gradient) - threshold, 0)
        x_new /= np.linalg.norm(x_new)
        # Step 4: Check stopping criterion
        if np.linalg.norm(x - x_new) < tol:
            break
        x = x_new
    return x



# 示例使用
if __name__ == "__main__":
    np.random.seed(0)
    # 生成随机稀疏信号
    n = 1000
    k_true = 50
    x_true = np.zeros(n)
    non_zero_indices = np.random.choice(n, k_true, replace=False)
    x_true[non_zero_indices] = np.random.randn(k_true)

    ## 测量矩阵
    m = 200
    A = np.random.randn(m, n)

    ## 测量向量
    y = A @ x_true

    ## 使用IHT恢复信号
    x_recovered = iterative_hard_thresholding(A, y, mu=0.001, k=50, T=1000, epsilon=1e-6)
    # x_recovered = iht1(A, y, k_true,)
    ## 评估恢复效果
    recovery_error = np.linalg.norm(x_recovered - x_true) / np.linalg.norm(x_true)
    print(f"恢复误差: {recovery_error:.6f}")


































































