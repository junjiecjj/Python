#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:04:17 2024
https://blog.csdn.net/qq_44648285/article/details/143363030
@author: jack
"""

import numpy as np
import copy


def soft_thresholding(x, tau):
    """
    软阈值函数，用于保持信号的稀疏性
    """
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)

def AMPforCS(A, y, tau, T = 100, epsilon = 1e-6, verbose = False):
    """
    近似消息传递算法（AMP） approximate_message_passing
    参数:
        A: 测量矩阵 (m x n)
        b: 测量向量 (m,)
        tau: 阈值参数
        T: 最大迭代次数
        epsilon: 收敛阈值
        verbose: 是否打印迭代信息
    返回: x: 重建信号 (n,)
    """
    m, n = A.shape
    delta = m / n
    x = np.zeros(n)
    z = copy.deepcopy(y)
    for t in range(T):
        # 信号更新
        x_temp = A.T @ z + x
        x_new = soft_thresholding(x_temp, tau)
        # 计算收缩函数的导数
        eta_prime = (x_temp > tau) | (x_temp < -tau)
        eta_prime = eta_prime.astype(float)
        eta_prime_mean = np.mean(eta_prime)
        # 残差更新
        z_new = y - A @ x_new + z * eta_prime_mean / delta
        # 收敛判定
        diff = np.linalg.norm(x_new - x)
        if verbose:
            print(f"迭代次数: {t+1}, 变化量: {diff:.6f}")
            if diff < epsilon:
                print(f"算法在迭代次数 {t+1} 时收敛。")
                break
        # 更新信号和残差
        x = copy.deepcopy(x_new)
        z = copy.deepcopy(z_new)
    return x

# 示例使用
# if __name__ == "__main__":
np.random.seed(10)
# 生成随机稀疏信号
n = 100
K = 5
x = np.zeros(n)
non_zero_indices = np.random.choice(n, K, replace=False)
x[non_zero_indices] = np.random.randn(K)

# 测量矩阵
m = 50
A = np.random.randn(m, n)

# 测量向量
y = A @ x

# 设置阈值参数
tau = 1/(K/n)  # 可以根据实际情况调整

# 使用AMP恢复信号
x_hat = AMPforCS(A, y, tau, T = 100, epsilon = 1e-6, verbose = True)

# 评估恢复效果
recovery_error = np.linalg.norm(x_hat - x) / np.linalg.norm(x)
print(f"恢复误差: {recovery_error:.6f}")


# x = np.linspace(-2, 2, 100)
# y = soft_thresholding(x, 0.5)
# plot(x,y)




















