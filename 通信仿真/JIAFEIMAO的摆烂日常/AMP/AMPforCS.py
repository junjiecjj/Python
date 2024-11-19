#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:04:17 2024
https://blog.csdn.net/qq_44648285/article/details/143363030
@author: jack
"""

import numpy as np
import copy


# def soft_thresholding(x, tau):
#     """
#     软阈值函数，用于保持信号的稀疏性
#     """
#     return np.sign(x) * np.maximum(np.abs(x) - tau, 0)

# def AMPforCS(A, y, tau, T = 100, epsilon = 1e-6, verbose = False):
#     """
#     近似消息传递算法（AMP） approximate_message_passing
#     参数:
#         A: 测量矩阵 (m x n)
#         b: 测量向量 (m,)
#         tau: 阈值参数
#         T: 最大迭代次数
#         epsilon: 收敛阈值
#         verbose: 是否打印迭代信息
#     返回: x: 重建信号 (n,)
#     """
#     m, n = A.shape
#     delta = m / n
#     x = np.zeros(n)
#     z = copy.deepcopy(y)
#     for t in range(T):
#         # 信号更新
#         x_temp = A.T @ z + x
#         x_new = soft_thresholding(x_temp, tau)
#         # 计算收缩函数的导数
#         eta_prime = (x_temp > tau) | (x_temp < -tau)
#         eta_prime = eta_prime.astype(float)
#         eta_prime_mean = np.mean(eta_prime)
#         # 残差更新
#         z_new = y - A @ x_new + z * eta_prime_mean / delta
#         # 收敛判定
#         diff = np.linalg.norm(x_new - x)
#         if verbose:
#             print(f"迭代次数: {t+1}, 变化量: {diff:.6f}")
#             if diff < epsilon:
#                 print(f"算法在迭代次数 {t+1} 时收敛。")
#                 break
#         # 更新信号和残差
#         x = copy.deepcopy(x_new)
#         z = copy.deepcopy(z_new)
#     return x

# # 示例使用
# # if __name__ == "__main__":
# np.random.seed(10)
# # 生成随机稀疏信号
# n = 100
# K = 5
# x = np.zeros(n)
# non_zero_indices = np.random.choice(n, K, replace=False)
# x[non_zero_indices] = np.random.randn(K)

# # 测量矩阵
# m = 50
# A = np.random.randn(m, n)

# # 测量向量
# y = A @ x

# # 设置阈值参数
# tau = 1/(K/n)  # 可以根据实际情况调整

# # 使用AMP恢复信号
# x_hat = AMPforCS(A, y, tau, T = 100, epsilon = 1e-6, verbose = True)

# # 评估恢复效果
# recovery_error = np.linalg.norm(x_hat - x) / np.linalg.norm(x)
# print(f"恢复误差: {recovery_error:.6f}")


# # x = np.linspace(-2, 2, 100)
# # y = soft_thresholding(x, 0.5)
# # plot(x,y)


import numpy as np
from matplotlib import pyplot as plt

# 初始化输入数据 A 和 b
def generate_data(n, m, sparsity=0.1, scale_factor=1000.0):
    A = np.random.randn(m, n)  # 随机生成一个 m x n 的矩阵
    x = np.zeros(n)
    non_zero_indices = np.random.choice(n, size=int(n * sparsity), replace=False)
    x[non_zero_indices] = scale_factor * np.random.randn(len(non_zero_indices))  # 放大信号
    b = A @ x + 0.0001 * np.random.randn(m)  # 降低噪声强度
    return A, b, x


# AMP算法
def amp(A, b, max_iter=50, threshold=1e-6, alpha=1e-4, regularization_strength=0.1):
    m, n = A.shape
    x = np.zeros(n)  # 初始化信号
    r = b - A @ x  # 初始残差
    signal_change = []
    residual_change = []

    for iter in range(max_iter):
        # 计算新的信号
        x_new = A.T @ r + x
        # 软阈值操作
        x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - regularization_strength, 0)
        # 对 x_new 进行裁剪，避免数值过大
        x_new = np.sign(x_new) * np.minimum(np.abs(x_new), 1e3)

        r_new = b - A @ x_new  # 更新残差

        # 自适应调整 alpha（学习率）
        if iter % 50 == 0:  # 每50轮迭代调整一次学习率
            alpha = alpha * 0.95  # 使用更慢的学习率衰减

        # 更新信号：x_new是加权后的新估计
        x_new = alpha * x_new + (1 - alpha) * x  # 更新信号

        # 计算变化量
        signal_change.append(np.linalg.norm(x_new - x))
        residual_change.append(np.linalg.norm(r_new - r))

        if signal_change[-1] < threshold and residual_change[-1] < threshold:
            print(f'Converged at iteration {iter + 1}')
            break

        x = x_new
        r = r_new

        if iter % 10 == 0:
            print(f"Iteration {iter + 1}: Signal change = {signal_change[-1]:.6e}, Residual change = {residual_change[-1]:.6e}")

    return x, signal_change, residual_change


# Example usage
n, m = 100, 500
sparsity = 0.05
A, b, x_true = generate_data(n, m, sparsity, scale_factor=100.0)

# 运行 AMP 算法
x_recovered, signal_change, residual_change = amp(A, b)

# 打印恢复的信号
print(f'Recovered signal: {x_recovered}')
print(f'True signal: {x_true}')
fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.plot(x_recovered, color = 'r', linestyle='--', marker = 'o',)
axs.plot(x_true, color = 'b', linestyle='--', marker = '*',)
axs.set_xlabel('Iteration')
axs.set_ylabel('MSE')
plt.show()
plt.close()












