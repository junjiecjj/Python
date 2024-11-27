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



# 初始化输入数据 A 和 b
def generate_data(n, m, sparsity = 0.1, scale_factor = 1000.0, noise_var = 0.00001):
    A = np.random.randn(m, n)/np.sqrt(m)  # 随机生成一个 m x n 的矩阵
    x = np.zeros(n)
    non_zero_indices = np.random.choice(n, size=int(n * sparsity), replace=False)
    x[non_zero_indices] = scale_factor * np.random.randn(len(non_zero_indices))  # 放大信号
    b = A @ x + noise_var * np.random.randn(m)  # 降低噪声强度
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


# OMP 算 法
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












