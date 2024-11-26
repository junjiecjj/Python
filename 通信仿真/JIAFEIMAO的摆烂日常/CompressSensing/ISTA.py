#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:51:09 2024

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import mean_squared_error

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
# rcParams['figure.dpi'] = 300

def generate_data(n_samples = 100, n_features = 50, sparsity = 5, noise_level = 0.1):
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    indices = np.random.choice(n_features, sparsity, replace = False)
    true_coef[indices] = np.random.randn(sparsity) * 10
    y = X @ true_coef + noise_level * np.random.randn(n_samples)
    return X, y, true_coef

# ISTA算法
def ISTA(X, y, lambda_, eta, max_iter = 1000, tol = 1e-6):
    beta = np.zeros(X.shape[1])
    cost_history = []
    mse_history = []

    for i in range(max_iter):
        gradient = X.T @ (X @ beta - y)
        beta_temp = beta - eta * gradient
        beta = np.sign(beta_temp) * np.maximum(np.abs(beta_temp) - eta * lambda_, 0)
        cost = 0.5 * np.linalg.norm(y - X @ beta) ** 2 + lambda_ * np.sum(np.abs(beta))
        mse = mean_squared_error(true_coef, beta)

        cost_history.append(cost)
        mse_history.append(mse)

        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < tol:
            break
    return beta, cost_history, mse_history

# FISTA算法
def FISTA(X, y, lambda_, eta, max_iter = 1000, tol = 1e-6):
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
        mse = mean_squared_error(true_coef, beta)

        cost_history.append(cost)
        mse_history.append(mse)

        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < tol:
            break
    return beta, cost_history, mse_history

n_samples, n_features, sparsity = 100, 50, 5
lambda_, eta = 0.1, 0.001
X, y, true_coef = generate_data(n_samples, n_features, sparsity)
beta_ista, cost_history_ista, mse_history_ista = ISTA(X, y, lambda_, eta)
beta_fista, cost_history_fista, mse_history_fista = FISTA(X, y, lambda_, eta)

fig, axs = plt.subplots(3, 2, figsize=(14, 15), constrained_layout=True)
fig.suptitle("ISTA与FISTA算法对比", fontsize=16, weight='bold')

axs[0, 0].stem(true_coef, linefmt='b-', markerfmt='bo', basefmt=" ", label="真实系数")
axs[0, 0].stem(beta_ista, linefmt='r--', markerfmt='rs', basefmt=" ", label="ISTA估计系数")
axs[0, 0].stem(beta_fista, linefmt='g-.', markerfmt='g^', basefmt=" ", label="FISTA估计系数")
axs[0, 0].legend()
axs[0, 0].set_title("真实系数与估计系数对比", fontsize=12)
axs[0, 0].set_xlabel("特征索引", fontsize=10)
axs[0, 0].set_ylabel("系数值", fontsize=10)
axs[0, 1].plot(cost_history_ista, color='crimson', linestyle='-', label="ISTA收敛")
axs[0, 1].plot(cost_history_fista, color='mediumseagreen', linestyle='-', label="FISTA收敛")
axs[0, 1].legend()
axs[0, 1].set_title("损失函数收敛", fontsize=12)
axs[0, 1].set_xlabel("迭代次数", fontsize=10)
axs[0, 1].set_ylabel("损失值", fontsize=10)

axs[1, 0].plot(mse_history_ista, color='crimson', linestyle='-', label="ISTA均方误差")
axs[1, 0].plot(mse_history_fista, color='mediumseagreen', linestyle='-', label="FISTA均方误差")
axs[1, 0].legend()
axs[1, 0].set_title("均方误差 (MSE) 收敛", fontsize=12)
axs[1, 0].set_xlabel("迭代次数", fontsize=10)
axs[1, 0].set_ylabel("均方误差", fontsize=10)

non_zero_ista = np.sum(beta_ista != 0)
non_zero_fista = np.sum(beta_fista != 0)
axs[1, 1].bar(["ISTA", "FISTA"], [non_zero_ista, non_zero_fista], color=['crimson', 'mediumseagreen'])
axs[1, 1].set_title("稀疏性对比（非零系数数目）", fontsize=12)
axs[1, 1].set_ylabel("非零系数数目", fontsize=10)

axs[2, 0].plot(np.log10(cost_history_ista), color='crimson', linestyle='-', label="ISTA收敛速度")
axs[2, 0].plot(np.log10(cost_history_fista), color='mediumseagreen', linestyle='-', label="FISTA收敛速度")
axs[2, 0].legend()
axs[2, 0].set_title("收敛速度对比（对数刻度）", fontsize=12)
axs[2, 0].set_xlabel("迭代次数", fontsize=10)
axs[2, 0].set_ylabel("对数损失值", fontsize=10)

axs[2, 1].bar(["ISTA", "FISTA"], [len(cost_history_ista), len(cost_history_fista)], color=['crimson', 'mediumseagreen'])
axs[2, 1].set_title("计算时间（迭代次数）对比", fontsize=12)
axs[2, 1].set_ylabel("迭代次数", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


#%% 下面来分析ISTA和FISTA两个算法的参数
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from scipy.linalg import norm

np.random.seed(0)
n_samples, n_features = 100, 50
X = np.random.randn(n_samples, n_features)
true_coef = np.zeros(n_features)
true_coef[:5] = np.random.randn(5) * 10
y = X @ true_coef + 0.1 * np.random.randn(n_samples)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 300

def ista(X, y, eta, lambd, max_iter=500):
    beta = np.zeros(X.shape[1])
    loss_history = []
    for i in range(max_iter):
        grad = X.T @ (X @ beta - y)
        beta_temp = beta - eta * grad
        beta = np.sign(beta_temp) * np.maximum(np.abs(beta_temp) - eta * lambd, 0)
        try:
            loss = 0.5 * norm(X @ beta - y)**2 + lambd * norm(beta, 1)
            loss_history.append(loss)
        except OverflowError:
            print("损失值爆炸")
            break
    return beta, loss_history

def fista(X, y, eta, lambd, max_iter=500):
    beta = np.zeros(X.shape[1])
    beta_prev = np.zeros(X.shape[1])
    t = 1
    loss_history = []
    for i in range(max_iter):
        grad = X.T @ (X @ beta - y)
        beta_temp = beta - eta * grad
        beta_new = np.sign(beta_temp) * np.maximum(np.abs(beta_temp) - eta * lambd, 0)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        beta = beta_new + ((t - 1) / t_new) * (beta_new - beta_prev)
        beta_prev = beta_new
        t = t_new
        try:
            loss = 0.5 * norm(X @ beta - y)**2 + lambd * norm(beta, 1)
            loss_history.append(loss)
        except OverflowError:
            print("损失值爆炸")
            break
    return beta, loss_history

etas = [0.001, 0.01, 0.1]
lambdas = [0.1, 0.5, 1.0]
max_iter = 500

fig, axs = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=True)
fig.suptitle("ISTA 与 FISTA 参数影响分析", fontsize=16)

for idx, eta in enumerate(etas):
    for jdx, lambd in enumerate(lambdas):
        # ISTA
        beta_ista, loss_ista = ista(X, y, eta, lambd, max_iter)
        axs[idx, jdx].plot(loss_ista, label=f"ISTA, η={eta}, λ={lambd}", color='blue', linestyle='--')

        # FISTA
        beta_fista, loss_fista = fista(X, y, eta, lambd, max_iter)
        axs[idx, jdx].plot(loss_fista, label=f"FISTA, η={eta}, λ={lambd}", color='green')

        axs[idx, jdx].set_title(f"学习率 η={eta} & 正则化 λ={lambd}")
        axs[idx, jdx].set_xlabel("迭代次数")
        axs[idx, jdx].set_ylabel("损失值")
        axs[idx, jdx].legend()
        axs[idx, jdx].grid()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()






























































