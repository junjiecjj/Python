



#%%>>>>>>>>>>>>>>>>>>>>>>> 1-feature 期望最大化算法 (Expectation-Maximization, EM) >>>>>>>>>>>>>>>>>>>>>>>>>>>


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 生成虚拟数据集
np.random.seed(42)

# 参数设置
mu1, sigma1, n1 = 2, 0.5, 400  # 第一组高斯分布参数
mu2, sigma2, n2 = 8, 1.0, 600  # 第二组高斯分布参数

# 生成两组高斯分布数据
data1 = np.random.normal(mu1, sigma1, n1)
data2 = np.random.normal(mu2, sigma2, n2)
data = np.hstack((data1, data2))  # 合并数据集

# EM算法初始化
def initialize_parameters(data, k):
    weights = np.ones(k) / k
    means = np.random.choice(data, k, replace=False)
    variances = np.random.random_sample(k)
    return weights, means, variances

# E-step: 计算责任度
def e_step(data, weights, means, variances):
    responsibilities = np.zeros((len(data), len(means)))
    for i in range(len(means)):
        responsibilities[:, i] = weights[i] * norm.pdf(data, means[i], np.sqrt(variances[i]))
    responsibilities /= responsibilities.sum(1, keepdims=True)
    return responsibilities

# M-step: 更新参数
def m_step(data, responsibilities):
    nk = responsibilities.sum(axis=0)
    weights = nk / len(data)
    means = (responsibilities.T @ data) / nk
    variances = np.zeros(len(means))
    for i in range(len(means)):
        variances[i] = (responsibilities[:, i] * (data - means[i])**2).sum() / nk[i]
    return weights, means, variances

# 计算对数似然函数
def compute_log_likelihood(data, weights, means, variances):
    log_likelihood = 0
    for i in range(len(means)):
        log_likelihood += weights[i] * norm.pdf(data, means[i], np.sqrt(variances[i]))
    return np.sum(np.log(log_likelihood))

# EM算法主函数
def em_algorithm(data, k, max_iter=100, tol=1e-6):
    weights, means, variances = initialize_parameters(data, k)
    log_likelihoods = []

    for iteration in range(max_iter):
        responsibilities = e_step(data, weights, means, variances)
        weights, means, variances = m_step(data, responsibilities)
        log_likelihood = compute_log_likelihood(data, weights, means, variances)
        log_likelihoods.append(log_likelihood)

        if iteration > 0 and abs(log_likelihood - log_likelihoods[-2]) < tol:
            break

    return weights, means, variances, log_likelihoods

## 运行EM算法
k = 2  # 混合高斯模型的成分数量
weights, means, variances, log_likelihoods = em_algorithm(data, k)

# 绘图
plt.figure(figsize=(12, 8))

# 子图1：原始数据与高斯分布拟合曲线
plt.subplot(2, 1, 1)
plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Data histogram')

x = np.linspace(min(data), max(data), 1000)
for i in range(k):
    plt.plot(x, weights[i] * norm.pdf(x, means[i], np.sqrt(variances[i])), label=f'Gaussian {i+1}')
plt.title('Data and Fitted Gaussian Distributions')
plt.xlabel('Data points')
plt.ylabel('Density')
plt.legend()

# 子图2：对数似然函数变化
plt.subplot(2, 1, 2)
plt.plot(log_likelihoods, marker='o')
plt.title('Log-Likelihood during EM iterations')
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> n-feature 期望最大化算法 (Expectation-Maximization, EM) >>>>>>>>>>>>>>>>>>>>>>>>>>>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 读取数据集
data = pd.read_csv("Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 初始化参数
def initialize_params_fixed(X, K):
    n, d = X.shape
    pi = np.ones(K) / K  # 初始化每个混合成分的权重
    mu = X[np.random.choice(n, K, False), :]  # 随机选择K个初始均值
    sigma = np.array([np.eye(d) for _ in range(K)])  # 初始化协方差矩阵为单位矩阵
    return pi, mu, sigma

# 计算多元正态分布
def multivariate_gaussian(X, mu, sigma):
    return multivariate_normal(mean = mu, cov = sigma).pdf(X)

# E 步：计算每个点属于每个成分的责任值 (gamma)
def expectation_step_stable(X, pi, mu, sigma):
    N = X.shape[0]
    K = len(pi)
    gamma = np.zeros((N, K))
    for k in range(K):
        try:
            gamma[:, k] = pi[k] * multivariate_gaussian(X, mu[k], sigma[k])  ## 书上公式(21)(22)
        except np.linalg.LinAlgError:
            # 如果协方差矩阵是奇异矩阵，加入微小正则化项以确保正定性
            sigma[k] += np.eye(X.shape[1]) * 1e-6
            gamma[:, k] = pi[k] * multivariate_gaussian(X, mu[k], sigma[k])  ##
    # 防止零除错误，保证数值稳定性
    gamma_sum = np.sum(gamma, axis = 1, keepdims = True)
    gamma_sum[gamma_sum == 0] = 1e-10  # 防止除以零
    gamma = gamma / gamma_sum  ## 书上公式(23)
    return gamma

# M 步：更新GMM的参数
def maximization_step(X, gamma):
    N, d = X.shape
    K = gamma.shape[1]
    Nk = np.sum(gamma, axis = 0)  # 计算每个聚类的总责任值
    pi = Nk / N  # 更新混合系数, 书上公式(24)
    mu = gamma.T @ X / Nk[:, np.newaxis]  # 更新均值, 书上公式(25)
    sigma = np.zeros((K, d, d))  # 更新协方差矩阵, 书上公式(26)
    for k in range(K):
        X_centered = X - mu[k]
        gamma_diag = np.diag(gamma[:, k])
        sigma[k] =  X_centered.T @ gamma_diag @ X_centered / Nk[k]   ##  np.dot(X_centered.T, np.dot(gamma_diag, X_centered)) / Nk[k]
    return pi, mu, sigma

# 计算对数似然
def compute_log_likelihood(X, pi, mu, sigma):
    N = X.shape[0]
    K = len(pi)
    log_likelihood = 0
    for n in range(N):
        tmp = 0
        for k in range(K):
            tmp += pi[k] * multivariate_gaussian(X[n], mu[k], sigma[k])
        log_likelihood += np.log(tmp)
    return log_likelihood

# GMM 实现，包含数值稳定性修复
def gmm_fixed_stable(X, K, max_iter = 100, tol = 1e-6):
    pi, mu, sigma = initialize_params_fixed(X, K)
    log_likelihoods = []
    for i in range(max_iter):
        # E 步
        gamma = expectation_step_stable(X, pi, mu, sigma)
        # M 步
        pi, mu, sigma = maximization_step(X, gamma)
        # 添加小的正则化项，确保协方差矩阵为正定
        sigma += np.eye(sigma.shape[1]) * 1e-6
        # 计算对数似然
        log_likelihood = compute_log_likelihood(X, pi, mu, sigma)
        log_likelihoods.append(log_likelihood)
        # 检查是否收敛
        if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break
    return pi, mu, sigma, log_likelihoods, gamma

# 数据可视化：原始数据分布
def plot_original_data(X):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c='blue', label='Data points', alpha=0.5)
    plt.title('Original Data Distribution')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()
    return

# 分类结果展示
def plot_clusters(X, gamma, mu):
    K = gamma.shape[1]
    colors = ['r', 'g', 'b', 'y', 'm']
    plt.figure(figsize = (8, 6))
    for k in range(K):
        plt.scatter(X[:, 0], X[:, 1], c = gamma[:, k], cmap = 'viridis', label = f'Cluster {k+1}', alpha = 0.6)
    plt.scatter(mu[:, 0], mu[:, 1], c = 'black', marker = 'x', s = 100, label = 'Centroids')
    plt.title('GMM Clustering')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    return

# 对数似然收敛图
def plot_log_likelihood(log_likelihoods):
    plt.figure(figsize = (8, 6))
    plt.plot( log_likelihoods)
    plt.title('Log Likelihood Convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Log Likelihood')
    plt.show()
    return

# 各类别概率分布图
def plot_probability_distributions(gamma):
    plt.figure(figsize=(8, 6))
    K = gamma.shape[1]
    for k in range(K):
        plt.hist(gamma[:, k], bins = 20, alpha = 0.5, label = f'Cluster {k+1}')
    plt.title('Probability Distributions for Each Cluster')
    plt.xlabel('Probability')
    plt.ylabel('Number of Points')
    plt.legend()
    plt.show()
    return

# 运行 GMM 算法
K = 3  # 假设数据有 3 个聚类
pi, mu, sigma, log_likelihoods, gamma = gmm_fixed_stable(X, K)

# 绘制图形
plot_original_data(X)  # 原始数据分布图
plot_clusters(X, gamma, mu)  # 分类结果图
plot_log_likelihood(log_likelihoods)  # 对数似然收敛图
plot_probability_distributions(gamma)  # 各类别概率分布图











































































































































































































































































































































































































































































































































































































































































































































