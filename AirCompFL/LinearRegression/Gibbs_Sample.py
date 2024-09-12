




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:12:44 2024

@author: jack

https://blog.csdn.net/zhangxiuli006/article/details/115409238

https://blog.csdn.net/u012290039/article/details/105696097

https://blog.csdn.net/fjssharpsword/article/details/80365089

https://blog.csdn.net/google19890102/article/details/51755245

https://shunliz.gitbooks.io/machine-learning/content/nlp/lda/lda-gibbs.html


"""




#%%>>>>>>>>>>>>>>>>>>>>>>> 2. 贝叶斯估计 (Bayesian Estimation)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, invgamma

# 设置随机数种子
np.random.seed(42)

# 生成虚拟数据集
true_mu = 5
true_sigma = 2
n_samples = 50
data = np.random.normal(true_mu, true_sigma, n_samples)

# 贝叶斯推断: 假设先验为常见的非信息性先验,均值的先验分布为高斯分布,方差的先验分布为Inverse-Gamma分布
mu_prior_mean = 0
mu_prior_std = 10
alpha_prior = 1
beta_prior = 1

# Gibbs采样
n_iter = 1000
mu_samples = np.zeros(n_iter)
sigma_samples = np.zeros(n_iter)

mu_current = np.mean(data)
sigma2_current = np.var(data)

for i in range(n_iter):
    # 更新均值mu
    mu_n = (mu_prior_mean / (mu_prior_std**2) + np.sum(data) / sigma2_current) / (1/(mu_prior_std**2) + n_samples / sigma2_current)
    sigma_n = np.sqrt(1 / (1/(mu_prior_std**2) + n_samples / sigma2_current))
    mu_current = np.random.normal(mu_n, sigma_n)

    # 更新方差sigma^2
    alpha_n = alpha_prior + n_samples / 2
    beta_n = beta_prior + 0.5 * np.sum((data - mu_current)**2)
    sigma2_current = invgamma.rvs(alpha_n, scale=beta_n)

    mu_samples[i] = mu_current
    sigma_samples[i] = np.sqrt(sigma2_current)

# 生成图形
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 数据分布和拟合的高斯分布
sns.histplot(data, kde=True, color="blue", ax=axs[0, 0])
x = np.linspace(min(data), max(data), 100)
for i in range(100):
    sample_mu = mu_samples[i]
    sample_sigma = sigma_samples[i]
    axs[0, 0].plot(x, norm.pdf(x, sample_mu, sample_sigma), color='red', alpha=0.1)
axs[0, 0].set_title('Data Distribution with Posterior Samples')

# 均值mu的后验分布
sns.histplot(mu_samples, kde=True, ax=axs[0, 1])
axs[0, 1].axvline(x=true_mu, color='red', linestyle='--')
axs[0, 1].set_title('Posterior Distribution of $\mu$')

# 方差sigma的后验分布
sns.histplot(sigma_samples, kde=True, ax=axs[1, 0])
axs[1, 0].axvline(x=true_sigma, color='red', linestyle='--')
axs[1, 0].set_title('Posterior Distribution of $\sigma$')

# 均值与方差的联合后验分布
sns.scatterplot(x = mu_samples, y = sigma_samples, ax = axs[1, 1], s = 10, alpha = 0.5)
axs[1, 1].set_title('Joint Posterior Distribution of $\mu$ and $\sigma$')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>
# https://zhuanlan.zhihu.com/p/614076300
import numpy as np

# 吉布斯抽样
def gibbs_sampling(conditional_prob, initial_state, num_samples):
    n = len(initial_state)

    # 初始化状态序列
    state_sequence = np.zeros((num_samples, n))
    state_sequence[0] = initial_state

    # 生成样本
    for i in range(1, num_samples):
        for j in range(n):
            # 计算条件概率分布
            prob_distribution = conditional_prob[j](state_sequence[i - 1])
            # 抽取新状态
            new_state_j = np.random.choice([0, 1], p = prob_distribution)
            # 更新状态序列
            state_sequence[i][j] = new_state_j
    return state_sequence

# 测试代码
if __name__ == '__main__':
    # 定义条件概率分布
    def p_x_given_y(y):
        return [1 - y[1], y[1]]
    def p_y_given_x(x):
        return [1 / (1 + np.exp(-x[0])), 1 / (1 + np.exp(x[0]))]
    conditional_prob = [p_x_given_y, p_y_given_x]

    # 初始状态
    initial_state = [0, 0]

    # 生成样本
    state_sequence = gibbs_sampling(conditional_prob, initial_state, 1000)

    # 输出结果
    print('State sequence:', state_sequence[-10:])




































































































































































































































































































