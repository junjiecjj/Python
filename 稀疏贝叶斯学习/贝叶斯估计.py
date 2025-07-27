#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 13:52:50 2025

@author: jack


https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247491371&idx=1&sn=e453258a63e4d1afa9430aa044f982ce&chksm=c15cfdea938ccd515df8f5dc52292534a102a1999888066036968876d01ae074b0669a0a2d72&mpshare=1&scene=1&srcid=0727IEptvqGm8ihZalqZiSdz&sharer_shareinfo=646f35a3042195fc722d1d942932a865&sharer_shareinfo_first=646f35a3042195fc722d1d942932a865&exportkey=n_ChQIAhIQsfsdAF6nCQkknuUvJS04kRKfAgIE97dBBAEAAAAAAC0ANbvuzrwAAAAOpnltbLcz9gKNyK89dVj0vQfHYWE6908G%2BX9ZmdTz3kMPC%2BJtLhPMpq71hsEVu35pPJa%2BkhDqr%2BifeVidqKQMuOYO7%2BYIJemdGqAztFfQRVzn7DnRLc9XaFdT9vujtbeYoLgUb8LukyzBFT9bbZTtjgEldRENwP5uVEHMPBnjoD%2FSL47SWmSXKRPeSVYMpHszXBtsWrhUGyDb7AD8264Qb7nfGFV6FOX%2FQSDBuzmBRgDsp6Smj%2B6cbFnWcNf3JNJ3JeEZjyW5qOtFnFNKhkOuUmjHWbBrhWa7iguOnUgvEisCxJvKjhJHposd2DrHOFGmhtS%2F6feL4NTkkQQYTCQhGoaH2a26YdU3&acctmode=0&pass_ticket=uUYwpKkI%2BoJkqZRafZhzN1tf3KH4m%2BQ9%2FrbUXTkAGlt5RfynR3hf0WM4OGAB3MJ%2F&wx_header=0#rd

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta, binom


np.random.seed(42)

# 1. 商品数量
n_items = 500

# 每个商品的销售数量（10 ~ 500 之间）
sales = np.random.randint(10, 500, size=n_items)

# 每个商品的真实退货率（从 beta 分布生成）
true_return_rates = np.random.beta(2, 5, size=n_items)

# 实际退货数量（从 binomial 分布生成）
returns = np.random.binomial(sales, true_return_rates)

# 构建 DataFrame
df = pd.DataFrame({
    "item_id": np.arange(n_items),
    "sales": sales,
    "returns": returns,
    "true_return_rate": true_return_rates
})

# 计算频率估计（最大似然估计 MLE）
df["mle_return_rate"] = df["returns"] / df["sales"]

# 2. 贝叶斯估计部分
# 设置 beta 分布的先验参数
alpha_prior = 2
beta_prior = 5

# 后验参数（每个商品）
df["posterior_alpha"] = alpha_prior + df["returns"]
df["posterior_beta"] = beta_prior + (df["sales"] - df["returns"])

# 后验期望作为贝叶斯估计值
df["bayes_estimate"] = df["posterior_alpha"] / (df["posterior_alpha"] + df["posterior_beta"])

# 3. 可视化分析

# 3.1 比较 MLE 与 贝叶斯估计（带颜色区分销售量）
plt.figure(figsize=(14, 6))
sns.scatterplot(x="mle_return_rate", y="bayes_estimate", size="sales", hue="sales", data=df, palette="coolwarm", sizes=(20, 200))
plt.plot([0, 1], [0, 1], 'k--', label="y = x")
plt.title("MLE vs Bayesian Estimate of Return Rate")
plt.xlabel("Maximum Likelihood Estimate (MLE)")
plt.ylabel("Bayesian Estimate")
plt.legend(title="Sales", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 3.2 销售量小的商品中，MLE 和贝叶斯估计偏差最大
small_sample = df[df["sales"] < 50]
plt.figure(figsize=(14, 6))
plt.scatter(small_sample["mle_return_rate"], small_sample["bayes_estimate"], color='orange', label='Items with Low Sales')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Low-sales Items: MLE vs Bayesian Estimate")
plt.xlabel("MLE")
plt.ylabel("Bayesian Estimate")
plt.legend()
plt.show()

# 3.3 后验分布示意图（选一个商品）
idx = df.sample(1).index[0]
item = df.loc[idx]

x = np.linspace(0, 1, 500)
posterior = beta.pdf(x, item["posterior_alpha"], item["posterior_beta"])
plt.figure(figsize=(12, 5))
plt.plot(x, posterior, color='crimson', lw=3)
plt.title(f"Posterior Distribution of Return Rate for Item {item['item_id']}")
plt.axvline(item["mle_return_rate"], color='blue', ls='--', label="MLE")
plt.axvline(item["bayes_estimate"], color='green', ls='--', label="Bayes Estimate")
plt.legend()
plt.xlabel("Return Rate θ")
plt.ylabel("Density")
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>> 2. 贝叶斯估计 (Bayesian Estimation)
# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247485956&idx=1&sn=abac1ec541bf3c8f51cf38114c22fc0d&chksm=c1ae6c993e3698811a85a706a4389cc6d8d8ffbdab05eb7462452868215344fa76287aa3da3b&mpshare=1&scene=1&srcid=0822GQ8QZOOaAbFcHOYS6ozJ&sharer_shareinfo=792b67ed4ccef1bee1c59d47e8285e91&sharer_shareinfo_first=792b67ed4ccef1bee1c59d47e8285e91&exportkey=n_ChQIAhIQNvSF0aFR9HfJVszrGewg%2BhKfAgIE97dBBAEAAAAAALyqIGEjphAAAAAOpnltbLcz9gKNyK89dVj0M7UjKheWmvj7E62WcFIB2ejlGDIP%2BD39Lj7wRBB%2FKOBEHidvMsrcbhWl3CfGBb9ThSAfcizDTQW1OWhW8npbVbzVLLFl2k%2B8vjBde50MIWp6Mnl02PbpqaBbY7H3r8zrV9PYsJ5cgYd4XLTg11uFwmTUIW6L%2Fm4P34sLQIjSyCSPWPT5tDbJpkR7rph3%2F9qRmqmBzzArTLwkM2RMS0SeAVWTtTZfOLpNNuMh%2Bd7UksOLQI7rl581mG8FcM9ts9zGEoYAK%2Biyj7%2FsHhjITPyU0AsrS6aql4%2BPx1tiUYF2qe8FUB5BCawmA8QOJmXbfbeTS6FB4zkI%2FNVk&acctmode=0&pass_ticket=4JksxdRP9ZK%2BlfuY8ugpV87Z0z2yUPk0b12Oab%2FDhhovXUUPFL8AK5gxyje%2FNWTP&wx_header=0#rd
#
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
axs[0, 1].set_title('Posterior Distribution of $\\mu$')

# 方差sigma的后验分布
sns.histplot(sigma_samples, kde=True, ax=axs[1, 0])
axs[1, 0].axvline(x=true_sigma, color='red', linestyle='--')
axs[1, 0].set_title('Posterior Distribution of $\\sigma$')

# 均值与方差的联合后验分布
sns.scatterplot(x=mu_samples, y=sigma_samples, ax=axs[1, 1], s=10, alpha=0.5)
axs[1, 1].set_title('Joint Posterior Distribution of $\\mu$ and $\\sigma$')

plt.tight_layout()
plt.show()






































































































































































