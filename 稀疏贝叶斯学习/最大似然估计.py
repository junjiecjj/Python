#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 13:53:35 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247491312&idx=1&sn=106c6735ad5dde3135f775fa1051d8f7&chksm=c121c93b30d916f804876064a86381ad30982ca5bd9bc6b2fcac1a086c435a527fbc618875a7&mpshare=1&scene=1&srcid=07276MLvTQFYUZ26cagUWkB1&sharer_shareinfo=9041b020cbf3a5d1ee9b3bfd739b8077&sharer_shareinfo_first=9041b020cbf3a5d1ee9b3bfd739b8077&exportkey=n_ChQIAhIQuWeqf9qfm3W10jRNN%2Fm8IRKfAgIE97dBBAEAAAAAAFfjI2fTehcAAAAOpnltbLcz9gKNyK89dVj0LmL49sfQERwukgt%2BdMFqAH0jmZuHQvPW%2B0zFMA5gbp4JSsjQAU5TXKSjz68UuXxzTDjPqRNLvN23MJ6yqEZdSxDCcFwagwYVc%2B5kbNAPcDnDwGevl9kR77vsl09lknxWVNWwaT3m%2B7W5ifuMJrKOPgYolKzovpOn4ICO2qi2iBg09ZTBoUZviOpNCBHwso86%2F5HUOKs%2Bj3NHXwt2HnwSaQM%2FgsgttopjD9b93sjMiIYq7EOQ%2F5MLr%2FTPDzmQemqo9IQA0PT6rjj8pVAzzzdvzwlELJygH3zYT4RUryPGv8ecCFBVerzVPT9XXmoSnM1TpJl6KuI7VTrX&acctmode=0&pass_ticket=RXpdkX0GrcXToIWtjdoyIFeQfWWMJMIuuW%2BSewNqt6ipE6N9MIdd5Z34a%2BKL84V5&wx_header=0#rd


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.optimize import minimize
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.labelspacing'] = 0.2

np.random.seed(42)

# 模拟真实数据，真实 λ = 0.5（即平均等待时间 2 分钟）
lambda_real = 0.5
n_samples = 1000
data = np.random.exponential(scale=1/lambda_real, size=n_samples)

# 最大似然估计 λ 的函数
def negative_log_likelihood(lmbda):
    if lmbda <= 0:
        return np.inf
    n = len(data)
    return - (n * np.log(lmbda) - lmbda * np.sum(data))

# 数值优化求解最大似然估计
result = minimize(negative_log_likelihood, x0=np.array([1.0]), bounds=[(1e-5, None)])
lambda_mle = result.x[0]

# 可视化数据直方图和拟合曲线
x_vals = np.linspace(0, 15, 1000)
pdf_real = expon.pdf(x_vals, scale=1/lambda_real)
pdf_mle = expon.pdf(x_vals, scale=1/lambda_mle)

plt.figure(figsize=(12, 6))
# 数据直方图
plt.hist(data, bins=30, density=True, color='#FF9999', alpha=0.6, edgecolor='black', label="观察数据")

# 拟合曲线（真实 λ）
plt.plot(x_vals, pdf_real, color='blue', lw=2.5, label="真实分布 (λ=0.5)", linestyle='--')

# 拟合曲线（MLE λ）
plt.plot(x_vals, pdf_mle, color='green', lw=3, label=f"MLE拟合 (λ≈{lambda_mle:.3f})")

# 样式设置
plt.title("MLE 拟合指数分布（出租车等待时间）", fontsize=16)
plt.xlabel("等待时间（分钟）", fontsize=14)
plt.ylabel("概率密度", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# 打印结果分析
print(f"真实 λ: {lambda_real}")
print(f"MLE 估计的 λ: {lambda_mle:.5f}")
print(f"平均等待时间的估计（MLE）: {1/lambda_mle:.3f} 分钟")


#%%>>>>>>>>>>>>>>>>>>>>>>> 1. 最大似然估计 (Maximum Likelihood Estimation, MLE)
# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247485956&idx=1&sn=abac1ec541bf3c8f51cf38114c22fc0d&chksm=c1ae6c993e3698811a85a706a4389cc6d8d8ffbdab05eb7462452868215344fa76287aa3da3b&mpshare=1&scene=1&srcid=0822GQ8QZOOaAbFcHOYS6ozJ&sharer_shareinfo=792b67ed4ccef1bee1c59d47e8285e91&sharer_shareinfo_first=792b67ed4ccef1bee1c59d47e8285e91&exportkey=n_ChQIAhIQNvSF0aFR9HfJVszrGewg%2BhKfAgIE97dBBAEAAAAAALyqIGEjphAAAAAOpnltbLcz9gKNyK89dVj0M7UjKheWmvj7E62WcFIB2ejlGDIP%2BD39Lj7wRBB%2FKOBEHidvMsrcbhWl3CfGBb9ThSAfcizDTQW1OWhW8npbVbzVLLFl2k%2B8vjBde50MIWp6Mnl02PbpqaBbY7H3r8zrV9PYsJ5cgYd4XLTg11uFwmTUIW6L%2Fm4P34sLQIjSyCSPWPT5tDbJpkR7rph3%2F9qRmqmBzzArTLwkM2RMS0SeAVWTtTZfOLpNNuMh%2Bd7UksOLQI7rl581mG8FcM9ts9zGEoYAK%2Biyj7%2FsHhjITPyU0AsrS6aql4%2BPx1tiUYF2qe8FUB5BCawmA8QOJmXbfbeTS6FB4zkI%2FNVk&acctmode=0&pass_ticket=4JksxdRP9ZK%2BlfuY8ugpV87Z0z2yUPk0b12Oab%2FDhhovXUUPFL8AK5gxyje%2FNWTP&wx_header=0#rd
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

# 生成模拟数据
np.random.seed(42)
true_mu = 5
true_sigma = 2
data = np.random.normal(true_mu, true_sigma, size=1000)

# 定义负对数似然函数
def neg_log_likelihood(params, data):
    mu, sigma = params[0], params[1]
    if sigma <= 0:
        return np.inf
    log_likelihood = np.sum(np.log(norm.pdf(data, mu, sigma)))
    return -log_likelihood

# 使用优化器找到 MLE 参数
initial_guess = [0, 1]
result = minimize(neg_log_likelihood, initial_guess, args=(data,), method='L-BFGS-B', bounds=[(None, None), (1e-5, None)])
mle_mu, mle_sigma = result.x  # 其实就是data的均值和标准差

# 绘制数据的直方图与拟合的正态分布曲线对比图
x = np.linspace(min(data), max(data), 1000)
pdf_true = norm.pdf(x, true_mu, true_sigma)
pdf_mle = norm.pdf(x, mle_mu, mle_sigma)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Histogram of data')
plt.plot(x, pdf_true, 'r--', label=f'True Gaussian\n($\\mu$={true_mu}, $\\sigma$={true_sigma})')
plt.plot(x, pdf_mle, 'b-', label=f'MLE Gaussian\n($\\mu$={mle_mu:.2f}, $\\sigma$={mle_sigma:.2f})')
plt.title('Data Histogram and Fitted Gaussian')
plt.xlabel('Data')
plt.ylabel('Density')
plt.legend()

# 绘制对数似然函数关于均值和标准差的等高线图
mu_values = np.linspace(4, 6, 100)
sigma_values = np.linspace(1.5, 2.5, 100)
log_likelihood_values = np.zeros((len(mu_values), len(sigma_values)))

for i, mu in enumerate(mu_values):
    for j, sigma in enumerate(sigma_values):
        log_likelihood_values[i, j] = -neg_log_likelihood([mu, sigma], data)

mu_grid, sigma_grid = np.meshgrid(mu_values, sigma_values)
plt.subplot(1, 2, 2)
contour = plt.contour(mu_grid, sigma_grid, log_likelihood_values.T, levels=50, cmap='viridis')
plt.plot(mle_mu, mle_sigma, 'ro', label='MLE Estimate')
plt.colorbar(contour, label='Negative Log-Likelihood')
plt.title('Log-Likelihood Contours')
plt.xlabel('Mean (mu)')
plt.ylabel('Standard Deviation (sigma)')
plt.legend()

plt.tight_layout()
plt.show()




