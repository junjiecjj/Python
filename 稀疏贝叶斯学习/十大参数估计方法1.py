#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 22:23:55 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247491548&idx=1&sn=f2d9d621f461d5acc8d573ce206b4a71&chksm=c16c88e10c98531a96f4f21e931a1283edf4addfe1d165e331adfb0abae6778e2ff57197a7e3&mpshare=1&scene=1&srcid=0819qUS4E9NSZ356QkjQ9irN&sharer_shareinfo=e5381a969db54a586f0678499414b98f&sharer_shareinfo_first=e5381a969db54a586f0678499414b98f&exportkey=n_ChQIAhIQbQE38ZhuUPxUBdPbUpOc5xKfAgIE97dBBAEAAAAAAFDHEQyCmR4AAAAOpnltbLcz9gKNyK89dVj0DYGl4jQWYO%2Brb2u4CkUja92VaE%2FrkVkxrG4%2BXsl2TJ4SPB%2FAbSKjXFI8N4nl0atMRxjriQsOc7iBZVXhQa9IpjqeqpMZhMt%2Fun7VXec0vA3ZLPwz1o5Ov6AVAHKbJgyNz%2F3qwVv7CXJ%2Fr1MPvmA4aBu3wuAr%2BI5oqT%2BiHztcXDHW1ejIupIarLFhGGnIA6031JJyMP5C%2FOOQieXPwzSEo6qyXUPxcsytH%2BrG9HOS9ot4gK3WqkeuR%2BlMDhBvaJ9mn7nb25lXQvxfQ2EQSRuVEl63mqfWNuM3tHHAYPxyuL%2B90%2FOOi0%2FloYNPkMa%2BdDF7xEwDlVz8jDfC&acctmode=0&pass_ticket=HN5dHV6WmLvp%2BqneDXwH3OPAbZkj3qJW4nYjA7ZknnykI64UFhVmS932VYnDUc51&wx_header=0#rd


"""

#%%>>>>>>>>>>>>>>>>>>>>>>> 最大似然估计

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 生成一个正态分布的虚拟数据集
np.random.seed(42)
n = 1000# 样本大小
mu_true = 5# 真正的均值
sigma_true = 2# 真正的标准差
data = np.random.normal(mu_true, sigma_true, n)

# 最大似然估计 (MLE) 计算
mu_mle = np.mean(data)  # MLE估计的均值
sigma_mle = np.std(data)  # MLE估计的标准差

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制数据的直方图
ax.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data Histogram')

# 绘制MLE估计的正态分布曲线
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_mle, sigma_mle)
ax.plot(x, p, 'k', linewidth=2, label=f'MLE Normal Fit (μ={mu_mle:.2f}, σ={sigma_mle:.2f})')

# 绘制真实的正态分布曲线
p_true = norm.pdf(x, mu_true, sigma_true)
ax.plot(x, p_true, 'r--', linewidth=2, label=f'True Normal Distribution (μ={mu_true}, σ={sigma_true})')

# 设置图形属性
ax.set_title('Maximum Likelihood Estimation for Normal Distribution')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.legend()

# 显示图形
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 贝叶斯估计
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 设定虚拟数据
np.random.seed(42)
true_theta = 0.3# 真实点击率
n_samples = [5, 20, 100]  # 观察到的样本数量
alpha_prior, beta_prior = 2, 2# Beta(2,2) 作为先验分布

# 生成伯努利数据
data = {n: np.random.binomial(1, true_theta, n) for n in n_samples}

# 创建画布
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# 逐步更新后验分布
x = np.linspace(0, 1, 1000)
colors = ["blue", "green", "red"]

for i, n in enumerate(n_samples):
    successes = np.sum(data[n])
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + n - successes

    # 绘制先验和后验分布
    ax[i].plot(x, stats.beta.pdf(x, alpha_prior, beta_prior), 'k--', label="Prior", linewidth=2)
    ax[i].plot(x, stats.beta.pdf(x, alpha_post, beta_post), color=colors[i], label=f"Posterior (n={n})", linewidth=2)
    ax[i].fill_between(x, stats.beta.pdf(x, alpha_post, beta_post), color=colors[i], alpha=0.3)
    ax[i].set_title(f"Posterior Distribution (n={n})")
    ax[i].set_xlabel(r"$\theta$")
    ax[i].set_ylabel("Density")
    ax[i].legend()

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 最小二乘估计

import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据
np.random.seed(42)
x = np.linspace(1, 10, 50)  # 广告支出（自变量）
y_true = 2 * x + 3# 真实的销售额（因变量），即 y = 2x + 3
y = y_true + np.random.normal(0, 2, size=x.shape)  # 添加噪声到销售额

# 使用最小二乘法估计线性回归参数
X = np.vstack([np.ones_like(x), x]).T  # 构建设计矩阵
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y  # 计算回归系数

# 提取回归系数
beta_0, beta_1 = beta_hat
print(f"估计的回归系数：β0 = {beta_0:.2f}, β1 = {beta_1:.2f}")

# 绘制数据和回归直线
plt.figure(figsize=(10, 6))

# 绘制原始数据点
plt.scatter(x, y, color='orange', label='Observed Data', alpha=0.7)

# 绘制拟合线
y_pred = beta_0 + beta_1 * x
plt.plot(x, y_pred, color='blue', label=f'Fitted Line: y = {beta_0:.2f} + {beta_1:.2f}x')

# 添加噪声与拟合直线的残差
residuals = y - y_pred
plt.scatter(x, residuals, color='red', label='Residuals', alpha=0.5)

# 设置图形的标题和标签
plt.title('Least Squares Fit of Advertising Spend vs Sales', fontsize=16)
plt.xlabel('Advertising Spend (Units: 10,000)', fontsize=12)
plt.ylabel('Sales (Units: 10,000)', fontsize=12)
plt.legend()

plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 梯度下降估计
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100个数据点，范围在[0,2]
y = 4 + 3 * X + np.random.randn(100, 1)  # 线性关系 y = 4 + 3x + 噪声

# 初始化参数
theta = np.random.randn(2, 1)  # 随机初始化theta0和theta1
alpha = 0.1# 学习率
iterations = 100# 迭代次数
m = len(X)

# 添加x0 = 1列
X_b = np.c_[np.ones((m, 1)), X]  # 在X前面加一列1

# 梯度下降
losses = []
for _ in range(iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)  # 计算梯度
    theta -= alpha * gradients  # 更新参数
    loss = np.mean((X_b.dot(theta) - y) ** 2)  # 计算损失
    losses.append(loss)

# 预测拟合直线
y_pred = X_b.dot(theta)

# 绘制图像
fig, ax1 = plt.subplots(figsize=(10, 5))

# 1. 原始数据和拟合曲线
ax1.scatter(X, y, color='blue', label='Data')  # 数据点
ax1.plot(X, y_pred, color='red', linewidth=2, label='Fitted Line')  # 线性拟合
ax1.set_xlabel("X")
ax1.set_ylabel("y")
ax1.legend()
ax1.set_title("Linear Regression with Gradient Descent")

# 2. 损失函数变化曲线（次坐标轴）
ax2 = ax1.twinx()
ax2.plot(range(iterations), losses, color='green', linewidth=2, linestyle='dashed', label='Loss')
ax2.set_ylabel("Loss")
ax2.legend(loc='upper right')

plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 最大后验概率估计

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma

# 生成虚拟数据
np.random.seed(42)
true_mu = 5
true_sigma = 2
data = np.random.normal(loc=true_mu, scale=true_sigma, size=100)

# 先验分布的超参数
mu_0 = 0# 均值先验
sigma_0 = 2# 方差先验
alpha = 2# Inverse-Gamma分布的超参数
beta = 1

# MAP估计的最大后验概率计算
# 对于正态分布，MAP估计的后验概率是联合似然与先验的乘积
def map_estimation(data, mu_0, sigma_0, alpha, beta):
    # 似然函数的对数
    n = len(data)
    mu_hat = np.mean(data)  # MLE估计为均值
    sigma_hat = np.std(data, ddof=1)  # MLE估计为样本标准差

    # 先验的对数
    log_prior_mu = -0.5 * (mu_0 - mu_hat)**2 / sigma_0**2
    log_prior_sigma = (alpha - 1) * np.log(sigma_hat**2) - (beta / sigma_hat**2)

    # 对数似然函数
    log_likelihood = -n/2 * np.log(2 * np.pi * sigma_hat**2) - np.sum((data - mu_hat)**2) / (2 * sigma_hat**2)

    # 后验
    log_posterior = log_likelihood + log_prior_mu + log_prior_sigma

    return mu_hat, sigma_hat

mu_hat, sigma_hat = map_estimation(data, mu_0, sigma_0, alpha, beta)

# 绘制数据的直方图与估计的正态分布
x = np.linspace(min(data)-1, max(data)+1, 100)
pdf = norm.pdf(x, mu_hat, sigma_hat)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=20, density=True, alpha=0.6, color='yellow', label='Data Histogram')
plt.plot(x, pdf, color='blue', label='MAP Estimate PDF')
plt.title(f'MAP Estimation: $\mu={mu_hat:.2f}$, $\sigma={sigma_hat:.2f}$')
plt.legend()
plt.grid(True)

# 绘制MAP估计的置信区间
confidence_interval = [mu_hat - 1.96 * sigma_hat / np.sqrt(len(data)), mu_hat + 1.96 * sigma_hat / np.sqrt(len(data))]
plt.axvline(confidence_interval[0], color='green', linestyle='--', label='95% Confidence Interval')
plt.axvline(confidence_interval[1], color='green', linestyle='--')
plt.show()

# 绘制先验与后验的对比图
mu_range = np.linspace(-5, 10, 100)
prior_mu = norm.pdf(mu_range, mu_0, sigma_0)
posterior_mu = norm.pdf(mu_range, mu_hat, sigma_hat)

plt.figure(figsize=(10, 6))
plt.plot(mu_range, prior_mu, label='Prior', color='red')
plt.plot(mu_range, posterior_mu, label='Posterior', color='blue')
plt.title('Prior vs Posterior for $\mu$')
plt.legend()
plt.grid(True)
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 线性回归估计
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 生成虚拟数据集
np.random.seed(42)
n = 100
X = np.random.rand(n, 1) * 10# 自变量：0到10之间的随机数
Y = 2.5 * X + 3 + np.random.randn(n, 1) * 2# 因变量：线性关系加上噪声

# 创建线性回归模型
model = LinearRegression()
model.fit(X, Y)

# 预测
Y_pred = model.predict(X)

# 计算均方误差
mse = mean_squared_error(Y, Y_pred)

# 创建图形
plt.figure(figsize=(12, 8))

# 1. 回归线图
plt.subplot(2, 2, 1)
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression line')
plt.title(f"Linear Regression (MSE = {mse:.2f})")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# 2. 残差图
residuals = Y - Y_pred
plt.subplot(2, 2, 2)
plt.scatter(X, residuals, color='purple', label='Residuals')
plt.axhline(0, color='black', linewidth=2)
plt.title("Residual Plot")
plt.xlabel('X')
plt.ylabel('Residuals')

# 3. 散点图
plt.subplot(2, 2, 3)
sns.scatterplot(x=X.flatten(), y=Y.flatten(), color='green')
plt.title("Scatter Plot of Data")
plt.xlabel('X')
plt.ylabel('Y')

# 4. 残差分布图
plt.subplot(2, 2, 4)
sns.histplot(residuals, kde=True, color='orange')
plt.title("Residual Distribution")
plt.xlabel('Residuals')
plt.ylabel('Frequency')

# 显示所有图形
plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>> 蒙特卡洛方法

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

# 目标函数
def f(x):
    return np.exp(-x**2)

# 真实积分值
def true_integral():
    result, _ = spi.quad(f, 0, 1)
    return result

# 蒙特卡洛积分估计
def monte_carlo_integral(n):
    x_samples = np.random.uniform(0, 1, n)
    estimate = np.mean(f(x_samples))  # 计算均值
    return estimate

# 计算积分的真实值
I_true = true_integral()

# 进行不同样本数的估计
sample_sizes = np.logspace(1, 5, 50, dtype=int)  # 10到100000个样本
estimates = np.array([monte_carlo_integral(n) for n in sample_sizes])
errors = np.abs(estimates - I_true)

# 生成绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 收敛性分析
axes[0, 0].plot(sample_sizes, estimates, label='Monte Carlo Estimates', color='red')
axes[0, 0].axhline(y=I_true, color='black', linestyle='dashed', label='True Value')
axes[0, 0].set_xscale('log')
axes[0, 0].set_xlabel("Sample Size")
axes[0, 0].set_ylabel("Estimate")
axes[0, 0].set_title("Convergence Analysis")
axes[0, 0].legend()

# 2. 误差分析
axes[0, 1].plot(sample_sizes, errors, label='Error', color='blue')
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].set_xlabel("Sample Size")
axes[0, 1].set_ylabel("Error")
axes[0, 1].set_title("Error Analysis")
axes[0, 1].legend()

# 3. 采样点分布直方图
samples = np.random.uniform(0, 1, 10000)
axes[1, 0].hist(samples, bins=30, color='green', alpha=0.7, edgecolor='black')
axes[1, 0].set_title("Random Sample Distribution")
axes[1, 0].set_xlabel("x Sample Values")
axes[1, 0].set_ylabel("Frequency")

# 4. 目标函数图 + 积分区域示意
x = np.linspace(0, 1, 1000)
y = f(x)
axes[1, 1].plot(x, y, label='$e^{-x^2}$', color='purple')
axes[1, 1].fill_between(x, y, alpha=0.3, color='purple')
axes[1, 1].set_title("Objective Function and Integral Region")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("f(x)")
axes[1, 1].legend()

plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 矩估计法

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# 设置随机种子，保证结果可重复
np.random.seed(42)

# 生成正态分布样本数据，假设真实参数mu=5, sigma=2 (sigma^2=4)
mu_true = 5
sigma_true = 2
n = 100# 样本大小
data = np.random.normal(mu_true, sigma_true, n)

# 计算样本的第1阶和第2阶矩
m1 = np.mean(data)  # 样本均值 (第1阶矩)
m2 = np.mean(data**2)  # 样本的二次矩 (第2阶矩)

# 使用矩估计法估计参数
mu_hat = m1
sigma_hat = np.sqrt(m2 - m1**2)  # 估计的sigma

# 输出估计的参数
print(f"估计的均值 (mu) = {mu_hat:.2f}")
print(f"估计的标准差 (sigma) = {sigma_hat:.2f}")

# 绘制数据分析图
plt.figure(figsize=(10, 6))

# 图1: 样本数据的直方图
plt.subplot(1, 2, 1)
sns.histplot(data, kde=True, color='orange', bins=15)
plt.title('Sample Data Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 图2: 样本数据的QQ图
plt.subplot(1, 2, 2)
stats.probplot(data, dist="norm", plot=plt)
plt.title('QQ Plot')

plt.tight_layout()
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>> 卡尔曼滤波估计

import numpy as np
import matplotlib.pyplot as plt

# 设定参数
dt = 0.1# 时间间隔
total_time = 10# 总时间
n = int(total_time / dt)  # 时间步数

# 真实物体的运动模型：位置与速度
real_position = np.zeros(n)
real_velocity = np.ones(n)  # 假设物体的速度是恒定的

# 生成真实位置（带噪声）
for t in range(1, n):
    real_position[t] = real_position[t - 1] + real_velocity[t - 1] * dt

# 生成带噪声的测量值
measurement_noise = np.random.normal(0, 0.5, size=n)  # 测量噪声
measurements = real_position + measurement_noise

# 初始化卡尔曼滤波器参数
x_hat = np.zeros(n)  # 状态估计（位置）
P = np.ones(n)  # 协方差矩阵
A = 1# 状态转移矩阵
H = 1# 观测矩阵
Q = 0.1# 过程噪声协方差
R = 0.5# 测量噪声协方差
K = np.zeros(n)  # 卡尔曼增益

# 卡尔曼滤波算法
for k in range(1, n):
    # 预测步骤
    x_hat[k] = A * x_hat[k - 1]
    P[k] = A * P[k - 1] * A + Q

    # 更新步骤
    K[k] = P[k] * H / (H * P[k] * H + R)
    x_hat[k] = x_hat[k] + K[k] * (measurements[k] - H * x_hat[k])
    P[k] = (1 - K[k] * H) * P[k]

# 绘图：真实位置、测量位置与卡尔曼滤波估计位置
plt.figure(figsize=(10, 6))
plt.plot(np.arange(n) * dt, real_position, label='True Position', color='blue', linewidth=2)
plt.plot(np.arange(n) * dt, measurements, label='Noisy Measurements', color='red', alpha=0.6, linestyle='--')
plt.plot(np.arange(n) * dt, x_hat, label='Kalman Filter Estimate', color='green', linewidth=2)
plt.title('Kalman Filter Estimate - Position')
plt.xlabel('Time [s]')
plt.ylabel('Position')
plt.legend()
plt.grid(True)

# 绘制卡尔曼增益随时间的变化
plt.figure(figsize=(10, 6))
plt.plot(np.arange(n) * dt, K, label='Kalman Gain', color='purple', linewidth=2)
plt.title('Kalman Gain Over Time')
plt.xlabel('Time [s]')
plt.ylabel('Kalman Gain')
plt.legend()
plt.grid(True)

plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 极大熵估计

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

# 生成二维数据
np.random.seed(42)
mu_real = np.array([2, 3])
cov_real = np.array([[1.5, 0.5], [0.5, 1.0]])
data = np.random.multivariate_normal(mu_real, cov_real, size=500)

# 计算数据的均值和协方差
mu_empirical = np.mean(data, axis=0)
cov_empirical = np.cov(data, rowvar=False)

# 极大熵估计的优化目标函数
def entropy_objective(params):
    mu = np.array([params[0], params[1]])
    sigma1, sigma2, rho = params[2], params[3], params[4]
    cov_matrix = np.array([[sigma1**2, rho*sigma1*sigma2], [rho*sigma1*sigma2, sigma2**2]])
    return -multivariate_normal(mu, cov_matrix).entropy()

# 初始猜测
init_params = [mu_empirical[0], mu_empirical[1], np.sqrt(cov_empirical[0,0]), np.sqrt(cov_empirical[1,1]), 0]

# 约束确保协方差矩阵正定
constraints = [{'type': 'ineq', 'fun': lambda x: x[2]},  # sigma1 > 0
               {'type': 'ineq', 'fun': lambda x: x[3]},  # sigma2 > 0
               {'type': 'ineq', 'fun': lambda x: 1 - abs(x[4])}]  # -1 < rho < 1

# 优化
result = minimize(entropy_objective, init_params, constraints=constraints)
mu_maxent = result.x[:2]
sigma1_maxent, sigma2_maxent, rho_maxent = result.x[2], result.x[3], result.x[4]
cov_maxent = np.array([[sigma1_maxent**2, rho_maxent*sigma1_maxent*sigma2_maxent],
                        [rho_maxent*sigma1_maxent*sigma2_maxent, sigma2_maxent**2]])

# 可视化
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# 直方图
sns.histplot(data, bins=20, kde=True, ax=ax[0, 0])
ax[0, 0].set_title("Data Histogram")

# 拟合曲线
x, y = np.mgrid[-2:6:.1, 0:6:.1]
pos = np.dstack((x, y))
pdf = multivariate_normal(mu_maxent, cov_maxent).pdf(pos)
ax[0, 1].contourf(x, y, pdf, cmap="coolwarm")
ax[0, 1].set_title("Maximum Entropy Fitted Distribution")

# 热力图
sns.kdeplot(x=data[:, 0], y=data[:, 1], fill=True, cmap="viridis", ax=ax[1, 0])
ax[1, 0].set_title("Data Density Heatmap")

# 3D 概率密度图
ax3d = fig.add_subplot(2, 2, 4, projection='3d')
ax3d.plot_surface(x, y, pdf, cmap="plasma")
ax3d.set_title("3D Surface Plot of the Maximum Entropy Distribution")

plt.tight_layout()
plt.show()











