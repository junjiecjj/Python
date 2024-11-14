#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:35:57 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247485956&idx=1&sn=abac1ec541bf3c8f51cf38114c22fc0d&chksm=c1ae6c993e3698811a85a706a4389cc6d8d8ffbdab05eb7462452868215344fa76287aa3da3b&mpshare=1&scene=1&srcid=0822GQ8QZOOaAbFcHOYS6ozJ&sharer_shareinfo=792b67ed4ccef1bee1c59d47e8285e91&sharer_shareinfo_first=792b67ed4ccef1bee1c59d47e8285e91&exportkey=n_ChQIAhIQNvSF0aFR9HfJVszrGewg%2BhKfAgIE97dBBAEAAAAAALyqIGEjphAAAAAOpnltbLcz9gKNyK89dVj0M7UjKheWmvj7E62WcFIB2ejlGDIP%2BD39Lj7wRBB%2FKOBEHidvMsrcbhWl3CfGBb9ThSAfcizDTQW1OWhW8npbVbzVLLFl2k%2B8vjBde50MIWp6Mnl02PbpqaBbY7H3r8zrV9PYsJ5cgYd4XLTg11uFwmTUIW6L%2Fm4P34sLQIjSyCSPWPT5tDbJpkR7rph3%2F9qRmqmBzzArTLwkM2RMS0SeAVWTtTZfOLpNNuMh%2Bd7UksOLQI7rl581mG8FcM9ts9zGEoYAK%2Biyj7%2FsHhjITPyU0AsrS6aql4%2BPx1tiUYF2qe8FUB5BCawmA8QOJmXbfbeTS6FB4zkI%2FNVk&acctmode=0&pass_ticket=4JksxdRP9ZK%2BlfuY8ugpV87Z0z2yUPk0b12Oab%2FDhhovXUUPFL8AK5gxyje%2FNWTP&wx_header=0#rd


"""
#%%>>>>>>>>>>>>>>>>>>>>>>> 1. 最大似然估计 (Maximum Likelihood Estimation, MLE)

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
mle_mu, mle_sigma = result.x

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

#%%>>>>>>>>>>>>>>>>>>>>>>> 3. 最小二乘估计 (Ordinary Least Squares, OLS)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 生成虚拟数据集
np.random.seed(42)
n_samples = 1000
X1 = np.random.rand(n_samples)
X2 = np.random.rand(n_samples)
noise = np.random.normal(0, 0.1, n_samples)
Y = 3 * X1 + 2 * X2 + noise  # Y = 3*X1 + 2*X2 + noise

# 将X1和X2合并成一个2D矩阵
X = np.vstack((X1, X2)).T

# 使用OLS进行线性回归
model = LinearRegression()
model.fit(X, Y)

# 预测
Y_pred = model.predict(X)

# 计算残差
residuals = Y - Y_pred
mse = mean_squared_error(Y, Y_pred)

# 绘制图形
fig = plt.figure(figsize=(14, 10))

# 第一个子图：3D散点图和回归平面
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X1, X2, Y, color='blue', label='Data Points')
ax1.plot_trisurf(X1, X2, Y_pred, color='red', alpha=0.5, label='Regression Plane')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('Y')
ax1.set_title('3D Scatter Plot with Regression Plane')

# 第二个子图：实际值 vs 预测值
ax2 = fig.add_subplot(222)
ax2.scatter(Y, Y_pred, color='green')
ax2.plot([min(Y), max(Y)], [min(Y), max(Y)], color='red', linestyle='--')
ax2.set_xlabel('Actual Y')
ax2.set_ylabel('Predicted Y')
ax2.set_title('Actual vs Predicted Y')

# 第三个子图：残差图
ax3 = fig.add_subplot(223)
ax3.scatter(Y_pred, residuals, color='purple')
ax3.axhline(y=0, color='red', linestyle='--')
ax3.set_xlabel('Predicted Y')
ax3.set_ylabel('Residuals')
ax3.set_title('Residual Plot')

# 第四个子图：X1、X2对Y的影响
ax4 = fig.add_subplot(224)
ax4.scatter(X1, Y, color='blue', alpha=0.6, label='X1')
ax4.scatter(X2, Y, color='orange', alpha=0.6, label='X2')
ax4.set_xlabel('X1 or X2')
ax4.set_ylabel('Y')
ax4.legend()
ax4.set_title('X1 and X2 vs Y')

plt.suptitle(f'OLS Regression Analysis (MSE: {mse:.3f})')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 4. 岭回归 (Ridge Regression)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 生成虚拟数据集
np.random.seed(42)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)

# 生成线性模型的真实系数，并添加噪声
true_coefs = np.random.randn(n_features)
y = X.dot(true_coefs) + np.random.normal(scale=0.5, size=n_samples)

# 数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义不同的岭回归alpha参数
alphas = [0.1, 1, 10, 100]

# 用于存储不同alpha下的模型系数
coefs = []

# 创建图形窗口
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# 岭回归模型训练和预测
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    # 记录模型系数
    coefs.append(ridge.coef_)

    # 绘制实际值与预测值的关系图
    ax[0].scatter(y_test, y_pred, label=f'alpha={alpha:.1f}')
    ax[0].set_xlabel('Actual Values')
    ax[0].set_ylabel('Predicted Values')
    ax[0].set_title('Actual vs Predicted Values')
    ax[0].legend()
    ax[0].grid(True)

# 转置系数矩阵以便绘图
coefs = np.array(coefs).T

# 绘制不同alpha值下模型系数的变化
for coef, feature in zip(coefs, range(n_features)):
    ax[1].plot(alphas, coef, marker='o', label=f'Feature {feature+1}')

ax[1].set_xscale('log')
ax[1].set_xlabel('alpha')
ax[1].set_ylabel('Coefficient Value')
ax[1].set_title('Coefficient Paths')
ax[1].legend()
ax[1].grid(True)

# 显示图形
plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 5. Lasso回归 (Lasso Regression)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 设置随机种子
np.random.seed(42)

# 生成虚拟数据
n_samples, n_features = 1000, 10
X = np.random.randn(n_samples, n_features)

# 设定真实的系数，其中只有前4个变量有实际的贡献
true_coefs = np.array([5, -3, 2, 1] + [0] * (n_features - 4))

# 生成目标变量
y = np.dot(X, true_coefs) + np.random.normal(0, 1, size=n_samples)

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义不同的Lasso正则化强度
alphas = np.logspace(-2, 1, 10)

# 保存结果
coefs = []
errors = []

# 训练不同正则化强度下的Lasso回归模型
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
    y_pred = lasso.predict(X_test)
    errors.append(mean_squared_error(y_test, y_pred))

# 转换结果为numpy数组以便于绘图
coefs = np.array(coefs)
errors = np.array(errors)

# 绘制图形
fig, ax1 = plt.subplots(figsize=(12, 8))

# 绘制Lasso系数路径图
ax1.plot(alphas, coefs, marker='o')
ax1.set_xscale('log')
ax1.set_xlabel('Alpha (Regularization Strength)')
ax1.set_ylabel('Coefficients')
ax1.set_title('Lasso Paths and Prediction Error')
ax1.grid(True)

# 在相同图形上添加第二个y轴，用于绘制测试误差
ax2 = ax1.twinx()
ax2.plot(alphas, errors, color='red', linestyle='--', marker='x', label='MSE')
ax2.set_ylabel('Mean Squared Error')
ax2.legend(loc='upper left')

# 显示图形
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 6. 最小化均方误差 (Minimization of Mean Squared Error, MSE)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 生成虚拟数据
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
# 真实模型是三次多项式
y_true = 0.5 * X**3 - 2 * X**2 + X + 2
# 添加噪声
y = y_true + np.random.normal(scale=3, size=X.shape)

# 准备图表
plt.figure(figsize=(15, 10))

# 绘制原始数据点
plt.subplot(2, 2, 1)
plt.scatter(X, y, color='black', label='Data with noise')
plt.plot(X, y_true, color='red', label='True polynomial (degree 3)', linewidth=2)
plt.title('Original Data and True Polynomial')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# 不同阶数的多项式拟合和MSE分析
degrees = [1, 2, 3, 5]
colors = ['blue', 'green', 'orange', 'purple']

# 绘制多项式拟合曲线
plt.subplot(2, 2, 2)
for i, degree in enumerate(degrees):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)

    plt.plot(X, y_pred, color=colors[i], label=f'Degree {degree}')

plt.scatter(X, y, color='black', label='Data with noise')
plt.plot(X, y_true, color='red', label='True polynomial (degree 3)', linewidth=2)
plt.title('Polynomial Fits of Different Degrees')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# 绘制MSE随多项式阶数变化的图
mse_list = []
plt.subplot(2, 2, 3)
for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)

    mse = mean_squared_error(y, y_pred)
    mse_list.append(mse)
    print(f"Degree {degree}, MSE: {mse}")

plt.plot(degrees, mse_list, marker='o', color='red')
plt.title('MSE vs. Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.xticks(degrees)
plt.grid(True)

# 绘制多项式拟合的残差图
plt.subplot(2, 2, 4)
for i, degree in enumerate(degrees):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)

    residuals = y - y_pred
    plt.scatter(X, residuals, color=colors[i], label=f'Degree {degree}')

plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residuals of Polynomial Fits')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.legend()

plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 7. 期望最大化算法 (Expectation-Maximization, EM)
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

# 运行EM算法
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

#%%>>>>>>>>>>>>>>>>>>>>>>> 8. 梯度下降 (Gradient Descent)

import numpy as np
import matplotlib.pyplot as plt

############# 生成线性回归数据
np.random.seed(42)
X_lin = 2 * np.random.rand(100, 1)
y_lin = 4 + 3 * X_lin + np.random.randn(100, 1)

# 线性回归梯度下降
X_b = np.c_[np.ones((100, 1)), X_lin]  # 添加x0 = 1
theta = np.random.randn(2, 1)
learning_rate = 0.1
n_iterations = 1000
m = 100

loss_history_lin = []
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y_lin)
    theta -= learning_rate * gradients
    loss = np.mean((X_b.dot(theta) - y_lin) ** 2)
    loss_history_lin.append(loss)

# 绘制结果
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 线性回归数据和拟合曲线
ax[0].scatter(X_lin, y_lin, color="blue", label="Data Points")
ax[0].plot(X_lin, X_b.dot(theta), color="red", label="Linear Regression Fit")
ax[0].set_title("Linear Regression Fit")
ax[0].legend()

# 线性回归损失随迭代的变化
ax[1].plot(range(n_iterations), loss_history_lin, color="green")
ax[1].set_title("Linear Regression Loss")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Loss")


############# 生成逻辑回归数据
X_log = np.random.rand(100, 2) * 10 - 5
y_log = (X_log[:, 0] * 0.5 + X_log[:, 1] * 0.3 + np.random.randn(100) > 0).astype(int)

# 逻辑回归梯度下降
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X_log_b = np.c_[np.ones((100, 1)), X_log]
theta_log = np.random.randn(3, 1)
learning_rate = 0.1
n_iterations = 1000

loss_history_log = []
for iteration in range(n_iterations):
    logits = X_log_b.dot(theta_log)
    h = sigmoid(logits)
    gradients = 1/m * X_log_b.T.dot(h - y_log.reshape(-1, 1))
    theta_log -= learning_rate * gradients
    loss = -np.mean(y_log * np.log(h) + (1 - y_log) * np.log(1 - h))
    loss_history_log.append(loss)

# 绘制结果
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 逻辑回归数据分布和决策边界
ax[0].scatter(X_log[y_log == 0][:, 0], X_log[y_log == 0][:, 1], color="blue", label="Class 0")
ax[0].scatter(X_log[y_log == 1][:, 0], X_log[y_log == 1][:, 1], color="red", label="Class 1")
x_values = [np.min(X_log[:, 0] - 1), np.max(X_log[:, 0] + 1)]
y_values = -(theta_log[0] + theta_log[1] * x_values) / theta_log[2]
ax[0].plot(x_values, y_values, label="Decision Boundary", color="green")
ax[0].set_title("Logistic Regression Decision Boundary")
ax[0].legend()

# 逻辑回归损失随迭代的变化
ax[1].plot(range(n_iterations), loss_history_log, color="purple")
ax[1].set_title("Logistic Regression Loss")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Loss")

plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 9. 最小化交叉熵 (Minimization of Cross-Entropy)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# 生成虚拟数据集
X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建逻辑回归模型并训练
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

# 计算训练和测试集上的交叉熵损失
y_train_prob = model.predict_proba(X_train)[:, 1]
y_test_prob = model.predict_proba(X_test)[:, 1]

train_loss = log_loss(y_train, y_train_prob)
test_loss = log_loss(y_test, y_test_prob)

# 创建网格以绘制决策边界
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                     np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制图形
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# 图1：数据分布和决策边界
ax[0].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
scatter = ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
legend1 = ax[0].legend(*scatter.legend_elements(), title="Classes")
ax[0].add_artist(legend1)
ax[0].set_title("Data Distribution with Decision Boundary")
ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Feature 2")

# 图2：训练和测试集上的交叉熵损失
ax[1].bar(['Train Loss', 'Test Loss'], [train_loss, test_loss], color=['blue', 'orange'])
ax[1].set_title("Cross-Entropy Loss")
ax[1].set_ylim(0, 1)
ax[1].set_ylabel("Log Loss")

# 图3：训练过程中的损失函数曲线（虚拟）
# 假设有一个训练过程记录的虚拟损失函数
epochs = np.arange(1, 21)
train_loss_curve = np.linspace(0.9, train_loss, len(epochs))
test_loss_curve = np.linspace(0.9, test_loss, len(epochs))

ax[2].plot(epochs, train_loss_curve, label='Train Loss')
ax[2].plot(epochs, test_loss_curve, label='Test Loss', linestyle='--')
ax[2].set_title("Training Process: Loss Curve")
ax[2].set_xlabel("Epochs")
ax[2].set_ylabel("Log Loss")
ax[2].legend()

plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 10. 卡尔曼滤波器 (Kalman Filter)

import numpy as np
import matplotlib.pyplot as plt

# 模拟数据集生成
np.random.seed(42)  # 固定随机数种子

# 时间步数
n_timesteps = 50

# 实际初始状态 (位置=0，速度=1)
true_initial_position = 0
true_initial_velocity = 1

# 实际状态 (位置和速度) 的噪声
process_noise_std_position = 0.1
process_noise_std_velocity = 0.1

# 观测噪声 (位置的测量噪声)
measurement_noise_std = 0.5

# 状态和观测矩阵
A = np.array([[1, 1],  # 状态转移矩阵 (状态更新)
              [0, 1]])  # 速度保持恒定
H = np.array([[1, 0]])  # 观测矩阵 (仅观测位置)

# 初始化状态
true_states = np.zeros((n_timesteps, 2))
true_states[0] = [true_initial_position, true_initial_velocity]

# 初始化观测
measurements = np.zeros(n_timesteps)

# 生成实际状态和观测值
for t in range(1, n_timesteps):
    process_noise = np.random.normal(0, [process_noise_std_position, process_noise_std_velocity])
    true_states[t] = A @ true_states[t-1] + process_noise
    measurements[t] = H @ true_states[t] + np.random.normal(0, measurement_noise_std)

# 卡尔曼滤波器
# 初始化估计的状态
estimated_states = np.zeros((n_timesteps, 2))
estimated_states[0] = [0, 0]  # 初始位置和速度的估计值
P = np.eye(2)  # 初始估计误差协方差矩阵

# 定义过程噪声协方差矩阵和测量噪声协方差矩阵
Q = np.array([[process_noise_std_position**2, 0],
              [0, process_noise_std_velocity**2]])  # 过程噪声协方差
R = measurement_noise_std**2  # 测量噪声协方差

# 卡尔曼滤波器迭代
for t in range(1, n_timesteps):
    # 预测阶段
    predicted_state = A @ estimated_states[t-1]
    P = A @ P @ A.T + Q

    # 更新阶段
    y_tilde = measurements[t] - H @ predicted_state
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)  # 卡尔曼增益

    estimated_states[t] = predicted_state + K @ y_tilde
    P = (np.eye(2) - K @ H) @ P

# 绘制结果
plt.figure(figsize=(12, 8))

# 图1：真实位置 vs 估计位置 vs 观测位置
plt.subplot(2, 1, 1)
plt.plot(true_states[:, 0], label='True Position', color='g')
plt.plot(estimated_states[:, 0], label='Estimated Position', color='b', linestyle='--')
plt.scatter(range(n_timesteps), measurements, label='Measured Position', color='r', marker='o')
plt.title('Position: True vs Estimated vs Measured')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()

# 图2：真实速度 vs 估计速度
plt.subplot(2, 1, 2)
plt.plot(true_states[:, 1], label='True Velocity', color='g')
plt.plot(estimated_states[:, 1], label='Estimated Velocity', color='b', linestyle='--')
plt.title('Velocity: True vs Estimated')
plt.xlabel('Time Step')
plt.ylabel('Velocity')
plt.legend()

plt.tight_layout()
plt.show()




















































