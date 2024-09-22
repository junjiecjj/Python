#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:44:56 2024

@author: jack
"""


#%%>>>>>>>>>>>>>> 1. 随机梯度下降
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)
# 添加偏置项 (X0 = 1)
X_b = np.c_[np.ones((len(X), 1)), X]

# 初始化参数 theta
theta_initial = np.random.randn(2, 1)
lr = 0.1
n_iterations = 1000
bs = 20

# 定义梯度下降函数
def stochastic_gradient_descent(X, y, theta, batchsize = 10, learning_rate=0.01, iterations=100):
    m = len(y)
    # 梯度下降优化
    theta_path = []
    history = {'cost': []}

    for iteration in range(iterations):
        idx = np.random.choice(m, batchsize)
        X_bs = X[idx]; y_bs = y[idx]
        gradients = 2/batchsize * X_bs.T @ (X_bs@theta  - y_bs)
        theta = theta - learning_rate * gradients
        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history

# 运行梯度下降算法
theta_path, history = stochastic_gradient_descent(X_b, y, theta_initial, batchsize = bs, learning_rate = lr, iterations = n_iterations)

# 将路径转换为数组
theta_path = np.array(theta_path)

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(0, n_iterations, 10):  # 每10次迭代绘制一次直线
    axs[0].plot(X, X_b.dot(theta_path[i]), color='red', alpha=0.5)
axs[0].plot(X, X_b.dot(theta_path[-1]), color='green', linewidth=4, label='Final Model')
axs[0].set_title('Linear Regression with stochastic GD')
axs[0].set_xlabel('X')
axs[0].set_ylabel('y')
axs[0].legend()

# 绘制损失函数的变化
axs[1].plot(range(n_iterations), history['cost'], color='red', linewidth=2, label='Loss Function')
axs[1].set_title('Loss Function Over Iterations')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Mean Squared Error')
axs[1].legend()

# 显示图形
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>> 1. 随机梯度下降
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + 2 * X**2 + np.random.randn(1000, 1)
# 增加多项式特征
X_b = np.c_[np.ones((1000, 1)), X, X**2]

# 初始化参数
theta_initial = np.random.randn(3, 1)  # 两个参数：截距和斜率
lr = 0.1
n_iterations = 1000
bs = 20

# 定义梯度下降函数
def stochastic_gradient_descent(X, y, theta, batchsize = 10, learning_rate=0.01, iterations=100):
    m = len(y)
    # 梯度下降优化
    theta_path = []
    history = {'cost': []}

    for iteration in range(iterations):
        idx = np.random.choice(m, batchsize)
        X_bs = X[idx]; y_bs = y[idx]
        gradients = 2/batchsize * X_bs.T @ (X_bs @ theta  - y_bs)
        theta = theta - learning_rate * gradients
        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history

# 运行梯度下降算法
theta_path, history = stochastic_gradient_descent(X_b, y, theta_initial, batchsize = bs, learning_rate = lr, iterations = n_iterations)
# 将路径转换为数组
theta_path = np.array(theta_path)

# 生成用于画图的曲线
X_new = np.linspace(0, 2, 100).reshape(100, 1)
X_new_poly = np.c_[np.ones((100, 1)), X_new, X_new**2]
y_predict = X_new_poly@theta_path[-1]

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(0, n_iterations, 10):  # 每10次迭代绘制一次直线
    axs[0].plot(X_new, X_new_poly@theta_path[i], color='red', alpha=0.5)
axs[0].plot(X_new, y_predict, color='green', linewidth=4, label='Final Model')
axs[0].set_title('Polyomial Regression with stochastic GD')
axs[0].set_xlabel('X')
axs[0].set_ylabel('y')
axs[0].legend()

# 绘制损失函数的变化
axs[1].plot(range(n_iterations), history['cost'], color='red', linewidth=2, label='Loss Function')
axs[1].set_title('Loss Function Over Iterations')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Mean Squared Error')
axs[1].legend()

# 显示图形
plt.tight_layout()
plt.show()

