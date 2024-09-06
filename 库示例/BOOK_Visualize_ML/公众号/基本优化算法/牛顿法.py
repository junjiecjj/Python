#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:02:05 2024

@author: jack
"""

#%%>>>>>>>>>>>>>> 7. 牛顿法 (Newton's Method)
import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数和其一阶、二阶导数
def f(x):
    return x**2 + 3*x + 2

def f_prime(x):
    return 2*x + 3

def f_double_prime(x):
    return 2

# 牛顿法优化函数
def newtons_method(x0, tol=1e-6, max_iter=100):
    x = x0
    iters = 0
    history = [x]

    while iters < max_iter:
        x_new = x - f_prime(x) / f_double_prime(x)
        history.append(x_new)
        if abs(x_new - x) < tol:
            break
        x = x_new
        iters += 1

    return x, history

# 初始值和优化
x0 = 10
optimal_x, history = newtons_method(x0)

# 生成x值和对应的函数值用于绘图
x_vals = np.linspace(-10, 10, 400)
y_vals = f(x_vals)

# 图形1：目标函数和优化路径
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, label='f(x) = x^2 + 3x + 2')
plt.scatter(history, f(np.array(history)), color='red')
for i, x in enumerate(history):
    plt.text(x, f(x), f'{i}', fontsize=12, color='red')
plt.title('Newton\'s Method Optimization Path')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# 图形2：迭代次数 vs 函数值
plt.subplot(1, 2, 2)
iter_nums = np.arange(len(history))
function_values = f(np.array(history))
plt.plot(iter_nums, function_values, marker='o')
plt.title('Iteration vs Function Value')
plt.xlabel('Iteration')
plt.ylabel('f(x)')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>> 10. 牛顿法 (Newton's Method), 机器学习例子: 线性回归
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)

# 添加偏置项 x0 = 1 到 X 中
X_b = np.c_[np.ones((1000, 1)), X]

def Newton(X, y, theta, tol=1e-6, iterations=100):
    m = len(y)
    theta = theta_initial
    theta_path = []
    theta_path.append(theta)
    history = {'cost': []}
    cost = np.mean((X @ theta  - y) ** 2)
    history['cost'].append(cost)

    for iteration in range(iterations):
        theta_new = theta - np.linalg.inv(2/m * X.T @ X) @ (2/m * X.T @ (X@theta  - y) )
        if np.abs(theta_new - theta).sum() < tol:
            break
        theta = theta_new
        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history

# 初始化参数 theta
theta_initial = np.random.randn(2, 1)
n_iterations = 100
tol = 1e-30

# 运行梯度下降算法
theta_path, history = Newton(X_b, y, theta_initial, tol=tol, iterations = n_iterations)

# 将路径转换为数组
theta_path = np.array(theta_path)

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(theta_path.shape[0]):
    axs[0].plot(X, X_b@theta_path[i], color='green', alpha=0.5)
axs[0].plot(X, X_b.dot(theta_path[-1]), color='red', linewidth=4, label='Final Model')
axs[0].set_title('Linear Regression with ConjugateGradient')
axs[0].set_xlabel('X')
axs[0].set_ylabel('y')
axs[0].legend()

# 绘制损失函数的变化
axs[1].plot(range(len(history['cost'])), history['cost'], color='red', linewidth=2, label='Loss Function')
axs[1].set_title('Loss Function Over Iterations')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Mean Squared Error')
axs[1].legend()

# 显示图形
plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>> 10. 牛顿法 (Newton's Method), 机器学习例子: 多项式回归
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + 2 * X**2 + np.random.randn(1000, 1)
# 增加多项式特征
X_b = np.c_[np.ones((1000, 1)), X, X**2]

def Newton(X, y, theta, tol=1e-6, iterations=100):
    m = len(y)
    theta = theta_initial
    theta_path = []
    theta_path.append(theta)
    history = {'cost': []}
    cost = np.mean((X @ theta  - y) ** 2)
    history['cost'].append(cost)

    for iteration in range(iterations):
        theta_new = theta - np.linalg.inv(2/m * X.T @ X) @ (2/m * X.T @ (X@theta  - y) )
        if np.abs(theta_new - theta).sum() < tol:
            break
        theta = theta_new
        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history

# 初始化参数 theta
theta_initial = np.random.randn(X_b.shape[1], 1)
n_iterations = 100
tol = 1e-40

# 运行梯度下降算法
theta_path, history = Newton(X_b, y, theta_initial, tol=tol, iterations = n_iterations)

# 将路径转换为数组
theta_path = np.array(theta_path)

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 生成用于画图的曲线
X_new = np.linspace(0, 2, 100).reshape(100, 1)
X_new_poly = np.c_[np.ones((100, 1)), X_new, X_new**2]
y_predict = X_new_poly@theta_path[-1]

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(len(theta_path)):  # 每10次迭代绘制一次直线
    axs[0].plot(X_new, X_new_poly@theta_path[i], color='g', linewidth=2,)
axs[0].plot(X_new, y_predict, color='r', linewidth=4, label='Final Model')
axs[0].set_title('Polyomial Regression with ConjugateGradient')
axs[0].set_xlabel('X')
axs[0].set_ylabel('y')
axs[0].legend()

# 绘制损失函数的变化
axs[1].plot(range(len(history['cost'])), history['cost'], color='red', linewidth=2, label='Loss Function')
axs[1].set_title('Loss Function Over Iterations')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Mean Squared Error')
axs[1].legend()

# 显示图形
plt.tight_layout()
plt.show()













