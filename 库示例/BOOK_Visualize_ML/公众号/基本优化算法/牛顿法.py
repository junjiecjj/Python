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


########## 机器学习例子
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)

# 绘制原始数据散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Original data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

# 添加偏置项 x0 = 1 到 X 中
X_b = np.c_[np.ones((1000, 1)), X]

# 定义梯度下降函数
def gradient_descentNewton(X, y, theta, tol=1e-6, iterations=100):
    # 初始化参数
    m = len(y)
    history = {'cost': []}
    for iteration in range(iterations):
        theta_new = theta - np.linalg.inv(2/m * X.T @ X) @ (2/m * X.T @ (X@theta  - y) )
        if np.abs(theta_new - theta).sum() < tol:
            break
        theta = theta_new
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)

    return theta, history

# 初始化参数 theta
theta_initial = np.random.randn(2, 1)

# 运行梯度下降算法
theta_best, history = gradient_descentNewton(X_b, y, theta_initial, tol=1e-6, iterations = 100)

# 绘制损失函数变化曲线
plt.figure(figsize=(10, 6))
plt.plot(history['cost'], c='r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.grid(True)
plt.show()

# 绘制拟合曲线和数据散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Original data')
plt.plot(X, X_b@theta_best , c='r', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Fitted Line')
plt.legend()
plt.grid(True)
plt.show()


















