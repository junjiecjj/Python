#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:03:18 2024

@author: jack
"""


#%%>>>>>>>>>>>>>> 8. 共轭梯度法 (Conjugate Gradient Method)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义二次函数
def quadratic_function(x):
    return 0.5 * (x[0]**2 + 10 * x[1]**2)

# 定义二次函数的梯度
def quadratic_gradient(x):
    return np.array([x[0], 10 * x[1]])

# 共轭梯度法实现
def conjugate_gradient(f, grad_f, x0, iterations=50):
    x = x0
    trajectory = [x]
    gradient = grad_f(x)
    direction = -gradient

    for _ in range(iterations):
        step_size = 0.1  # 步长
        x = x + step_size * direction
        next_gradient = grad_f(x)
        beta = np.dot(next_gradient, next_gradient) / np.dot(gradient, gradient)
        direction = -next_gradient + beta * direction
        gradient = next_gradient
        trajectory.append(x)
    return np.array(trajectory)

# 初始点和优化
x0 = np.array([3.0, 4.0])
trajectory = conjugate_gradient(quadratic_function, quadratic_gradient, x0)

# 绘制函数曲面和优化轨迹
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = quadratic_function([X, Y])

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('Quadratic Function')

ax1.plot(trajectory[:, 0], trajectory[:, 1], quadratic_function(trajectory.T), color='r', marker='o')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(X, Y)')

# 绘制优化过程中的收敛曲线
iterations = len(trajectory)
steps = np.arange(iterations)

ax2 = fig.add_subplot(122)
ax2.plot(steps, trajectory[:, 0], label='X')
ax2.plot(steps, trajectory[:, 1], label='Y')
ax2.set_title('Convergence of Variables')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Variable Value')
ax2.legend()

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



# 共轭梯度法实现
def ConjugateGradientMethod(X, y, theta_initial, iterations=100):
    # 初始化参数
    m = len(y)
    history = {'cost': []}

    theta = theta_initial
    # trajectory = [theta]
    gradient = 2/m * X.T @ (X @ theta - y)
    direction = -gradient

    for _ in range(iterations):
        step_size = 0.1  # 步长
        theta = theta + step_size * direction
        next_gradient = 2/m * X.T @ (X @ theta - y)
        beta = ((next_gradient.T @ next_gradient) / (gradient.T @ gradient) )[0,0]
        direction = - next_gradient + beta * direction
        gradient = next_gradient
        # trajectory.append(theta)

        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)

    return theta, history

# 初始化参数 theta
theta_initial = np.random.randn(2, 1)

# 运行梯度下降算法
theta_best, history = ConjugateGradientMethod(X_b, y, theta_initial, iterations = 100)

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














