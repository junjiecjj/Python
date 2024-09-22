#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:04:16 2024

@author: jack
"""

#%%>>>>>>>>>>>>>> 9. BFGS (Broyden-Fletcher-Goldfarb-Shanno Algorithm)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 生成非线性数据
np.random.seed(42)
x_data = np.linspace(-10, 10, 100)
y_data = 5 * x_data**2 + 3 * x_data + 10 + np.random.normal(0, 10, size=x_data.shape)

# 定义二次多项式模型
def model(x, params):
    return params[0] * x**2 + params[1] * x + params[2]

# 定义损失函数（均方误差）
def loss(params, x, y):
    return np.mean((model(x, params) - y)**2)

# 初始参数猜测
initial_params = [1, 1, 1]

# 使用BFGS算法进行优化
result = minimize(loss, initial_params, args=(x_data, y_data), method='BFGS')

# 优化后的参数
optimized_params = result.x
print(f'Optimized parameters: {optimized_params}')

# 绘制原始数据与拟合曲线
plt.figure(figsize=(14, 7))

# 原始数据
plt.scatter(x_data, y_data, color='blue', label='Original Data')

# 拟合曲线
y_fit = model(x_data, optimized_params)
plt.plot(x_data, y_fit, color='red', label='Fitted Curve', linewidth=2)

plt.title('Original Data and Fitted Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 绘制损失函数值的收敛过程
plt.figure(figsize=(14, 7))
plt.plot(result.hess_inv.diagonal(), label='Loss Value')
plt.title('Convergence of Loss Function')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()


#%%>>>>>>>>>>>>>> 9. L-BFGS
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 定义目标函数（一个二次函数）
def objective_function(x):
    return 0.5 * (x[0]**2 + 3 * x[1]**2)

# 定义目标函数的梯度
def objective_gradient(x):
    return np.array([x[0], 3 * x[1]])

# 生成目标函数的等高线图数据
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = 0.5 * (X**2 + 3 * Y**2)

# 初始点
initial_point = np.array([2.0, 2.0])

# 使用L-BFGS算法进行优化
result = minimize(objective_function, initial_point, method='L-BFGS-B', jac=objective_gradient, options={'disp': True})

# 获取优化过程中记录的点
trajectory = result.x_iters if hasattr(result, 'x_iters') else [initial_point, result.x]

# 绘制图形
fig, ax = plt.subplots(1, 3, figsize=(12, 6))

# 1. 损失函数值的变化图
losses = [objective_function(p) for p in trajectory]
ax[0].plot(range(len(losses)), losses, marker='o', color='blue', label='Loss')
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Loss')
ax[0].set_title('Loss vs. Iteration')
ax[0].grid(True)
ax[0].legend()

# 2. 参数值的变化图
trajectory = np.array(trajectory)
ax[1].plot(trajectory[:, 0], marker='o', color='red', label='x1 (param 1)')
ax[1].plot(trajectory[:, 1], marker='o', color='green', label='x2 (param 2)')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Parameter Value')
ax[1].set_title('Parameter Values vs. Iteration')
ax[1].grid(True)
ax[1].legend()

# 3. 目标函数的等高线图和优化轨迹
ax[2] = plt.axes([0.7, 0.1, 0.25, 0.8])  # 调整axes位置，以便在右侧绘制
ax[2].contour(X, Y, Z, levels=30, cmap='coolwarm')
ax[2].plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='black', label='Optimization Path')
ax[2].plot(initial_point[0], initial_point[1], 'ro', label='Start')
ax[2].plot(result.x[0], result.x[1], 'go', label='End')
ax[2].set_xlabel('x1')
ax[2].set_ylabel('x2')
ax[2].set_title('Optimization Path on Contour Plot')
ax[2].legend()

plt.tight_layout()
plt.show()




