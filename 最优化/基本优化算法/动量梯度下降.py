#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:55:59 2024

@author: jack
"""

#%%>>>>>>>>>>>>>> 3. 动量法 (Momentum)

import numpy as np
import matplotlib.pyplot as plt

# 定义二次函数及其梯度
def func(x):
    return 0.5 * x**2

def grad(x):
    return x

# 初始化参数
x = 10  # 初始点
learning_rate = 0.1
momentum = 0.9 # 动量系数
velocity = 0
num_iterations = 50

# 存储优化过程中的值
x_values = []
func_values = []

# 动量法优化
for i in range(num_iterations):
    grad_val = grad(x)
    velocity = momentum * velocity + learning_rate * grad_val
    x -= velocity

    x_values.append(x)
    func_values.append(func(x))

# 绘制优化过程中的函数值变化
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(func_values, 'b-o')
plt.title('Function Value during Optimization')
plt.xlabel('Iteration')
plt.ylabel('Function Value')

plt.subplot(1, 2, 2)
plt.plot(x_values, 'r-o')
plt.title('x Value during Optimization')
plt.xlabel('Iteration')
plt.ylabel('x Value')

plt.tight_layout()
plt.show()

# 绘制函数和优化轨迹
x_range = np.linspace(-10, 10, 400)
y_range = func(x_range)

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_range, label='Function: $0.5x^2$')
# plt.scatter(x_values, func_values, color='red', label='Optimization Path', zorder=5)
plt.plot(x_values, func_values, 'r-o', label='Optimization Path', zorder=5)
plt.title('Optimization Path on the Function')
plt.xlabel('x')
plt.ylabel('Function Value')
plt.legend()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>>> 机器学习例子: 线性回归
import numpy as np
import matplotlib.pyplot as plt


# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)
# 添加偏置项 (X0 = 1)
X_b = np.c_[np.ones((len(X), 1)), X]

# 定义梯度下降函数
def momentum_gradient_descent(X, y, theta, learning_rate=0.01, iterations=100, momentum = 0.9):
    m = len(y)
    # 梯度下降优化
    theta_path = []
    history = {'cost': []}

    # 初始化参数
    momentum = momentum # 动量系数
    velocity = np.array([0]*X_b.shape[1]).reshape(-1,1)

    history = {'cost': []}
    for iteration in range(iterations):
        grad_val = 2/m * X.T @ (X@theta  - y)
        velocity = momentum * velocity + learning_rate * grad_val
        theta = theta - velocity
        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)

    return theta_path, history

# 初始化参数 theta
theta_initial = np.random.randn(X_b.shape[1], 1)
lr = 0.01
n_iterations = 1000
moment = 0.9

# 运行梯度下降算法
theta_path, history = momentum_gradient_descent(X_b, y, theta_initial, learning_rate = lr, iterations = n_iterations, momentum = moment)

# 将路径转换为数组
theta_path = np.array(theta_path)

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(0, n_iterations, 10):  # 每10次迭代绘制一次直线
    axs[0].plot(X, X_b@theta_path[i], color='red', alpha=0.5)
axs[0].plot(X, X_b.dot(theta_path[-1]), color='green', linewidth=4, label='Final Model')
axs[0].set_title('Linear Regression with Gradient Descent')
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

#%%>>>>>>>>>>>>>>>>>>>>>>>>> 机器学习例子: 多项式拟合
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + 2 * X**2 + np.random.randn(1000, 1)
# 增加多项式特征
X_b = np.c_[np.ones((1000, 1)), X, X**2]

# 初始化参数 theta
theta_initial = np.random.randn(X_b.shape[1], 1)
lr = 0.1
n_iterations = 1000
moment = 0.9

# 定义梯度下降函数
def momentum_gradient_descent(X, y, theta, learning_rate=0.01, iterations=100, momentum = 0.9):
    m = len(y)
    # 梯度下降优化
    theta_path = []
    history = {'cost': []}

    # 初始化参数
    momentum = momentum # 动量系数
    velocity = np.array([0]*X_b.shape[1]).reshape(-1,1)

    history = {'cost': []}
    for iteration in range(iterations):
        grad_val = 2/m * X.T @ (X@theta  - y)
        velocity = momentum * velocity + learning_rate * grad_val
        theta = theta - velocity
        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)

    return theta_path, history

# 运行梯度下降算法
theta_path, history = momentum_gradient_descent(X_b, y, theta_initial, learning_rate = lr, iterations = n_iterations, momentum = moment)

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
for i in range(0, n_iterations, 10):  # 每10次迭代绘制一次直线
    axs[0].plot(X_new, X_new_poly@theta_path[i], color='g', linewidth=2,)
axs[0].plot(X_new, y_predict, color='r', linewidth=4, label='Final Model')
axs[0].set_title('Polyomial Regression with Gradient Descent')
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



#%%>>>>>>>>>>>>>> 3. 动量法:解析
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 设置随机种子以便结果可重复
np.random.seed(42)

# 定义一个简单的二次函数：f(x) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# 定义函数的梯度
def gradient(x, y):
    return np.array([2*x, 2*y])

# 动量法优化
def Momentum_optimizer(start_x, start_y, lr, num_iterations, momentum = 0.9, perturbation = 0.1):
    # 初始化参数
    x = start_x
    y = start_y
    v_x, v_y = 0, 0

    # 用于存储历史数据
    history = []

    for i in range(num_iterations):
        # 计算梯度
        grad_x, grad_y = gradient(x, y)

        # 更新速度
        v_x = momentum * v_x - lr * (grad_x + perturbation * np.random.randn(1)[0])
        v_y = momentum * v_y - lr * (grad_y + perturbation * np.random.randn(1)[0])

        # 更新参数
        x = x + v_x
        y = y + v_y

        # 保存参数的历史记录
        history.append([x, y, f(x, y)])

    return x, y, f(x, y), np.array(history)


# 定义用于绘制函数的网格
x_range = np.arange(- 10 , 10 , 0.1 )
y_range = np.arange(- 10 , 10 , 0.1 )
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

# 执行梯度下降并绘制结果
start_x, start_y = 8 , 8
learning_rate = 0.1
num_iterations = 15
momu = 0.9
pb = 2

# 运行动量优化器
x_opt, y_opt, f_opt, history = Momentum_optimizer(start_x, start_y, learning_rate, num_iterations, momentum = momu, perturbation = pb)

## 3D
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8), constrained_layout=True)
ax.set_proj_type('ortho')

norm_plt = plt.Normalize(Z.min(), Z.max())
colors = cm.RdYlBu_r(norm_plt(Z))
ax.plot_wireframe(X, Y, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5) # color = '#0070C0',

ax.plot(history[:,0], history[:,1], history[:,2], c = 'b' , marker= 'o', ms = 5 )
ax.scatter(history[-1,0], history[-1,1], history[-1,2], c = 'r' , marker= '*', s = 40, zorder = 10 )
ax.set_proj_type('ortho')

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
ax.set_xlabel(r'$\it{x_1}$', fontdict = font, labelpad = 2)
ax.set_ylabel(r'$\it{x_2}$', fontdict = font, labelpad = 2)
ax.set_zlabel(r'$\it{f}(\it{x_1},\it{x_2}$)', fontdict = font,  )

ax.tick_params(axis='both', direction='in', width=3, length = 5,  labelsize=15, labelfontfamily = 'Times New Roman', pad = 1)

# ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/只保留网格线.svg', format='svg')
plt.show()

## 2D courter
fig, ax = plt.subplots( figsize = (7.5, 6), constrained_layout=True)
CS = ax.contour(X, Y, Z, levels = 20, cmap = 'RdYlBu_r', linewidths = 1)
fig.colorbar(CS)

ax.plot(history[:,0], history[:,1],  c = 'b' , marker= 'o', ms = 5 )
ax.scatter(history[-1,0], history[-1,1],  c = 'r' , marker= '*', s = 200, zorder = 10)
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
ax.set_xlabel(r'$\it{x_1}$', fontdict = font, labelpad = 2)
ax.set_ylabel(r'$\it{x_2}$', fontdict = font, labelpad = 2)

ax.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=15, labelfontfamily = 'Times New Roman', pad = 1)

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())

ax.grid(False)
# fig.savefig('Figures/只保留网格线.svg', format='svg')
plt.show()




#%%>>>>>>>>>>>>>> 3. 动量法: 解析
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 设置随机种子以便结果可重复
np.random.seed(42)

def f(x, y):
    return (1-y**5+x**5)*np.exp(-x**2-y**2)


def gradient(x, y):
    gd_x = (5 * x**4 * np.exp(-x**2-y**2) - 2*x * (1 - y**5 + x**5) * np.exp(-x**2-y**2))
    gd_y = (-5 * y**4 * np.exp(-x**2-y**2) - 2*y * (1-y**5 + x**5) * np.exp(-x**2-y**2))
    return np.array([gd_x, gd_y])

# 动量法优化
def Momentum_optimizer(start_x, start_y, lr, num_iterations, momentum = 0.9, perturbation = 0.1):
    # 初始化参数
    x = start_x
    y = start_y
    v_x, v_y = 0, 0

    # 用于存储历史数据
    history = []

    for i in range(num_iterations):
        # 计算梯度
        grad_x, grad_y = gradient(x, y)

        # 更新速度
        v_x = momentum * v_x - lr * (grad_x + perturbation * np.random.randn(1)[0])
        v_y = momentum * v_y - lr * (grad_y + perturbation * np.random.randn(1)[0])

        # 更新参数
        x = x + v_x
        y = y + v_y

        # 保存参数的历史记录
        history.append([x, y, f(x, y)])
    # print(f"{x}, {y}, {}")
    return x, y, f(x, y), np.array(history)



# 定义用于绘制函数的网格
x_range = np.arange(-2 , 2 , 0.1 )
y_range = np.arange(-2 , 2 , 0.1 )
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

# 执行梯度下降并绘制结果
start_x, start_y = 0 , 0
learning_rate = 0.06
num_iterations = 15
momu = 0.9
pb = 0.1

# 运行动量优化器
x_opt, y_opt, f_opt, history = Momentum_optimizer(start_x, start_y, learning_rate, num_iterations, momentum = momu, perturbation = pb)

## 3D
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8), constrained_layout=True)
ax.set_proj_type('ortho')

norm_plt = plt.Normalize(Z.min(), Z.max())
colors = cm.RdYlBu_r(norm_plt(Z))
ax.plot_wireframe(X, Y, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5) # color = '#0070C0',

ax.plot(history[:,0], history[:,1], history[:,2], c = 'b' , marker= 'o', ms = 5 )
ax.scatter(history[-1,0], history[-1,1], history[-1,2], c = 'r' , marker= '*', s = 40, zorder = 10 )
ax.set_proj_type('ortho')

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
ax.set_xlabel(r'$\it{x_1}$', fontdict = font, labelpad = 2)
ax.set_ylabel(r'$\it{x_2}$', fontdict = font, labelpad = 2)
ax.set_zlabel(r'$\it{f}(\it{x_1},\it{x_2}$)', fontdict = font,  )

ax.tick_params(axis='both', direction='in', width=3, length = 5,  labelsize=15, labelfontfamily = 'Times New Roman', pad = 1)

# ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/只保留网格线.svg', format='svg')
plt.show()

## 2D courter
fig, ax = plt.subplots( figsize = (7.5, 6), constrained_layout=True)
CS = ax.contour(X, Y, Z, levels = 30, cmap = 'RdYlBu_r', linewidths = 1)
fig.colorbar(CS)

ax.plot(history[:,0], history[:,1],  c = 'b' , marker= 'o', ms = 5 )
ax.scatter(history[-1,0], history[-1,1],  c = 'r' , marker= '*', s = 200, zorder = 10)
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
ax.set_xlabel(r'$\it{x_1}$', fontdict = font, labelpad = 2)
ax.set_ylabel(r'$\it{x_2}$', fontdict = font, labelpad = 2)

ax.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=15, labelfontfamily = 'Times New Roman', pad = 1)

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())

ax.grid(False)
# fig.savefig('Figures/只保留网格线.svg', format='svg')
plt.show()


























