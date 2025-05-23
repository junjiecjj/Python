#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:37:34 2024

@author: jack
"""

#%%>>>>>>>>>>>>>> 10. 共轭梯度法 (Conjugate Gradient Method), 机器学习例子: 线性回归
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)
# 添加偏置项 x0 = 1 到 X 中
X_b = np.c_[np.ones((1000, 1)), X]


# 共轭梯度法实现
def ConjugateGradient(X, y, theta_initial, iterations=100):
    m = len(y)
    theta = theta_initial; theta_path = []; theta_path.append(theta)
    history = {'cost': []}

    gradient = 2/m * X.T @ (X @ theta - y)
    direction = -gradient

    for _ in range(iterations):
        # step_size = 0.01  # 步长
        lst = np.linspace(0, 1, 101)
        idx = np.array([np.mean((X @ (theta + lr * direction) - y) ** 2) for lr in lst]).argmin()
        step_size = lst[idx] # 通过线搜索确定步长
        theta = theta + step_size * direction; theta_path.append(theta)
        next_gradient = 2/m * X.T @ (X @ theta - y)
        beta = ((next_gradient.T @ next_gradient) / (gradient.T @ gradient) )[0,0]
        direction = - next_gradient + beta * direction
        gradient = next_gradient

        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history

theta_initial = np.random.randn(X_b.shape[1], 1)
n_iterations = 100

# 运行梯度下降算法
theta_path, history = ConjugateGradient(X_b, y, theta_initial,  iterations = n_iterations, )

# 将路径转换为数组
theta_path = np.array(theta_path)

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(0, n_iterations, 10):  # 每10次迭代绘制一次直线
    axs[0].plot(X, X_b@theta_path[i], color='green', alpha=0.5)
axs[0].plot(X, X_b.dot(theta_path[-1]), color='red', linewidth=4, label='Final Model')
axs[0].set_title('Linear Regression with ConjugateGradient')
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

#%%>>>>>>>>>>>>>> 10. ConjugateGradient, 机器学习例子: 多项式回归
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + 2 * X**2 + np.random.randn(1000, 1)
# 增加多项式特征
X_b = np.c_[np.ones((1000, 1)), X, X**2]

# 共轭梯度法实现
def ConjugateGradient(X, y, theta_initial, iterations=100):
    m = len(y)
    theta = theta_initial; theta_path = []; theta_path.append(theta)
    history = {'cost': []}

    gradient = 2/m * X.T @ (X @ theta - y)
    direction = -gradient

    for _ in range(iterations):
        # step_size = 0.01  # 步长
        lst = np.linspace(0, 1, 101)
        idx = np.array([np.mean((X @ (theta + ss * direction) - y) ** 2) for ss in lst]).argmin()
        step_size = lst[idx] # 通过线搜索确定步长
        theta = theta + step_size * direction; theta_path.append(theta)
        next_gradient = 2/m * X.T @ (X @ theta - y)
        beta = ((next_gradient.T @ next_gradient) / (gradient.T @ gradient) )[0,0]
        direction = - next_gradient + beta * direction
        gradient = next_gradient

        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history

theta_initial = np.random.randn(X_b.shape[1], 1)
n_iterations = 100

# 运行梯度下降算法
theta_path, history = ConjugateGradient(X_b, y, theta_initial,  iterations = n_iterations, )

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
axs[0].set_title('Polyomial Regression with ConjugateGradient')
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

#%%>>>>>>>>>>>>>> 10. ConjugateGradient: 解析
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


np.random.seed(42)

## 定义要最小化的函数（简单的二次函数）
def f(w):
    return w[0] ** 2/4 + w[1] ** 2
## 定义函数关于 x 和 y 的偏导数
def grad_f(w):
    return np.array([w[0]/2.0, 2*w[1]])

# # 定义目标函数及其梯度
# def f(w):
#     return w[0]**2 + w[1]**2 + 10*np.sin(w[0]) + 10*np.sin(w[1])
# def grad_f(w):
#     return np.array([2*w[0] + 10*np.cos(w[0]), 2*w[1] + 10*np.cos(w[1])])

# # 定义要最小化的函数
# def f(w):
#     return (1-w[1]**5 + w[0]**5)*np.exp(-w[0]**2 - w[1]**2)
# ## 定义函数关于 x 和 y 的偏导数
# def grad_f(w):
#     gx = (5 * w[0]**4 * np.exp(-w[0]**2-w[1]**2) - 2*w[0] * (1 - w[1]**5 + w[0]**5) * np.exp(-w[0]**2-w[1]**2))
#     gy = (-5 * w[1]**4 * np.exp(-w[0]**2-w[1]**2) - 2*w[1] * (1-w[1]**5 + w[0]**5) * np.exp(-w[0]**2-w[1]**2))
#     return np.array([gx, gy])

def ConjugateGradient_optimizer(theta_init,  num_iters = 1000,  perturbation = 0.1):
    ## 初始化
    theta = theta_init
    history = []
    history.append([theta[0], theta[1], f(theta)])

    gradient = grad_f(theta)
    direction = -gradient

    for _ in range(num_iters):
        # step_size = 0.01  # 步长
        lst = np.linspace(0, 1, 101)
        idx = np.array([f(theta + ss * direction) for ss in lst]).argmin()
        step_size = lst[idx] # 通过线搜索确定步长
        theta = theta + step_size * direction
        history.append([theta[0], theta[1], f(theta)]) # 保存参数的历史记录
        next_gradient = grad_f(theta)
        beta = np.dot(next_gradient, next_gradient) / np.dot(gradient, gradient)
        direction = -next_gradient + beta * direction
        gradient = next_gradient

    return theta[0], theta[1], f(theta), np.array(history)

# 定义用于绘制函数的网格
x_range = np.arange(-10, 10 , 0.1 )
y_range = np.arange(-10, 10 , 0.1 )
X, Y = np.meshgrid(x_range, y_range)
W_array = np.vstack([X.ravel(), Y.ravel()])
Z = f(W_array).reshape(X.shape)

theta_init = np.array([8, 8])
n_iterations = 100
pb = 0.1

# 执行优化算法
x_opt, y_opt, f_opt, history = ConjugateGradient_optimizer(theta_init,  num_iters = n_iterations, perturbation = pb )

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
# CS = ax.contourf(X, Y, Z, levels = 30, cmap = 'RdYlBu_r',  )
# fig.colorbar(CS)
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



































