#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:58:01 2024

@author: jack
"""

########## 5. RMSProp (Root Mean Square Propagation), 机器学习例子: 线性回归
import numpy as np
import matplotlib.pyplot as plt
import os

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)
# 添加偏置项 (X0 = 1)
X_b = np.c_[np.ones((len(X), 1)), X]


def RMSProp(X, y, theta_init, lr, gamma, epsilon=1e-8, iterations=100):
    m = len(y)
    # 梯度下降优化
    theta = theta_init; theta_path = []; theta_path.append(theta)
    history = {'cost': []}
    cost = np.mean((X @ theta  - y) ** 2)
    history['cost'].append(cost)

    E_g2 = np.zeros_like(theta)
    for i in range(iterations):
        gradient = 2/m * X.T @ (X @ theta - y)
        E_g2 = gamma * E_g2 + (1 - gamma) * gradient ** 2
        theta = theta - lr / (np.sqrt(E_g2 + epsilon)) * gradient
        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history

# 初始化参数 theta
theta_initial = np.random.randn(X_b.shape[1], 1)
lr = 0.1
n_iterations = 100
gamma = 0.9

# 运行梯度下降算法
theta_path, history = RMSProp(X_b, y, theta_initial, lr, gamma, epsilon=1e-8, iterations=n_iterations)

# 将路径转换为数组
theta_path = np.array(theta_path)

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(0, n_iterations, 10):  # 每10次迭代绘制一次直线
    axs[0].plot(X, X_b.dot(theta_path[i]), color='green', linewidth=2,)
axs[0].plot(X, X_b.dot(theta_path[-1]), color='r', linewidth=4, label='Final Model')

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,}
axs[0].set_title('Linear Regression with RMSprop', fontdict = font )
axs[0].set_xlabel('x', fontdict = font, labelpad = 2)
axs[0].set_ylabel('y', fontdict = font, labelpad = 2)

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,}
axs[0].legend(loc='best', prop =  font )

axs[0].tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)

# 绘制损失函数的变化
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,}
axs[1].plot(range(len(history['cost'])), history['cost'], color='red', linewidth=2, label='Loss Function')
axs[1].set_title('Loss Function Over Iterations', fontdict = font )
axs[1].set_xlabel('Iteration', fontdict = font, labelpad = 2)
axs[1].set_ylabel('Mean Squared Error', fontdict = font, labelpad = 2)
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,}
axs[1].legend(loc='best', prop =  font )
axs[1].tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)

# 显示图形
out_fig = plt.gcf()
optimizer = "RMSprop"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_LineRe.eps', bbox_inches='tight', pad_inches=0,)
plt.show()

#%%>>>>>>>>>>>>>> 5. RMSProp (Root Mean Square Propagation), 机器学习例子: 多项式拟合

import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + 2 * X**2 + np.random.randn(1000, 1)
# 增加多项式特征
X_b = np.c_[np.ones((1000, 1)), X, X**2]

def RMSProp(X, y, theta_init, lr, gamma, epsilon=1e-8, iterations=100):
    m = len(y)
    # 梯度下降优化
    theta = theta_init; theta_path = []; theta_path.append(theta)
    history = {'cost': []}
    cost = np.mean((X @ theta  - y) ** 2)
    history['cost'].append(cost)

    E_g2 = np.zeros_like(theta)
    for i in range(iterations):
        gradient = 2/m * X.T @ (X @ theta - y)
        E_g2 = gamma * E_g2 + (1 - gamma) * gradient ** 2
        theta = theta - lr / (np.sqrt(E_g2 + epsilon)) * gradient
        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history

# 初始化参数 theta
theta_initial = np.random.randn(X_b.shape[1], 1)
lr = 0.1
n_iterations = 100
gamma = 0.9

# 运行梯度下降算法
theta_path, history = RMSProp(X_b, y, theta_initial, lr, gamma, epsilon=1e-8, iterations=n_iterations)

# 将路径转换为数组
theta_path = np.array(theta_path)
# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

# 生成用于画图的曲线
X_new = np.linspace(0, 2, 100).reshape(100, 1)
X_new_poly = np.c_[np.ones((100, 1)), X_new, X_new**2]
y_predict = X_new_poly@theta_path[-1]

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(0, n_iterations, 10):  # 每10次迭代绘制一次直线
    axs[0].plot(X_new, X_new_poly@theta_path[i], color='g', linewidth=2,)
axs[0].plot(X_new, y_predict, color='r', linewidth=4, label='Final Model')
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,}
axs[0].set_title('Polyomial Regression with RMSprop', fontdict = font )
axs[0].set_xlabel('x', fontdict = font, labelpad = 2)
axs[0].set_ylabel('y', fontdict = font, labelpad = 2)

font = {'family':'Times New Roman','weight' : 'normal', 'size': 20,}
axs[0].legend(loc='best', prop =  font )

axs[0].tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)

# 绘制损失函数的变化
axs[1].plot(range(len(history['cost'])), history['cost'], color='red', linewidth=2, label='Loss Function')
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,}
axs[1].set_title('Loss Function Over Iterations', fontdict = font )
axs[1].set_xlabel('Iteration', fontdict = font, labelpad = 2)
axs[1].set_ylabel('Mean Squared Error', fontdict = font, labelpad = 2)
font = {'family':'Times New Roman','weight' : 'normal', 'size': 20,}
axs[1].legend(loc='best', prop =  font )
axs[1].tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)

# 显示图形
out_fig = plt.gcf()
optimizer = "RMSprop"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_Polyomial.eps', bbox_inches='tight', pad_inches=0,)
plt.show()

# #%%>>>>>>>>>>>>>> 5. RMSProp (Root Mean Square Propagation): 解析
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from sympy import lambdify, diff, exp, latex, sin, cos
# from sympy.abc import x, y
# # 设置随机种子
# np.random.seed(42)

# # 创建一个简单的损失函数，假设它有两个参数 w1 和 w2
# def f(w):
#     return (w[0] - 3)**2 + (w[1] + 4)**2 + np.sin(w[0]*5) + np.cos(w[1]*5)

# # 计算损失函数的梯度
# def grad_f(w):
#     dw1 = 2 * (w[0] - 3) + 5 * np.cos(w[0]*5)
#     dw2 = 2 * (w[1] + 4) - 5 * np.sin(w[1]*5)
#     return np.array([dw1, dw2])

# # RMSprop优化算法的实现
# def RMSProp_optimizer(theta_init, lr, gamma, epsilon, num_iterations, perturbation = 0.1):
#     theta = theta_init
#     E_g2 = np.zeros_like(theta)    # 初始化s为零向量
#     history = []

#     for i in range(num_iterations):
#         grad = grad_f(theta) +  perturbation * np.random.randn(*theta.shape)
#         E_g2 = gamma * E_g2 + (1 - gamma) * grad**2
#         theta = theta - lr * grad / (np.sqrt(E_g2) + epsilon)
#         # 保存参数的历史记录
#         history.append([theta[0], theta[1], f(theta)])

#     return theta[0], theta[1], f(theta), np.array(history)

# x_range = np.arange(-10 ,10 ,0.1 )
# y_range = np.arange(-10 ,10 ,0.1 )
# X, Y = np.meshgrid(x_range, y_range)
# W_array = np.vstack([X.ravel(), Y.ravel()])
# Z = f(W_array).reshape(X.shape)

# # 参数设置
# learning_rate = 0.2
# gamma = 0.9
# epsilon = 1e-8
# num_iterations = 600
# pb = 0
# theta_init = np.array([8, 8])

# # 执行优化算法
# x_opt, y_opt, f_opt, history = RMSProp_optimizer(theta_init, learning_rate, gamma, epsilon, num_iterations, perturbation = pb)

# ## 3D
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8), constrained_layout=True)
# ax.set_proj_type('ortho')

# norm_plt = plt.Normalize(Z.min(), Z.max())
# colors = cm.RdYlBu_r(norm_plt(Z))
# ax.plot_wireframe(X, Y, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5) # color = '#0070C0',

# ax.plot(history[:,0], history[:,1], history[:,2], c = 'b' , marker= 'o', ms = 5 )
# ax.scatter(history[-1,0], history[-1,1], history[-1,2], c = 'r' , marker= '*', s = 40, zorder = 10 )
# ax.set_proj_type('ortho')

# font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
# ax.set_xlabel(r'$\it{x_1}$', fontdict = font, labelpad = 2)
# ax.set_ylabel(r'$\it{x_2}$', fontdict = font, labelpad = 2)
# ax.set_zlabel(r'$\it{f}(\it{x_1},\it{x_2}$)', fontdict = font,  )

# ax.tick_params(axis='both', direction='in', width=3, length = 5,  labelsize=15, labelfontfamily = 'Times New Roman', pad = 1)

# # ax.view_init(azim=-120, elev=30)
# ax.grid(False)
# # fig.savefig('Figures/只保留网格线.svg', format='svg')
# plt.show()

# ## 2D courter
# fig, ax = plt.subplots( figsize = (7.5, 6), constrained_layout=True)
# # CS = ax.contourf(X, Y, Z, levels = 20, cmap = 'RdYlBu_r', linewidths = 1)
# # fig.colorbar(CS)
# CS = ax.contour(X, Y, Z, levels = 20, cmap = 'RdYlBu_r', linewidths = 1)
# fig.colorbar(CS)

# ax.plot(history[:,0], history[:,1],  c = 'b' , marker= 'o', ms = 5 )
# ax.scatter(history[-1,0], history[-1,1],  c = 'r' , marker= '*', s = 200, zorder = 10)
# font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
# ax.set_xlabel(r'$\it{x_1}$', fontdict = font, labelpad = 2)
# ax.set_ylabel(r'$\it{x_2}$', fontdict = font, labelpad = 2)

# ax.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=15, labelfontfamily = 'Times New Roman', pad = 1)

# ax.set_xlim(X.min(), X.max())
# ax.set_ylim(Y.min(), Y.max())

# ax.grid(False)
# # fig.savefig('Figures/只保留网格线.svg', format='svg')
# plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>>>> 5. RMSProp (Root Mean Square Propagation): 解析
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# 生成虚拟数据集
np.random.seed(42)

mod = "easy"

if mod == 'easy':
    funkind = 'convex fun'

    def f(w):
        return w[0] ** 2/4 + w[1] ** 2
    ## 定义函数关于 x 和 y 的偏导数
    def grad_f(w):
        return np.array([w[0]/2.0, 2*w[1]])
elif mod == 'hard':
    funkind = 'non-convex fun'

    def f(w):
        return (1-w[1]**5 + w[0]**5)*np.exp(-w[0]**2 - w[1]**2)
    ## 定义函数关于 x 和 y 的偏导数
    def grad_f(w):
        gx = (5 * w[0]**4 * np.exp(-w[0]**2-w[1]**2) - 2*w[0] * (1 - w[1]**5 + w[0]**5) * np.exp(-w[0]**2-w[1]**2))
        gy = (-5 * w[1]**4 * np.exp(-w[0]**2-w[1]**2) - 2*w[1] * (1-w[1]**5 + w[0]**5) * np.exp(-w[0]**2-w[1]**2))
        return np.array([gx, gy])

# RMSprop优化算法的实现
def RMSProp_optimizer(theta_init, lr, gamma, epsilon, num_iterations, perturbation = 0.1):
    theta = theta_init
    history = []
    history.append([theta[0], theta[1], f(theta)])

    E_g2 = np.zeros_like(theta)    # 初始化s为零向量
    for i in range(num_iterations):
        grad = grad_f(theta) +  perturbation * np.random.randn(*theta.shape)
        E_g2 = gamma * E_g2 + (1 - gamma) * grad**2
        theta = theta - lr * grad / (np.sqrt(E_g2) + epsilon)
        # 保存参数的历史记录
        history.append([theta[0], theta[1], f(theta)])

    return theta[0], theta[1], f(theta), np.array(history)

if mod == 'easy':
    # 定义用于绘制函数的网格
    x_range = np.arange(-10 , 10 , 0.1 )
    y_range = np.arange(-10 , 10 , 0.1 )
    X, Y = np.meshgrid(x_range, y_range)
    W_array = np.vstack([X.ravel(), Y.ravel()])
    Z = f(W_array).reshape(X.shape)
    # 执行梯度下降并绘制结果
    theta_init = np.array([8, 8])
elif mod == 'hard':
    # 定义用于绘制函数的网格
    x_range = np.arange(-6 , 6 , 0.1 )
    y_range = np.arange(-6 , 6 , 0.1 )
    X, Y = np.meshgrid(x_range, y_range)
    W_array = np.vstack([X.ravel(), Y.ravel()])
    Z = f(W_array).reshape(X.shape)
    # 执行梯度下降并绘制结果
    theta_init = np.array([0.01, 0.01])

# 参数设置
learning_rate = 0.2
gamma = 0.9
epsilon = 1e-8
num_iterations = 300

step = int(num_iterations/20)
pb = 10

# 执行梯度下降并绘制结果
x_opt, y_opt, f_opt, history = RMSProp_optimizer(theta_init, learning_rate, gamma, epsilon, num_iterations, perturbation = pb)

## 3D
fig, axs = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8), constrained_layout=True)
axs.set_proj_type('ortho')

norm_plt = plt.Normalize(Z.min(), Z.max())
colors = cm.RdYlBu_r(norm_plt(Z))
axs.plot_wireframe(X, Y, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5) # color = '#0070C0',

axs.plot(history[:,0], history[:,1], history[:,2], c = 'b' , marker= 'o', ms = 5, zorder = 2)
axs.scatter(history[-1,0], history[-1,1], history[-1,2], c = 'r' , marker= '*', s = 40, zorder = 10 )
axs.set_proj_type('ortho')

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 2)
axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 2)
axs.set_zlabel(r'$\mathrm{f}(\mathrm{\theta_1},\mathrm{\theta_2}$)', fontdict = font, labelpad = 6 )
axs.set_title(f"RMSprop on {funkind}, pb = {pb}", fontdict = font, )

axs.tick_params(axis='both', direction='in', width = 3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)
axs.ticklabel_format(style='sci', scilimits=(-1,2), axis='z')
# ax1.view_init(azim=-100, elev=20)

# 设置坐标轴线宽
axs.xaxis.line.set_lw(1)
axs.yaxis.line.set_lw(1)
axs.zaxis.line.set_lw(1)
axs.grid(False)

# 显示图形
out_fig = plt.gcf()
optimizer = "RMSprop"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_pb{pb}_3D_{mod}fun.eps',bbox_inches='tight', pad_inches=0, )
plt.show()

## 2D courter
fig, axs = plt.subplots( figsize = (7.5, 6), constrained_layout=True)
# CS = ax.contourf(X, Y, Z, levels = 20, cmap = 'RdYlBu_r', linewidths = 1)
# fig.colorbar(CS)
CS = axs.contour(X, Y, Z, levels = 30, cmap = 'RdYlBu_r', linewidths = 1)
fig.colorbar(CS)

axs.plot(history[:,0], history[:,1],  c = 'b' , marker= 'o', ms = 5 )
axs.scatter(history[-1,0], history[-1,1],  c = 'r' , marker= '*', s = 200, zorder = 10)
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 0)
axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 0)
axs.set_title(f"RMSprop on {funkind}, pb = {pb}", fontdict = font,  )

axs.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)

axs.set_xlim(X.min(), X.max())
axs.set_ylim(Y.min(), Y.max())
axs.grid(False)


# 显示图形
out_fig = plt.gcf()
optimizer = "RMSprop"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_pb{pb}_2D_{mod}fun.eps', bbox_inches='tight', pad_inches=0,)
plt.show()


#%%>>>>>>>>>>>>>> 5. RMSProp (Root Mean Square Propagation): 线性回归, 解析

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
# matplotlib.get_backend()
# matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
# matplotlib.use('Agg')

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)
# 添加偏置项 (X0 = 1)
X_b = np.c_[np.ones((len(X), 1)), X]

## 定义要最小化的函数（简单的二次函数）
def f(theta, X_b, y ):
    return np.mean((X_b @ theta  - y) ** 2, axis = 0)
## 定义函数关于 x 和 y 的偏导数
def grad_f(theta, X_b, y ):
    m = len(y)
    return 2/m * X_b.T @ (X_b @ theta  - y)

# RMSprop优化算法的实现
def RMSProp_optimizer(theta_init, lr, gamma, epsilon, num_iterations, perturbation = 0.1):
    # 初始化参数
    theta = theta_init
    history = []
    history.append([theta[0][0], theta[1][0], f(theta, X_b, y )[0]])

    E_g2 = np.zeros_like(theta)    # 初始化s为零向量
    for i in range(num_iterations):
        grad = grad_f(theta, X_b, y) +  perturbation * np.random.randn(*theta.shape)
        E_g2 = gamma * E_g2 + (1 - gamma) * grad**2
        theta = theta - lr * grad / (np.sqrt(E_g2) + epsilon)
        # 保存参数的历史记录
        history.append([theta[0][0], theta[1][0], f(theta, X_b, y)[0]])

    return theta[0][0], theta[1][0], f(theta, X_b, y)[0], np.array(history)

# 定义用于绘制函数的网格
theta1_range = np.arange(-40, 60 , 0.2)
theta2_range = np.arange(-40, 60 , 0.2)
Theta1, Theta2 = np.meshgrid(theta1_range, theta2_range)
Theta_array = np.vstack([Theta1.ravel(), Theta2.ravel()])
Z = f(Theta_array, X_b, y).reshape(Theta1.shape)

# Adam 优化算法参数
theta_init = np.array([39, 59]).reshape(-1,1)
learning_rate = 0.2
gamma = 0.9
epsilon = 1e-8
num_iterations = 300
pb = 0

# 执行优化算法
x_opt, y_opt, f_opt, history = RMSProp_optimizer(theta_init, learning_rate, gamma, epsilon, num_iterations, perturbation = pb)
###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

fig, axs = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8, 8), ) # constrained_layout=True
###### 3D wireframe
# ax1 = fig.add_subplot(111, projection = '3d')
axs.set_proj_type('ortho')

norm_plt = plt.Normalize(Z.min(), Z.max())
colors = cm.RdYlBu_r(norm_plt(Z))
axs.plot_wireframe(Theta1, Theta2, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5) # color = '#0070C0',

axs.plot(history[:,0], history[:,1], history[:,2], c = 'b' , marker= 'o', ms = 5, zorder = 2)
axs.scatter(history[-1,0], history[-1,1], history[-1,2], c = 'r' , marker= '*', s = 40, zorder = 10 )
axs.set_proj_type('ortho')

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 2)
axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 2)
axs.set_zlabel(r'$\mathrm{f}(\mathrm{\theta_1},\mathrm{\theta_2}$)', fontdict = font, labelpad = 6 )
axs.set_title(f"RMSprop on data, pb = {pb}", fontdict = font, )

axs.tick_params(axis='both', direction='in', width = 3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)
axs.ticklabel_format(style='sci', scilimits=(-1,2), axis='z')
# ax1.view_init(azim=-100, elev=20)

# 设置坐标轴线宽
axs.xaxis.line.set_lw(1)
axs.yaxis.line.set_lw(1)
axs.zaxis.line.set_lw(1)
axs.grid(False)


# 显示图形
out_fig = plt.gcf()
optimizer = "RMSprop"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_pb{pb}_3D_Xy.eps', )
plt.show()

###### 2D contour
fig, axs = plt.subplots(figsize=(7, 6), constrained_layout=True)
CS = axs.contour(Theta1, Theta2, Z, levels = 30, cmap = 'RdYlBu_r', linewidths = 1)
fig.colorbar(CS)

axs.plot(history[:,0], history[:,1],  c = 'b' , marker= 'o', ms = 5 )
axs.scatter(history[-1,0], history[-1,1],  c = 'r' , marker= '*', s = 200, zorder = 10)
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 0)
axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 0)
axs.set_title(f"RMSprop on data, pb = {pb}", fontdict = font,  )

axs.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)

axs.set_xlim(Theta1.min(), Theta1.max())
axs.set_ylim(Theta2.min(), Theta2.max())
axs.grid(False)

# 显示图形
out_fig = plt.gcf()
optimizer = "RMSprop"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_pb{pb}_2D_Xy.eps', bbox_inches='tight', pad_inches=0,)
plt.show()























