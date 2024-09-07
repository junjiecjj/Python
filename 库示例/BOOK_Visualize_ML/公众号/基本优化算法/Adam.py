#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:00:20 2024

@author: jack
"""

#%%>>>>>>>>>>>>>> 6. Adam (Adaptive Moment Estimation), 机器学习例子: 线性回归
import numpy as np
import matplotlib.pyplot as plt
import os


# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)
# 添加偏置项 x0 = 1 到 X 中
X_b = np.c_[np.ones((1000, 1)), X]

def Adam(X, y, theta_init, alpha, beta1, beta2, eps=1e-8, iterations=100):
    m = len(y)
    theta = theta_init; theta_path = []; theta_path.append(theta)

    history = {'cost': []}
    cost = np.mean((X @ theta  - y) ** 2)
    history['cost'].append(cost)

    m_t = np.zeros_like(theta)
    v_t = np.zeros_like(theta)
    for t in range(1, iterations + 1):
        gradient = 2/m * X.T @ (X @ theta - y)
        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * gradient ** 2
        m_t_hat = m_t / (1 - beta1 ** t)
        v_t_hat = v_t / (1 - beta2 ** t)
        theta = theta - alpha * m_t_hat / (np.sqrt(v_t_hat) + eps)

        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history

theta_initial = np.random.randn(X_b.shape[1], 1)
lr = 0.1
n_iterations = 100
epsilon = 1e-8
alpha = 0.1
beta1 = 0.2
beta2 = 0.2
# 运行梯度下降算法
theta_path, history = Adam(X_b, y, theta_initial, alpha, beta1, beta2, eps=epsilon, iterations=n_iterations)

# 将路径转换为数组
theta_path = np.array(theta_path)

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(0, n_iterations, 10):  # 每10次迭代绘制一次直线
    axs[0].plot(X, X_b.dot(theta_path[i]), color='green', linewidth=2,)
axs[0].plot(X, X_b.dot(theta_path[-1]), color='r', linewidth=4, label='Final Model')

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs[0].set_title('Linear Regression with ADAM', fontdict = font )
axs[0].set_xlabel('x', fontdict = font, labelpad = 2)
axs[0].set_ylabel('y', fontdict = font, labelpad = 2)

font = {'weight' : 'normal', 'size': 20,}
axs[0].legend(loc='best', prop =  font )

axs[0].tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)

# 绘制损失函数的变化
axs[1].plot(range(len(history['cost'])), history['cost'], color='red', linewidth=2, label='Loss Function')
axs[1].set_title('Loss Function Over Iterations', fontdict = font )
axs[1].set_xlabel('Iteration', fontdict = font, labelpad = 2)
axs[1].set_ylabel('Mean Squared Error', fontdict = font, labelpad = 2)
font = {'weight' : 'normal', 'size': 20,}
axs[1].legend(loc='best', prop =  font )
axs[1].tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)

# 显示图形
out_fig = plt.gcf()
optimizer = "ADAM"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_LineRe.eps', bbox_inches='tight', pad_inches=0,)
plt.show()

#%%>>>>>>>>>>>>>> 6. Adam (Adaptive Moment Estimation), 机器学习例子: 多项式回归
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + 2 * X**2 + np.random.randn(1000, 1)
# 增加多项式特征
X_b = np.c_[np.ones((1000, 1)), X, X**2]

def Adam(X, y, theta_init, alpha, beta1, beta2, eps=1e-8, iterations=100):
    m = len(y)
    theta = theta_init; theta_path = []; theta_path.append(theta)
    history = {'cost': []}
    cost = np.mean((X @ theta  - y) ** 2)
    history['cost'].append(cost)

    m_t = np.zeros_like(theta)
    v_t = np.zeros_like(theta)
    for t in range(1, iterations + 1):
        gradient = 2/m * X.T @ (X @ theta - y)
        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * gradient ** 2
        m_t_hat = m_t / (1 - beta1 ** t)
        v_t_hat = v_t / (1 - beta2 ** t)
        theta = theta - alpha * m_t_hat / (np.sqrt(v_t_hat) + eps)

        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history

theta_initial = np.random.randn(X_b.shape[1], 1)
lr = 0.1
n_iterations = 100
epsilon = 1e-8
beta1 = 0.2
beta2 = 0.2
# 运行梯度下降算法
theta_path, history = Adam(X_b, y, theta_initial, lr, beta1, beta2, eps=epsilon, iterations=n_iterations)

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
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs[0].set_title('Polyomial Regression with ADAM', fontdict = font )
axs[0].set_xlabel('x', fontdict = font, labelpad = 2)
axs[0].set_ylabel('y', fontdict = font, labelpad = 2)

font = {'weight' : 'normal', 'size': 20,}
axs[0].legend(loc='best', prop =  font )

axs[0].tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)

# 绘制损失函数的变化
axs[1].plot(range(len(history['cost'])), history['cost'], color='red', linewidth=2, label='Loss Function')
axs[1].set_title('Loss Function Over Iterations', fontdict = font )
axs[1].set_xlabel('Iteration', fontdict = font, labelpad = 2)
axs[1].set_ylabel('Mean Squared Error', fontdict = font, labelpad = 2)
font = {'weight' : 'normal', 'size': 20,}
axs[1].legend(loc='best', prop =  font )
axs[1].tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)

# 显示图形
out_fig = plt.gcf()
optimizer = "ADAM"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_Polyomial.eps',bbox_inches='tight', pad_inches=0, )
plt.show()


#%%>>>>>>>>>>>>>> 7.Adam (Adaptive Moment Estimation): 解析
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

def Adam_optimizer(theta_init, lr = 0.1, beta1 = 0.9, beta2 = 0.99, eps = 1e-8, num_iterations = 100, perturbation = 0.1):
    ## 初始化
    theta = theta_init
    history = []
    history.append([theta[0], theta[1], f(theta)])
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    # 执行梯度下降迭代
    for t in  range (1, num_iterations):
        gradient = grad_f(theta) + perturbation * np.random.randn(*theta.shape)
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        theta = theta - lr * m_hat / (np.sqrt(v_hat) + epsilon)
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

learning_rate = 0.1
beta1 = 0.9
beta2 = 0.99
epsilon = 1e-8
num_iterations = 1000
pb = 5

# 执行优化算法
x_opt, y_opt, f_opt, history = Adam_optimizer(theta_init, lr = learning_rate, beta1 = beta1, beta2=beta2, eps=epsilon, num_iterations=num_iterations, perturbation = pb)

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
axs.set_title(f"ADAM on {funkind}, pb = {pb}", fontdict = font, )

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
optimizer = "ADAM"
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
axs.set_title(f"ADAM on {funkind}, pb = {pb}", fontdict = font,  )

axs.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)

axs.set_xlim(X.min(), X.max())
axs.set_ylim(Y.min(), Y.max())
axs.grid(False)


# 显示图形
out_fig = plt.gcf()
optimizer = "ADAM"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_pb{pb}_2D_{mod}fun.eps', bbox_inches='tight', pad_inches=0,)
plt.show()

#%%>>>>>>>>>>>>>> 1. 梯度下降法: 线性回归, 解析

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

def Adam_optimizer(theta_init, lr = 0.1, beta1 = 0.9, beta2 = 0.99, eps = 1e-8, num_iterations = 100, perturbation = 0.1):
    # 初始化参数
    theta = theta_init
    history = []
    history.append([theta[0][0], theta[1][0], f(theta, X_b, y )[0]])

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    # 执行梯度下降迭代
    for t in  range (1, num_iterations):
        gradient = grad_f(theta, X_b, y) + perturbation * np.random.randn(*theta.shape)
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        theta = theta - lr * m_hat / (np.sqrt(v_hat) + epsilon)
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
beta1 = 0.9
beta2 = 0.99
epsilon = 1e-8
num_iterations = 1000
pb = 30

# 执行优化算法
x_opt, y_opt, f_opt, history = Adam_optimizer(theta_init, lr = learning_rate, beta1 = beta1, beta2=beta2, eps=epsilon, num_iterations=num_iterations, perturbation = pb)
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
axs.set_title(f"ADAM on data, pb = {pb}", fontdict = font, )

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
optimizer = "ADAM"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_pb{pb}_3D_Xy.eps', bbox_inches='tight', pad_inches=0,)
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
axs.set_title(f"ADAM on data, pb = {pb}", fontdict = font,  )

axs.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)

axs.set_xlim(Theta1.min(), Theta1.max())
axs.set_ylim(Theta2.min(), Theta2.max())
axs.grid(False)

# 显示图形
out_fig = plt.gcf()
optimizer = "ADAM"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_pb{pb}_2D_Xy.eps', bbox_inches='tight', pad_inches=0,)
plt.show()
















































































