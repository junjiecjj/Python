#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 21:35:26 2024

@author: jack
"""


import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
import os

# 生成虚拟数据集
m = 2000
np.random.seed(42)
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
# 添加偏置项 (X0 = 1)
X_b = np.c_[np.ones((len(X), 1)), X]


def f(theta, X_b, y ):
    return np.mean((X_b @ theta  - y) ** 2, axis = 0)

def grad_f(theta, X_b, y ):
    m = len(y)
    return 2/m * X_b.T @ (X_b @ theta  - y)

def grad_f_bs(theta, X_b, y, bs = 100):
    m = len(y)
    idx = np.random.choice(m, bs, replace=False)
    xx = X_b[idx]
    yy = y[idx]
    return 2/bs * xx.T @ (xx @ theta  - yy)


## 定义梯度下降算法
def Gradient_descent(theta_init, lr, num_iterations, perturbation = 0.1):
    # 初始化参数
    theta = theta_init
    history = []
    history.append([theta[0][0], theta[1][0], f(theta, X_b, y )[0]])
    # 执行梯度下降迭代
    for i in  range (num_iterations):
        # 计算梯度
        grad = grad_f(theta, X_b, y) + perturbation * np.random.randn(*theta.shape)
        # 更新参数
        theta = theta - lr * grad
        # print(f"{i}:{theta}")
        # 保存参数的历史记录
        history.append([theta[0][0], theta[1][0], f(theta, X_b, y)[0]])

    return theta[0][0], theta[1][0], f(theta, X_b, y)[0], np.array(history)


# 定义用于绘制函数的网格
theta1_range = np.arange(-40, 60 , 0.2)
theta2_range = np.arange(-40, 60 , 0.2)
Theta1, Theta2 = np.meshgrid(theta1_range, theta2_range)
Theta_array = np.vstack([Theta1.ravel(), Theta2.ravel()])
Z = f(Theta_array, X_b, y).reshape(Theta1.shape)

# 执行梯度下降并绘制结果
theta_init = np.array([39, 59]).reshape(-1,1)
learning_rate = 0.1
num_iterations = 100
pb = 0

x_opt, y_opt, f_opt, history = Gradient_descent(theta_init, learning_rate, num_iterations, perturbation = pb)

bs = 100
## 定义梯度下降算法
def Gradient_descent_device_select(theta_init, lr, num_iterations, bs = 2, perturbation = 0.1):
    # 初始化参数
    theta = theta_init
    history = []
    history.append([theta[0][0], theta[1][0], f(theta, X_b, y )[0]])
    # 执行梯度下降迭代
    for i in  range (num_iterations):
        # 计算梯度
        grad = grad_f_bs(theta, X_b, y, bs = bs) + perturbation * np.random.randn(*theta.shape)
        # 更新参数
        theta = theta - lr * grad
        # print(f"{i}:{theta}")
        # 保存参数的历史记录
        history.append([theta[0][0], theta[1][0], f(theta, X_b, y)[0]])

    return theta[0][0], theta[1][0], f(theta, X_b, y)[0], np.array(history)
x_opt_ds, y_opt_ds, f_opt_ds, history_ds = Gradient_descent_device_select(theta_init, learning_rate, num_iterations, bs = bs, perturbation = pb)

###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

fig, axs = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8, 8), ) # constrained_layout=True
axs.set_proj_type('ortho')

norm_plt = plt.Normalize(Z.min(), Z.max())
colors = cm.RdYlBu_r(norm_plt(Z))
axs.plot_wireframe(Theta1, Theta2, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5) # color = '#0070C0',

axs.plot(history[:,0], history[:,1], history[:,2], c = 'b' , marker= 'o', ms = 5, zorder = 2, label = "All data")
axs.scatter(history[-1,0], history[-1,1], history[-1,2], c = 'r' , marker= '*', s = 40, zorder = 10 )

axs.plot(history_ds[:,0], history_ds[:,1], history_ds[:,2], c = 'g' , marker= 'o', ms = 5, zorder = 2, label = "Mini-Batch")
axs.scatter(history_ds[-1,0], history_ds[-1,1], history_ds[-1,2], c = 'r' , marker= 'D', s = 40, zorder = 10 )

font = {'weight' : 'normal', 'size': 20,}
axs.legend(loc='best', prop =  font )
# axs.legend(title="Legend", prop =  font )

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 2)
axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 2)
axs.set_zlabel(r'$\mathrm{f}(\mathrm{\theta_1},\mathrm{\theta_2}$)', fontdict = font, labelpad = 6 )
axs.set_title(f"Gradient descent on data, pb = {pb}, bs = {bs}", fontdict = font, )

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
savedir = f'/home/jack/公共的/Figure/optimfigs/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'UserSelection_3D_{bs}_pb{pb}.eps',bbox_inches='tight', pad_inches=0, )
plt.show()

###### 2D contour
fig, axs = plt.subplots(figsize=(7, 6), constrained_layout=True)
CS = axs.contour(Theta1, Theta2, Z, levels = 30, cmap = 'RdYlBu_r', linewidths = 1)
fig.colorbar(CS)

axs.plot(history[:,0], history[:,1],  c = 'b' , marker= 'o', ms = 5, label = "All data")
axs.scatter(history[-1,0], history[-1,1],  c = 'r' , marker= '*', s = 200, zorder = 30)

axs.plot(history_ds[:,0], history_ds[:,1],  c = 'g' , marker= 'o', ms = 5, label = "Mini-Batch")
axs.scatter(history_ds[-1,0], history_ds[-1,1],  c = 'purple' , marker= 'D', s = 200, zorder = 20)

font = {'weight' : 'normal', 'size': 20,}
axs.legend(loc='best', prop =  font )

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 0)
axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 0)
axs.set_title(f"Gradient descent on data, pb = {pb}, bs = {bs}", fontdict = font,  )

axs.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)

axs.set_xlim(Theta1.min(), Theta1.max())
axs.set_ylim(Theta2.min(), Theta2.max())
axs.grid(False)


# 显示图形
out_fig = plt.gcf()

savedir = f'/home/jack/公共的/Figure/optimfigs/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'UserSelection_2D_{bs}_pb{pb}.eps',bbox_inches='tight', pad_inches=0, )
plt.show()



