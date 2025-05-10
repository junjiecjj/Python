#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:43:44 2024

@author: jack

"""


#%%>>>>>>>>>>>>>> 1. 梯度下降法: 线性回归, 解析

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
import os


mod = "hard"
if mod == 'easy':
    funkind = 'convexfun'

    def f(w):
        return w[0] ** 2/4 + w[1] ** 2

elif mod == 'hard':
    funkind = 'non-convexfun'

    def f(w):
        return (1-w[1]**5 + w[0]**5)*np.exp(-w[0]**2 - w[1]**2)

if mod == 'easy':
    # 定义用于绘制函数的网格
    x_range = np.arange(-10 ,10 , 0.1 )
    y_range = np.arange(-10 ,10 , 0.1 )
    X, Y = np.meshgrid(x_range, y_range)
    W_array = np.vstack([X.ravel(), Y.ravel()])
    Z = f(W_array).reshape(X.shape)

    # 执行梯度下降并绘制结果
    theta_init = np.array([8, 8])
elif mod == 'hard':
    # 定义用于绘制函数的网格
    x_range = np.arange(-4 , 4 , 0.1 )
    y_range = np.arange(-4 , 4 , 0.1 )
    X, Y = np.meshgrid(x_range, y_range)
    W_array = np.vstack([X.ravel(), Y.ravel()])
    Z = f(W_array).reshape(X.shape)

fig, ax = plt.subplots(figsize = (8,8), subplot_kw={'projection': '3d'},  constrained_layout = True)
ax.set_proj_type('ortho')

norm_plt = plt.Normalize(Z.min(), Z.max())
colors = cm.RdYlBu_r(norm_plt(Z))
surf = ax.plot_surface(X, Y, Z, facecolors = colors, rstride = 1, cstride = 1, linewidth = 1, shade = False) # 删除阴影
surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

# ax.plot_wireframe(X, Y, Z, color = [0.5,0.5,0.5], linewidth = 0.25)

##三维等高线
# colorbar = ax.contour(X, Y, Z, 10,  cmap = 'hsv')
# fig.colorbar(colorbar, ax = ax, shrink=0.5, aspect=20)

# 二维等高线
# colorbar = ax.contour(X, Y, Z, zdir='z', offset= Z.min(), levels = 10, linewidths = 2, cmap = "hsv")  # 生成z方向投影，投到x-y平面
# fig.colorbar(colorbar, ax=ax, shrink=0.5, aspect=20)

# 3D坐标区的背景设置为白色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.set_xlabel(r'$\it{\theta_1}$')
ax.set_ylabel(r'$\it{\theta_2}$')
ax.set_zlabel(r'$\it{f}$($\it{theta_1}$,$\it{theta_2}$)')

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)


# plt.subplots_adjust(top=1, bottom=0, right=1 , left=0. , hspace=0, wspace=0)
# 显示图形
out_fig = plt.gcf()
savedir = '/home/jack/公共的/Figure/optimfigs/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{funkind}.eps', bbox_inches='tight', pad_inches= -0.1,)
plt.show()






























































