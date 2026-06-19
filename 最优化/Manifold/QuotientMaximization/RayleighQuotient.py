#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 23:21:42 2026

@author: jack
"""


import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

#%% 把瑞利商看成是二元函数

# 导入包
import math
# import numpy as np
# import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex, simplify,symbols

from matplotlib import cm
# 导入色谱模块

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 8  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 12  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 12  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [12, 8] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 1     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12


#%% 单位球上看瑞利商
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

############## 0. 球坐标
# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals

# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(0, np.pi*1, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(0, np.pi*2, nphi+1)

# 单位球半径
r = 1

### 球坐标转化为三维直角坐标
# 第一种方法
# z轴坐标网格数据
# Z = np.outer(r*np.cos(theta), np.ones(nphi+1))
# # x轴坐标网格数据
# X = np.outer(r*np.sin(theta), np.cos(phi))
# # y轴坐标网格数据
# Y = np.outer(r*np.sin(theta), np.sin(phi))

# 第二种方法
pp_, tt_ = np.meshgrid(phi, theta)
# z轴坐标网格数据
Z = r*np.cos(tt_)
# x轴坐标网格数据
X = r*np.sin(tt_)*np.cos(pp_)
# y轴坐标网格数据
Y = r*np.sin(tt_)*np.sin(pp_)

################## 1. 计算瑞利商
# 每一行代表一个三维直角坐标系坐标点
# 所有坐标点都在单位球面上
Points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

# 定义矩阵Q
Q = np.array([[1, 0.5, 1],
              [0.5, 2, -0.2],
              [1, -0.2, 1]])
# 计算 xT @ Q @ x
Rayleigh_Q = np.diag(Points @ Q @ Points.T)
#
Rayleigh_Q_ = np.reshape(Rayleigh_Q, X.shape)

##################  1  小彩灯
fig = plt.figure(figsize = (12, 12), constrained_layout = True)
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color = '0.68',linewidth=0.25)
surf.set_facecolor((0, 0, 0, 0))

ax.scatter(X[::2,::2], Y[::2,::2], Z[::2,::2], c = Rayleigh_Q_[::2,::2], cmap = 'hsv', s = 15)

# x轴切面
level_idx = 0.5
xx_, zz_ = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
ax.plot_surface(xx_, xx_*0 + level_idx, zz_, color = 'b', alpha = 0.2)
ax.plot_wireframe(xx_, xx_*0 + level_idx, zz_, color = 'b', lw = 0.2)
# ax.plot_wireframe(X, Y, Z, color = [0.8, 0.8, 0.8], rstride=5, cstride=5,  linewidth = 0.25)
ax.contour(X, Y, Z, zdir='y', levels = [level_idx], linewidths = 2, linestyles = '--', colors = 'b')

# ## x轴切面
# level_idx = 0.5
# xx_, yy_ = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
# ax.plot_surface(xx_, yy_, xx_*0 + level_idx,  color = 'r', alpha = 0.1)
# ax.plot_wireframe(xx_, yy_, xx_*0 + level_idx, color = 'r', lw = 0.2)
# # ax.plot_wireframe(X, Y, Z, color = [0.8, 0.8, 0.8], rstride=5, cstride=5,  linewidth = 0.25)
# ax.contour(X, Y, Z, zdir='z', levels = [level_idx], linewidths = 2, linestyles = '--', colors = 'r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式
ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
k = 1.5
# three reference lines
ax.plot((-k, k), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k, k), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k, k), 'k')
ax.axis('off')
ax.set_xlim((-np.max(np.abs(X)), np.max(np.abs(X))))
ax.set_ylim((-np.max(np.abs(Y)), np.max(np.abs(Y))))
ax.set_zlim((-np.max(np.abs(Z)), np.max(np.abs(Z))))
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-155, elev=35)
ax.grid(False)
plt.show()


################## 2. 填充
norm_plt = plt.Normalize(Rayleigh_Q_.min(), Rayleigh_Q_.max())
colors = cm.RdYlBu_r(norm_plt(Rayleigh_Q_)) # (51, 101, 4)

fig = plt.figure(figsize = (20, 20))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,  linewidth=0.25, shade=False)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
k = 1.5
# three reference lines
ax.plot((-k, k), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k, k), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k, k), 'k')
ax.axis('off')
ax.set_xlim((-k, k))
ax.set_ylim((-k, k))
ax.set_zlim((-k, k))
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-130, elev=30)
ax.grid(False)
# fig.savefig('Figures/瑞利商，填充.svg', format='svg')
plt.show()

################## 3. 只有网格
fig = plt.figure(figsize = (20, 20))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,  linewidth=0.25, shade=False)

surf.set_facecolor((0,0,0,0))  # 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.plot((-k, k), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k, k), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k, k), 'k')
ax.axis('off')
k = 1.5
ax.set_xlim((-k, k))
ax.set_ylim((-k, k))
ax.set_zlim((-k, k))
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-130, elev=30)
ax.grid(False)
# fig.savefig('Figures/瑞利商，网格.svg', format='svg')
plt.show()


#%%###############  将瑞利商球面展开为平面

## θ、φ 网格数据,将球体展开成平面
fig, ax = plt.subplots()
c = ax.scatter(pp_, tt_, c = Rayleigh_Q_, cmap='RdYlBu_r', s = 1.8)

ax.set_ylim(tt_.min(), tt_.max())
ax.set_xlim(pp_.min(), pp_.max())
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$', rotation=0)
ax.set_xticks(np.linspace(0, 2*np.pi, 5))
ax.set_xticklabels(['0', r'$\pi/2 (90^\circ)$', r'$\pi (180^\circ)$', r'$3\pi/2 (270^\circ)$', r'$2\pi (360^\circ)$'])

ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(['0', r'$\pi/2 (90^\circ)$', r'$\pi (180^\circ)$'])
ax.invert_yaxis()
plt.axis('scaled')
plt.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# fig.savefig('Figures/角度网格散点 + 渲染.svg', format='svg')
plt.show()


## θ、φ 网格数据,将球体展开成平面
fig, ax = plt.subplots()
# 利用 Pcolormesh()方法在二维平面上绘制一个伪彩色网格。
c = ax.pcolormesh(pp_, tt_, Rayleigh_Q_, cmap='RdYlBu_r', shading='auto', vmin=Rayleigh_Q_.min(), vmax=Rayleigh_Q_.max())
fig.colorbar(c, ax=ax, shrink = 0.58)


ax.set_xlim(tt_.min(), tt_.max())
ax.set_ylim(pp_.min(), pp_.max())
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$', rotation=0)
ax.set_xticks(np.linspace(0, 2*np.pi, 5))
ax.set_xticklabels(['0', r'$\pi/2 (90^\circ)$', r'$\pi (180^\circ)$', r'$3\pi/2 (270^\circ)$', r'$2\pi (360^\circ)$'])

ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(['0', r'$\pi/2 (90^\circ)$', r'$\pi (180^\circ)$'])
ax.invert_yaxis()
plt.axis('scaled')
plt.grid()
plt.show()

## 3 填充等高线图+等高线
fig, ax = plt.subplots()
levels = np.linspace(Rayleigh_Q_.min(),Rayleigh_Q_.max(),18)

colorbar = ax.contourf(pp_, tt_, Rayleigh_Q_, levels = levels, cmap='RdYlBu_r')
ax.contour(pp_, tt_, Rayleigh_Q_, levels = levels, colors = 'w')

fig.colorbar(colorbar, ax=ax, shrink = 0.58)
ax.set_ylim(tt_.min(), tt_.max())
ax.set_xlim(pp_.min(), pp_.max())
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$', rotation=0)
ax.set_xticks(np.linspace(0, 2*np.pi, 5))
ax.set_xticklabels(['0', r'$\pi/2 (90^\circ)$', r'$\pi (180^\circ)$', r'$3\pi/2 (270^\circ)$', r'$2\pi (360^\circ)$'])

ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(['0', r'$\pi/2 (90^\circ)$', r'$\pi (180^\circ)$'])
ax.invert_yaxis()
plt.axis('scaled')
plt.grid()
plt.show()


## 4 等高线
fig, ax = plt.subplots()
colorbar = ax.contour(pp_,tt_, Rayleigh_Q_, levels = levels, cmap='RdYlBu_r')
fig.colorbar(colorbar, ax=ax, shrink = 0.58)
ax.set_ylim(tt_.min(), tt_.max())
ax.set_xlim(pp_.min(), pp_.max())
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$', rotation=0)
ax.set_xticks(np.linspace(0, 2*np.pi, 5))
ax.set_xticklabels(['0', r'$\pi/2 (90^\circ)$', r'$\pi (180^\circ)$', r'$3\pi/2 (270^\circ)$', r'$2\pi (360^\circ)$'])

ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(['0', r'$\pi/2 (90^\circ)$', r'$\pi (180^\circ)$'])
ax.invert_yaxis()
plt.axis('scaled')
plt.grid()
plt.show()



#%% 单位球面上的瑞利商等高线

## 2
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# 提取等高线
all_contours = ax.contour(pp_, tt_, Rayleigh_Q_, levels = levels, cmap='RdYlBu_r') # 四维到三维才需要这样的技巧画等高线，见 Book2_chap32,
ax.cla() # 擦去等高线这个“艺术家”
plt.show()

## 3 球面颜色映射 + 单色等高线
fig = plt.figure(figsize = (12, 12))
ax = fig.add_subplot(111, projection='3d')
norm_plt = plt.Normalize(Rayleigh_Q_.min(), Rayleigh_Q_.max())
colors = cm.RdYlBu_r(norm_plt(Rayleigh_Q_))
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = colors,  linewidth=0.25, shade=False)
# surf.set_facecolor((0,0,0,0)) ##

for level_idx, ctr_idx in zip(all_contours.levels, all_contours.allsegs):
    for i in range(0, len(ctr_idx)):
        phi_i, theta_i = ctr_idx[i][:,0],ctr_idx[i][:,1]
        # 单位球半径
        r = 1
        # 球坐标转化为三维直角坐标
        # z轴坐标网格数据
        Z_i = r*np.cos(theta_i)
        # x轴坐标网格数据
        X_i = r*np.sin(theta_i)*np.cos(phi_i)
        # y轴坐标网格数据
        Y_i = r*np.sin(theta_i)*np.sin(phi_i)
        # 绘制映射结果
        ax.plot(X_i, Y_i, Z_i, color = 'w',  linewidth = 1, zorder = 1e10)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
k = 1.5
# three reference lines
ax.plot((-k, k), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k, k), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k, k), 'k')
ax.axis('off')
ax.set_xlim((-k, k))
ax.set_ylim((-k, k))
ax.set_zlim((-k, k))
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-130, elev=30)
ax.grid(False)
# fig.savefig('瑞利商，球面颜色映射 + 单色等高线.svg', format='svg')
plt.show()

## 4 球面颜色映射 + 彩色等高线
from matplotlib.colors import Normalize
norm = Normalize(vmin = all_contours.levels.min(), vmax = all_contours.levels.max(),  clip = True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlBu_r)

fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color = [0.5, 0.5, 0.5], linewidth=0.5)
surf.set_facecolor((0,0,0,0))

for level_idx, ctr_idx in zip(all_contours.levels,  all_contours.allsegs):
    for i in range(0,len(ctr_idx)):
        phi_i,theta_i = ctr_idx[i][:,0],ctr_idx[i][:,1]
        # 单位球半径
        r = 1
        # 球坐标转化为三维直角坐标
        # z轴坐标网格数据
        Z_i = r*np.cos(theta_i)
        # x轴坐标网格数据
        X_i = r*np.sin(theta_i)*np.cos(phi_i)
        # y轴坐标网格数据
        Y_i = r*np.sin(theta_i)*np.sin(phi_i)
        # 绘制映射结果
        ax.plot(X_i, Y_i, Z_i,  color = mapper.to_rgba(level_idx),  linewidth = 3)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
k = 1.5
# three reference lines
ax.plot((-k, k), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k, k), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k, k), 'k')
ax.axis('off')
ax.set_xlim((-k, k))
ax.set_ylim((-k, k))
ax.set_zlim((-k, k))
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-130, elev=30)
ax.grid(False)
# fig.savefig('瑞利商，球面颜色映射 + 彩色等高线.svg', format='svg')
plt.show()

plt.close('all')































