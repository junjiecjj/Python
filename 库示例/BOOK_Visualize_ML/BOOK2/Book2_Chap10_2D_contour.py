#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 15:50:02 2024

@author: jack

 BK_2_Ch06_03.ipynb

 Chapter 10 平面等高线 | Book 2《可视之美》

"""

#%% 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex
from sympy.abc import x, y
# 导入符号变量
import os
from matplotlib import cm
# 导入色谱模块
# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.labelspacing'] = 0.2

num = 301
# 数列元素数量
# 1. 定义函数
x_array = np.linspace(-3, 3, num)
y_array = np.linspace(-3, 3, num)
xx, yy = np.meshgrid(x_array,y_array)
# 产生网格数据

# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x,y],f_xy)
# 将符号函数表达式转换为Python函数
ff = f_xy_fcn(xx,yy)


# 2. 平面等高线，填充
cmap_arrays = ['RdYlBu_r', 'Blues_r', 'rainbow', 'viridis']
# 四种不同色谱

levels = np.linspace(-10, 10, 21)
# 定义等高线高度
# 几幅图采用完全一样的等高线高度

# for 循环绘制四张图片
for cmap_idx in cmap_arrays:
    fig, ax = plt.subplots()
    colorbar = ax.contourf(xx,yy, ff, levels = levels, cmap=cmap_idx)
    ax.contour(xx,yy, ff, levels = levels, colors = 'k')
    # 绘制平面填充等高线
    cbar = fig.colorbar(colorbar, ax=ax)
    cbar.set_ticks([-10, -5, 0, 5, 10])
    cbar.ax.set_title('$\it{f}$($\it{x_1}$,$\it{x_2}$)',fontsize=8)
    # 增加色谱条，并指定刻度

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    # 控制横轴、纵轴取值范围

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.gca().set_aspect('equal', adjustable='box')
    # 横纵轴比例尺1:1
    title = 'Colormap = ' + str(cmap_idx)
    plt.title(title)
    # 给图像加标题
    plt.show()

# 3. 平面等高线，非填充
for cmap_idx in cmap_arrays:
    fig, ax = plt.subplots()
    # 绘制平面等高线，非填充
    colorbar = ax.contour(xx,yy, ff, levels = levels, cmap=cmap_idx)

    cbar = fig.colorbar(colorbar, ax=ax)
    cbar.set_ticks([-10, -5, 0, 5, 10])
    cbar.ax.set_title('$\it{f}$($\it{x_1}$,$\it{x_2}$)',fontsize=8)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.gca().set_aspect('equal', adjustable='box')

    title = 'Colormap = ' + str(cmap_idx)
    plt.title(title)
    plt.show()

#===================================================================================
##  10 平面等高线
#===================================================================================
#%% 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# 一维
x_array = np.linspace(-3,3,20)
y_array = np.ones_like(x_array)

fig, ax = plt.subplots(figsize = (4,1))
ax.scatter(x_array, y_array, s = 15)
ax.plot(x_array, y_array, color = [0.5,0.5,0.5], linewidth = 1.25)
ax.axis('off')

plt.show()


fig, ax = plt.subplots(figsize = (1, 4))
ax.scatter(y_array, x_array, s = 15)
ax.plot(y_array, x_array, color = [0.5,0.5,0.5], linewidth = 1.25)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
ax.axis('off')
plt.show()

#%%  二维
## 1
xx1_square, xx2_square = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx1_square, xx2_square, xx1_square*0, color = [0.5, 0.5, 0.5], linewidth = 0.25)
ax.scatter(xx1_square, xx2_square, xx1_square*0, s = 5)
ax.set_proj_type('ortho') # 另外一种设定正交投影的方式
ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(xx1_square.min(), xx1_square.max())
ax.set_ylim(xx2_square.min(), xx2_square.max())
ax.set_box_aspect([1,1,1])
ax.view_init(azim=90, elev=90)
ax.grid(False)
plt.show()

## 2
xx1_square, xx2_square = np.meshgrid(np.linspace(-3,3,20),np.linspace(-3,3,20))
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx1_square, xx2_square, xx1_square*0, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.scatter(xx1_square, xx2_square, xx1_square*0, s = 5)
ax.set_proj_type('ortho') # 另外一种设定正交投影的方式
ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlim(xx1_square.min(), xx1_square.max())
ax.set_ylim(xx2_square.min(), xx2_square.max())
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-120, elev=30)
ax.grid(False)
plt.show()

#%% 二维函数
# 定义一个符号函数
from sympy import symbols
x,y = symbols("x,y")
from sympy import lambdify, exp

f = x*exp(-x**2 - y**2)
f_fcn = lambdify([x,y], f)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx1_square, xx2_square, f_fcn(xx1_square, xx2_square), color = [0.5,0.5,0.5], linewidth = 0.25)
# 3D散点图
ax.scatter(xx1_square, xx2_square, f_fcn(xx1_square, xx2_square), s = 5, c = f_fcn(xx1_square, xx2_square), cmap = 'RdYlBu_r')
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(xx1_square.min(), xx1_square.max())
ax.set_ylim(xx2_square.min(), xx2_square.max())
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-120, elev=30)
ax.grid(False)

plt.show()

#%% 极坐标网格
theta = np.linspace(0, 2*np.pi, 20)
r     = np.linspace(0, 3, 10)
tt, rr = np.meshgrid(theta,r)
xx1_polar = np.cos(tt)*rr
xx2_polar = np.sin(tt)*rr

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx1_polar, xx2_polar, xx2_polar*0, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.scatter(xx1_polar, xx2_polar, xx2_polar*0, s = 5)
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(xx1_polar.min(), xx1_polar.max())
ax.set_ylim(xx2_polar.min(), xx2_polar.max())
ax.set_box_aspect([1,1,1])
ax.view_init(azim=90, elev=90)
ax.grid(False)

plt.show()

## 1
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx1_polar, xx2_polar, xx2_polar*0, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.scatter(xx1_polar, xx2_polar, xx2_polar*0, s = 5)
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(xx1_polar.min(), xx1_polar.max())
ax.set_ylim(xx2_polar.min(), xx2_polar.max())
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-120, elev=30)
ax.grid(False)

plt.show()

## 2
f_fcn = lambdify([x,y], f)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx1_polar, xx2_polar, f_fcn(xx1_polar, xx2_polar), color = [0.5,0.5,0.5], linewidth = 0.25)
ax.scatter(xx1_polar, xx2_polar, f_fcn(xx1_polar, xx2_polar), s = 5, c = f_fcn(xx1_polar, xx2_polar), cmap = 'RdYlBu_r')
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(xx1_polar.min(), xx1_polar.max())
ax.set_ylim(xx2_polar.min(), xx2_polar.max())
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-120, elev=30)
ax.grid(False)

plt.show()

#%% 三角网格
xx1_square, xx2_square = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
points_square = np.column_stack((xx1_square.ravel(), xx2_square.ravel()))
triang_square_auto = mtri.Triangulation(points_square[:,0], points_square[:,1])

fig, ax = plt.subplots()
ax.triplot(triang_square_auto, 'o-', color = '#0070C0')
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.axis('off')
ax.set_xlim(xx1_square.min(), xx1_square.max())
ax.set_ylim(xx2_square.min(), xx2_square.max())

plt.show()

## 2
theta = np.linspace(0, 2*np.pi, 20)
r     = np.linspace(0, 3, 10)
tt, rr = np.meshgrid(theta,r)
xx1_polar = np.cos(tt)*rr
xx2_polar = np.sin(tt)*rr
points_polar = np.column_stack((xx1_polar.ravel(),xx2_polar.ravel()))
triang_polar_auto = mtri.Triangulation(points_polar[:,0], points_polar[:,1])
fig, ax = plt.subplots()
# Plot the triangulation.
ax.triplot(triang_polar_auto, 'o-', color = '#0070C0')
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
ax.set_xlim(xx1_polar.min(), xx1_polar.max())
ax.set_ylim(xx2_polar.min(), xx2_polar.max())

plt.show()


## 3
def circle_points(num_r, num_n):
    r = np.linspace(0, 3, num_r)
    # print(r)
    # 极轴 [0, 3] 分成若干等份
    n = r*num_n + 1
    n = n.astype(int)
    # print(n)
    # 每一层散点数
    circles = np.empty((0,2))
    # 创建空数组
    for r_i, n_i in zip(r, n):
        t_i = np.linspace(0, 2*np.pi, n_i, endpoint = False)
        r_i = np.ones_like(t_i)*r_i
        # 极坐标到直角坐标系转换
        x_i = r_i*np.cos(t_i)
        y_i = r_i*np.sin(t_i)

        circle_i = np.column_stack([x_i, y_i])
        # print(circle_i)
        # 拼接极坐标点
        circles = np.append(circles, circle_i, axis = 0)
    return circles

points_circles = circle_points(10, 20)
triang_circles_auto = mtri.Triangulation(points_circles[:,0], points_circles[:,1])
fig, ax = plt.subplots()
ax.triplot(triang_circles_auto, 'o-', color = '#0070C0')
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(xx1_polar.min(), xx1_polar.max())
ax.set_ylim(xx2_polar.min(), xx2_polar.max())
ax.axis('off')
plt.show()

#%%
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex
from sympy.abc import x, y
# 从SymPy库中导入符号变量 x 和 y
from matplotlib import cm
# 导入色谱模块

# 自定义函数
def mesh(num = 101):
    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)
    return xx, yy

# 定义曲面函数
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)
f_xy_fcn = lambdify([x,y],f_xy)

# 生成数据
xx, yy = mesh(num = 121)
x_array = np.linspace(-3,3,101)
ff = f_xy_fcn(xx,yy)

#############  三维等高线
z_level = 2
xx_, yy_ = np.meshgrid(np.linspace(-3, 3, 2),np.linspace(-3, 3, 2))
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (10, 10))

## 曲面和等高线
ax.plot_wireframe(xx, yy, ff, color = [0.6, 0.6, 0.6], rstride = 5, cstride = 5, linewidth = 0.25)
ax.contour(xx, yy, ff, levels = [z_level], colors = 'b', linewidths = 1)

##  切面
zz_ = np.zeros_like(xx_) + z_level
ax.plot_surface(xx_, yy_, zz_, color = 'b', alpha = 0.1) # 中间
ax.plot_wireframe(xx_, yy_, zz_, color = 'b', lw = 0.2)  # 边界，不能指定rstride, cstride

ax.set_proj_type('ortho') # 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8, 8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
plt.show()

#############  三维等高线
z_level = 2
xx_, yy_ = np.meshgrid(np.linspace(-3, 3, 2),np.linspace(-3, 3, 2))
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (10, 10))

## 曲面
norm_plt = plt.Normalize(ff.min(), ff.max())
colors = cm.RdYlBu_r(norm_plt(ff))
surf = ax.plot_surface(xx, yy, ff, facecolors = colors,
                        rstride = 2,
                        cstride = 2,
                        linewidth = 1, # 线宽
                        shade = False) # 删除阴影
surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

## 曲面和等高线
# ax.plot_wireframe(xx,yy, ff, color = [0.6, 0.6, 0.6], rstride=5, cstride=5, linewidth = 0.25)
ax.contour(xx, yy, ff, levels = [z_level], colors = 'b', linewidths = 1)

##  切面
zz_ = np.zeros_like(xx_) + z_level
ax.plot_surface(xx_, yy_, zz_, color = 'b', alpha = 0.1)
ax.plot_wireframe(xx_, yy_, zz_, color = 'b', lw = 0.2)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8, 8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
plt.show()

#############  三维等高线
z_level = 2
xx_, yy_ = np.meshgrid(np.linspace(-3, 3, 2), np.linspace(-3, 3, 2))
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (10, 10))

norm_plt = plt.Normalize(ff.min(), ff.max())
colors = cm.RdYlBu_r(norm_plt(ff))

## 曲面和等高线
# surf = ax.plot_surface(xx, yy, ff, color = 'r', alpha = 0.1, shade = False) # 删除阴影
surf = ax.plot_surface(xx, yy, ff, facecolors = colors, rstride = 2, cstride = 2, linewidth = 1, shade = False) # 删除阴影
# surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

# ax.plot_wireframe(xx, yy, ff, color = [0.6, 0.6, 0.6], rstride=5, cstride=5, linewidth = 0.25)
ax.contour(xx, yy, ff, levels = [z_level], colors = 'b', linewidths = 1)

##  切面
zz_ = np.zeros_like(xx_) + z_level
ax.plot_surface(xx_, yy_, zz_, color = 'b', alpha = 0.1)
ax.plot_wireframe(xx_, yy_, zz_, color = 'b', lw = 0.2)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8, 8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
plt.show()

############### 三维等高线到平面
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (10, 10))

ax.plot_wireframe(xx, yy, ff, color = [0.6, 0.6, 0.6], rstride=5, cstride=5, linewidth = 0.25)
ax.contour(xx, yy, ff, levels = [z_level], colors = 'b', linewidths = 1)
ax.contour(xx, yy, ff, levels = [z_level], zdir='z', offset=-8, colors = 'b')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8, 8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)

plt.show()

################## 一系列等高线
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (10, 10))

ax.plot_wireframe(xx, yy, ff, color = [0.6, 0.6, 0.6], rstride=5, cstride=5, linewidth = 0.25)

CS = ax.contour(xx, yy, ff, levels = np.linspace(-8,8,17), cmap = 'RdYlBu_r', linewidths = 1)

# fig.colorbar(CS)
# 增加色谱条
ax.contour(xx, yy, ff, levels = np.linspace(-8,8,17), zdir='z', offset=-8, cmap = 'RdYlBu_r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8, 8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)

plt.show()

#################  平面等高线
fig, ax = plt.subplots()

CS = ax.contour(xx, yy, ff, levels = np.linspace(-8,9,18), cmap = 'RdYlBu_r', linewidths = 1)
fig.colorbar(CS)

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')

ax.set_xticks([])
ax.set_yticks([])

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(False)
ax.set_aspect('equal', adjustable='box')
# fig.savefig('Figures/平面等高线.svg', format='svg')
plt.show()

############## 打印等高线数值
fig, ax = plt.subplots()

CS = ax.contour(xx, yy, ff, levels = np.linspace(-8,9,18), cmap = 'RdYlBu_r', linewidths = 1)

ax.clabel(CS, fmt = '%2.1f', colors = 'k', fontsize=10)

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(False)
ax.set_aspect('equal', adjustable='box')

plt.show()

#################### 单色等高线
fig, ax = plt.subplots()

ax.contour(xx, yy, ff, levels = np.linspace(-8,9,18), colors = 'k', linewidths = 1)
# 负数用虚线，默认

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(False)
ax.set_aspect('equal', adjustable='box')

plt.show()

################### 填充等高线，空间
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx, yy, ff, color = [0.6, 0.6, 0.6], rstride=5, cstride=5, linewidth = 0.25)
CS = ax.contourf(xx, yy, ff, levels = np.linspace(-8,9,18), cmap = 'RdYlBu_r')
fig.colorbar(CS)
ax.contourf(xx, yy, ff, levels = np.linspace(-8,9,18), zdir='z', offset=-8, cmap = 'RdYlBu_r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8, 8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)

plt.show()

################## 填充等高线，平面

fig, ax = plt.subplots()

CS_filled = ax.contourf(xx, yy, ff, levels = np.linspace(-8,9,18), cmap = 'RdYlBu_r')
fig.colorbar(CS_filled)
CS = ax.contour(xx, yy, ff, levels = [0], colors = 'k', linewidths = 1)
ax.clabel(CS, fmt = '%2.1f', colors = 'k', fontsize=10)

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(False)
ax.set_aspect('equal', adjustable='box')

plt.show()








































































































































