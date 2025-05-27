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
    ax.contour(xx, yy, ff, levels = levels, colors = 'k')
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
#%%   BK_2_Ch10_01
### 一维
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
# fig.savefig('Figures/三角网格，极坐标网格散点.svg', format='svg')
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
# fig.savefig('Figures/三角网格，均匀圆盘散点.svg', format='svg')
plt.show()

#%% # 等高线原理 BK_2_Ch10_02

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

#%% BK_2_Ch10_03  # 绘制决策边界
import numpy as np
from matplotlib import pyplot as plt
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# Create color maps for 3-class classification problem, as with iris
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

x_min, x_max = 4, 8
y_min, y_max = 2, 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 100))



classifiers = [
    DecisionTreeClassifier(max_depth=5),
    GaussianNB(),
]
fig = plt.figure(figsize=(8, 12))
for idx, classifier_idx in enumerate(classifiers):
    ax = fig.add_subplot(2, 1, idx + 1)
    classifier_idx.fit(X, y)

    Z = classifier_idx.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlabel('Sepal length (cm)')
    plt.ylabel('Sepal width (cm)')
    plt.axis('tight')
# fig.savefig('Figures/决策树_朴素贝叶斯分类_contourf.svg', format='svg')



#%% BK_2_Ch10_04 # 等高线颗粒度

# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex
from sympy.abc import x, y
# 导入符号变量
from matplotlib import cm
# 导入色谱模块
import os


# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2)- 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x,y],f_xy)
# 将符号函数表达式转换为Python函数
### 颗粒度过低
def mesh(num = 101):
    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)
    return xx, yy

xx, yy = mesh(num = 7)
ff = f_xy_fcn(xx,yy)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx,yy, ff, color = [0.6, 0.6, 0.6], rstride=1, cstride=1, linewidth = 0.25)
ax.scatter(xx,yy, ff, marker = '.', color = 'k')
CS = ax.contour(xx, yy, ff, levels = np.linspace(-8,8,17), cmap = 'RdYlBu_r', linewidths = 1)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8, 8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/颗粒度过低，三维.svg', format='svg')
plt.show()
plt.close('all')


fig, ax = plt.subplots()
ax.scatter(xx, yy, marker = '.', color = 'k')
CS = ax.contour(xx, yy, ff, levels = np.linspace(-8,9,18), cmap = 'RdYlBu_r', linewidths = 1)
# fig.colorbar(CS)

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(False)
ax.set_aspect('equal', adjustable='box')
# fig.savefig('Figures/颗粒度过低，平面.svg', format='svg')



### 颗粒度合理
xx, yy = mesh(num = 6*50 + 1)
ff = f_xy_fcn(xx,yy)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx, yy, ff, color = [0.6, 0.6, 0.6], rstride=10, cstride=10, linewidth = 0.25)
# ax.scatter(xx,yy, ff,marker = '.', color = 'k')
CS = ax.contour(xx, yy, ff, levels = np.linspace(-8,8,17), cmap = 'RdYlBu_r', linewidths = 1)
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
# fig.savefig('Figures/颗粒度合理.svg', format='svg')


##2D courter
fig, ax = plt.subplots()
# ax.scatter(xx, yy, marker = '.', color = 'k')
CS = ax.contour(xx, yy, ff, levels = np.linspace(-8,9,18), cmap = 'RdYlBu_r', linewidths = 1)
# fig.colorbar(CS)

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(False)
ax.set_aspect('equal', adjustable='box')
# fig.savefig('Figures/颗粒度合理，平面.svg', format='svg')



#%%BK_2_Ch10_05 线性、非线性几何变换
import numpy as np
import matplotlib.pyplot as plt
import os
T = np.linspace(-2, 2, 2048)
X, Y = np.meshgrid(T, T)

fig = plt.figure(figsize=(4, 4))
# 原图
ax = fig.add_subplot(111)
ax.contour(np.abs(X - np.round(X)), levels = 1, colors="black", linewidths=0.25)
ax.contour(np.abs(Y - np.round(Y)), levels = 1, colors="black", linewidths=0.25)
# 这段代码绘制了一个包含黑色轮廓线的图形，
# 其中轮廓线表示了X和Y坐标与其最近整数的差的绝对值为1的区域。
ax.set_xticks([])
ax.set_yticks([])
# fig.savefig('Figures/原始网格.svg', format='svg')


fig = plt.figure(figsize=(8, 12))
# 缩放
X_,Y_ = 2 * X, Y
ax = fig.add_subplot(3, 2, 1)
ax.contour(np.abs(X_ - np.round(X_)), 1, colors="black", linewidths=0.25)
ax.contour(np.abs(Y_ - np.round(Y_)), 1, colors="black", linewidths=0.25)
# 这段代码绘制了一个包含黑色轮廓线的图形， 其中轮廓线表示了X和Y坐标与其最近整数的差的绝对值为1的区域。
ax.set_xticks([])
ax.set_yticks([])
X_,Y_ = X + Y, X - Y
ax = fig.add_subplot(3, 2, 2)
ax.contour(np.abs(X_ - np.round(X_)), 1, colors="black", linewidths=0.25)
ax.contour(np.abs(Y_ - np.round(Y_)), 1, colors="black", linewidths=0.25)

ax.set_xticks([])
ax.set_yticks([])

X_,Y_ = X, X + Y
ax = fig.add_subplot(3, 2, 3)

ax.contour(np.abs(X_ - np.round(X_)), 1, colors="black", linewidths=0.25)
ax.contour(np.abs(Y_ - np.round(Y_)), 1, colors="black", linewidths=0.25)

ax.set_xticks([])
ax.set_yticks([])

X_,Y_ = X**2,Y**2
ax = fig.add_subplot(3, 2, 4)

ax.contour(np.abs(X_ - np.round(X_)), 1, colors="black", linewidths=0.25)
ax.contour(np.abs(Y_ - np.round(Y_)), 1, colors="black", linewidths=0.25)

ax.set_xticks([])
ax.set_yticks([])

X_,Y_ = np.exp(X),np.exp(Y)
ax = fig.add_subplot(3, 2, 5)

ax.contour(np.abs(X_ - np.round(X_)), 1, colors="black", linewidths=0.25)
ax.contour(np.abs(Y_ - np.round(Y_)), 1, colors="black", linewidths=0.25)

ax.set_xticks([])
ax.set_yticks([])

X_,Y_ = 1/X,1/Y
ax = fig.add_subplot(3, 2, 6)

ax.contour(np.abs(X_ - np.round(X_)), 1, colors="black", linewidths=0.25)
ax.contour(np.abs(Y_ - np.round(Y_)), 1, colors="black", linewidths=0.25)

ax.set_xticks([])
ax.set_yticks([])
# fig.savefig('Figures/线性、非线性变换，第1组.svg', format='svg')


fig = plt.figure(figsize=(8, 12))
X_ = X + np.sin(Y * 2) * 0.2
Y_ = Y + np.cos(X * 2) * 0.3
ax = fig.add_subplot(3, 2, 1)
ax.contour(np.abs(X_ - np.round(X_)), 1, colors="black", linewidths=0.25)
ax.contour(np.abs(Y_ - np.round(Y_)), 1, colors="black", linewidths=0.25)

ax.set_xticks([])
ax.set_yticks([])

def Z_sq(X, Y):
    Z = X + 1j * Y
    c = Z ** 2
    X_ = c.real
    Y_ = c.imag
    return X_,Y_

X_,Y_ = Z_sq(X, Y)
ax = fig.add_subplot(3, 2, 2)
ax.contour(np.abs(X_ - np.round(X_)), 1, colors="black", linewidths=0.25)
ax.contour(np.abs(Y_ - np.round(Y_)), 1, colors="black", linewidths=0.25)

ax.set_xticks([])
ax.set_yticks([])

def Z_sq(X, Y):
    Z = X + 1j * Y
    c = np.sqrt(Z ** 2 + 1)
    X_ = c.real
    Y_ = c.imag
    return X_,Y_

X_,Y_ = Z_sq(X, Y)
ax = fig.add_subplot(3, 2, 3)

ax.contour(np.abs(X_ - np.round(X_)), 1, colors="black", linewidths=0.25)
ax.contour(np.abs(Y_ - np.round(Y_)), 1, colors="black", linewidths=0.25)

ax.set_xticks([])
ax.set_yticks([])

def Z_cubic(X, Y):
    Z = X + 1j * Y
    c = Z**3
    X_ = c.real
    Y_ = c.imag
    return X_,Y_

X_,Y_ = Z_cubic(X, Y)
ax = fig.add_subplot(3, 2, 4)

ax.contour(np.abs(X_ - np.round(X_)), 1, colors="black", linewidths=0.25)
ax.contour(np.abs(Y_ - np.round(Y_)), 1, colors="black", linewidths=0.25)

ax.set_xticks([])
ax.set_yticks([])

def Z_fourth(X, Y):
    Z = X + 1j * Y
    c = Z**4
    X_ = c.real
    Y_ = c.imag
    return X_,Y_
X_,Y_ = Z_fourth(X, Y)
ax = fig.add_subplot(3, 2, 5)

ax.contour(np.abs(X_ - np.round(X_)), 1, colors="black", linewidths=0.25)
ax.contour(np.abs(Y_ - np.round(Y_)), 1, colors="black", linewidths=0.25)

ax.set_xticks([])
ax.set_yticks([])

def Z_exp(X, Y):
    Z = X + 1j * Y
    c = np.exp(Z)
    X_ = c.real
    Y_ = c.imag
    return X_,Y_

X_,Y_ = Z_exp(X, Y)
ax = fig.add_subplot(3, 2, 6)

ax.contour(np.abs(X_ - np.round(X_)), 1, colors="black", linewidths=0.25)
ax.contour(np.abs(Y_ - np.round(Y_)), 1, colors="black", linewidths=0.25)

ax.set_xticks([])
ax.set_yticks([])
# fig.savefig('Figures/线性、非线性变换，第2组.svg', format='svg')
plt.show()
plt.close('all')

#%% BK_2_Ch10_06 三角剖分可视化, 参见 Book2_Chap18, Book2_Chap10,
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

### 定义符号函数
# 定义一个符号函数
from sympy import symbols
x,y = symbols("x,y")
from sympy import lambdify, exp

f = x*exp(-x**2 - y**2)
f_fcn = lambdify([x,y], f)

### 定义三角剖分点
# 生成了10个数据点，其中x和y分别表示每个数据点在x轴和y轴上的坐标
x_tri = np.asarray([-3., -1.,  1.,  3., -2.,  0.,  2., -1.,  1.,  0.])
y_tri = np.asarray([-3., -3., -3., -3., -1., -1., -1.,  1.,  1.,  3.])
points = np.column_stack([x_tri,y_tri])

### 采用 scipy.spatial.Delaunay 完成三角剖分
from scipy.spatial import Delaunay
tri_from_scipy = Delaunay(points)
# scipy.spatial中的Delaunay类可以帮助我们生成一个点集的Delaunay三角剖分， 它可以用于构建三角形网格、寻找最近邻、计算凸壳等等

fig, ax = plt.subplots(figsize = (5,5))
ax.triplot(points[:,0], points[:,1], tri_from_scipy.simplices)

# 通过triplot方法，我们可以使用三角形剖分对象来绘制三角形网格

ax.plot(points[:,0], points[:,1], '.r', markersize = 10, lw = 0.25)
ax.set_aspect('equal')
ax.set_xlim(-3,3); ax.set_ylim(-3,3)
# fig.savefig('Figures/triplot绘制三角剖分网格.svg', format='svg')
plt.show()
plt.close()


### 使用Triangulation
# 自定义哪些点构成一个三角形，逆时针顺序
triangles = [[0, 1, 4], [1, 2, 5], [2, 3, 6], [1, 5, 4], [2, 6, 5], [4, 5, 7],
             [5, 6, 8], [5, 8, 7], [7, 8, 9]]
triang = mtri.Triangulation(x_tri, y_tri, triangles)
# 通过Triangulation方法，我们可以使用x和y数组生成一个三角形剖分对象

### 等高线
fig, ax = plt.subplots(figsize = (5,5))
# Plot the triangulation.
ax.tricontourf(triang, f_fcn(x_tri, y_tri), cmap = 'RdYlBu_r')
ax.triplot(triang, 'ko-')
ax.set_aspect('equal')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
# fig.savefig('Figures/tricontourf绘制三角网格等高线.svg', format='svg')
plt.show()
plt.close()

### 采用Delaunay三角剖分，自动完成三角剖分
# import matplotlib.tri as mtri
triang_auto = mtri.Triangulation(x_tri, y_tri)
# 采用德劳内三角剖分 (Delaunay triangulation)
# 参考：
# https://mathworld.wolfram.com/DelaunayTriangulation.html

fig, ax = plt.subplots(figsize = (5,5))

# 三角剖分网格绘制等高线
ax.tricontourf(triang_auto, f_fcn(x_tri, y_tri),  cmap = 'RdYlBu_r', levels = 20)
ax.triplot(triang_auto, 'ko-')
ax.set_aspect('equal')
ax.set_xlim(-3,3); ax.set_ylim(-3,3)
# fig.savefig('Figures/自动完成三角剖分.svg', format='svg')
plt.show()
plt.close()

### 网格化散点三角剖分
xx,yy = np.meshgrid(np.linspace(-3,3,20),np.linspace(-3,3,20))
triang_mesh_auto = mtri.Triangulation(xx.ravel(), yy.ravel())
fig, ax = plt.subplots(figsize = (5,5))

# Plot the triangulation.
ax.tricontourf(triang_mesh_auto, f_fcn(xx.ravel(),yy.ravel()), cmap = 'RdYlBu_r', levels = 20)
ax.triplot(triang_mesh_auto, 'ko-')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
# fig.savefig('Figures/网格化散点三角剖分.svg', format='svg')
plt.show()
plt.close()

### 极坐标网络三角剖分
rr, tt = np.meshgrid(np.linspace(3, 0, 10, endpoint = True), np.linspace(0, 2*np.pi, 20, endpoint = True))
xx_polar = rr*np.cos(tt)
yy_polar = rr*np.sin(tt)
triang_polar_auto = mtri.Triangulation(xx_polar.ravel(), yy_polar.ravel())

fig, ax = plt.subplots(figsize = (5,5))
# Plot the triangulation.
ax.tricontourf(triang_polar_auto, f_fcn(xx_polar.ravel(),yy_polar.ravel()), cmap = 'RdYlBu_r', levels = 20)
ax.triplot(triang_polar_auto, 'ko-')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_aspect('equal', adjustable='box')
# fig.savefig('Figures/极坐标网络三角剖分.svg', format='svg')
plt.show()
plt.close()


fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='3d')

surf = ax.plot_trisurf(xx_polar.ravel(),  yy_polar.ravel(),  f_fcn(xx_polar.ravel(), yy_polar.ravel()),  cmap='RdYlBu_r', linewidth=0.2, shade=False)
surf.set_facecolor((0,0,0,0))
ax.set_proj_type('ortho')
ax.view_init(azim=-120, elev=30)
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.grid(False)
ax.set_box_aspect(aspect = (1,1,1))
# fig.savefig('Figures/极坐标网络三角剖分，曲面.svg', format='svg')
plt.show()
plt.close()


### 均匀圆盘网格
def circle_points(num_r, num_n):
    r = np.linspace(0,3,num_r)
    # print(r)
    # 极轴 [0, 3] 分成若干等份
    n = r*num_n + 1
    n = n.astype(int)
    # print(n)
    # 每一层散点数
    circles = np.empty((0,2))
    # 创建空数组
    for r_i, n_i in zip(r, n):
        t_i = np.linspace(0, 2*np.pi, n_i, endpoint=False)
        r_i = np.ones_like(t_i)*r_i

        x_i = r_i*np.cos(t_i)
        y_i = r_i*np.sin(t_i)
        # 极坐标到直角坐标系转换

        circle_i = np.column_stack([x_i, y_i])
        # print(circle_i)
        circles = np.append(circles,circle_i, axis=0)
        # 拼接极坐标点
    return circles
points_circles = circle_points(10, 20)
triang_circles_auto = mtri.Triangulation(points_circles[:,0], points_circles[:,1])

fig, ax = plt.subplots(figsize = (5,5))
# Plot the triangulation.
ax.tricontourf(triang_circles_auto, f_fcn(points_circles[:,0], points_circles[:,1]),  cmap = 'RdYlBu_r', levels = 20)
ax.triplot(triang_circles_auto, 'ko-')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_aspect('equal', adjustable='box')
# fig.savefig('Figures/均匀圆盘网格，等高线.svg', format='svg')
plt.show()
plt.close()


fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='3d')
# ax.scatter(points_circles[:,0],
#                 points_circles[:,1],
#                 f_fcn(points_circles[:,0], points_circles[:,1]), zorder = 1e6, c = 'k')
surf = ax.plot_trisurf(points_circles[:,0], points_circles[:,1], f_fcn(points_circles[:,0], points_circles[:,1]),  cmap='RdYlBu_r', linewidth=0.2, shade=False)
surf.set_facecolor((0,0,0,0))
ax.set_proj_type('ortho')
ax.view_init(azim=-120, elev=30)
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.grid(False)
ax.set_box_aspect(aspect = (1,1,1))
# fig.savefig('Figures/均匀圆盘网格，曲面.svg', format='svg')
plt.show()
plt.close()


### 不规则三角形
x_sq, y_sq = np.random.rand(2, 100)*6 - 3
points_rnd_sq = np.column_stack([x_sq,y_sq])
tri_points_rnd_sq_from_scipy = Delaunay(points_rnd_sq)
fig, ax = plt.subplots(figsize = (5,5))
ax.triplot(points_rnd_sq[:,0], points_rnd_sq[:,1], tri_points_rnd_sq_from_scipy.simplices)
ax.plot(points_rnd_sq[:,0], points_rnd_sq[:,1], '.r', markersize = 10)
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_aspect('equal', adjustable='box')
# fig.savefig('Figures/不规则三角形网格.svg', format='svg')
plt.show()
plt.close()


triang_auto_rnd_sq = mtri.Triangulation(points_rnd_sq[:,0], points_rnd_sq[:,1])
fig, ax = plt.subplots(figsize = (5,5))
# Plot the triangulation.
ax.tricontourf(triang_auto_rnd_sq, f_fcn(points_rnd_sq[:,0], points_rnd_sq[:,1]), cmap = 'RdYlBu_r', levels = 20)
ax.triplot(triang_auto_rnd_sq, 'ko-')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_aspect('equal', adjustable='box')
# fig.savefig('Figures/不规则三角形网格，等高线.svg', format='svg')
plt.show()
plt.close()


fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='3d')
# ax.scatter(points_circles[:,0],
#                 points_circles[:,1],
#                 f_fcn(points_circles[:,0], points_circles[:,1]), zorder = 1e6, c = 'k')
surf = ax.plot_trisurf(points_rnd_sq[:,0], points_rnd_sq[:,1],  f_fcn(points_rnd_sq[:,0], points_rnd_sq[:,1]), cmap='RdYlBu_r', linewidth=0.2, shade=False)
surf.set_facecolor((0,0,0,0))
ax.set_proj_type('ortho')
ax.view_init(azim=-120, elev=30)
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.grid(False)
ax.set_box_aspect(aspect = (1,1,1))
# fig.savefig('Figures/不规则三角形网格，曲面.svg', format='svg')
plt.show()
plt.close()

#%% BK_2_Ch10_07 三角剖分颗粒度, 参考 Bk_2_Ch32_06, BK_2_Ch10_07, Bk_2_Ch18_03
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

### 最小颗粒度
corners = np.array([[0, 0], [1, 0], [0.5,0.75**0.5]])
# 定义等边三角形的三个顶点

triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
# 构造三角形剖分对象

from scipy.spatial import Delaunay
tri_from_scipy = Delaunay(corners)

fig, ax = plt.subplots(figsize = (5,5))
ax.triplot(corners[:,0], corners[:,1], tri_from_scipy.simplices)
ax.plot(corners[:,0], corners[:,1], '.r', markersize = 10)

ax.set_aspect('equal')
ax.set_xlim(0,1); ax.set_ylim(0,1)
# fig.savefig('Figures/三角剖分，最小颗粒度.svg', format='svg')
plt.show()
plt.close()

### 不断提高颗粒度
refiner = tri.UniformTriRefiner(triangle)
#对三角形网格进行均匀细化，生成更密集的三角形网格，以提高绘制的精细度和准确性
subdiv_array = [1,2,3,4]
fig, ax = plt.subplots(figsize = (12,3))
for idx, subdiv_idx in enumerate(subdiv_array,1):
    trimesh_idx = refiner.refine_triangulation(subdiv=subdiv_idx)
    # 等边三角形被细化成4**subdiv 个三角形

    plt.subplot(1,4, idx)
    plt.triplot(trimesh_idx)
    plt.axis('off'); plt.axis('equal')
    plt.title('Small triangles = ' + str(4**subdiv_idx))
# fig.savefig('Figures/三角剖分，不断提高颗粒度.svg', format='svg')
plt.show()
plt.close()

























































