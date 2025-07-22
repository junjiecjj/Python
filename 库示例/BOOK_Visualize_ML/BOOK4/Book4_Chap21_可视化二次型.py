#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 17:20:22 2025

@author: jack
"""

#%% Bk2_Ch21_01   二元二次型
import numpy as np
import matplotlib.pyplot as plt
import os
from sympy import symbols, diff, lambdify, expand, simplify

# 二元函数
def fcn_n_grdnt(A, xx1, xx2):
    x1, x2 = symbols('x1 x2')
    # 符号向量
    x = np.array([[x1,x2]]).T
    # 二次型
    f_x = x.T@A@x
    f_x = f_x[0][0]
    print(simplify(expand(f_x)))

    # 计算梯度，符号
    grad_f = [diff(f_x, var) for var in (x1, x2)]

    # 计算二元函数值 f(x1, x2)
    f_x_fcn = lambdify([x1, x2], f_x)
    ff_x = f_x_fcn(xx1, xx2)

    # 梯度函数
    grad_fcn = lambdify([x1, x2], grad_f)

    # 采样，降低颗粒度
    xx1_ = xx1[::20,::20]
    xx2_ = xx2[::20,::20]

    # 计算梯度
    V = grad_fcn(xx1_,xx2_)

    # 修复梯度值
    if isinstance(V[1], int):
        V[1] = np.zeros_like(xx1_)

    if isinstance(V[0], int):
        V[0] = np.zeros_like(xx1_)
    return ff_x, V

# 可视化函数
def visualize(xx1, xx2, f2_array, gradient_array):
    fig = plt.figure( figsize=(12, 6), constrained_layout = True)
    # 第一幅子图
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.plot_wireframe(xx1, xx2, f2_array, rstride=10, cstride=10, color = [0.8,0.8,0.8], linewidth = 0.25)
    ax.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')
    ax.set_xlabel(r'$x_1$');
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$f(x_1,x_2)$')
    ax.set_proj_type('ortho')
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(azim=-120, elev=30);
    ax.grid(False)
    ax.set_xlim(xx1.min(), xx1.max());
    ax.set_ylim(xx2.min(), xx2.max())

    # 第二幅子图
    ax = fig.add_subplot(1, 3, 2)
    ax.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')
    xx1_ = xx1[::20,::20]
    xx2_ = xx2[::20,::20]
    # color_array = np.sqrt(gradient_array[0]**2 + gradient_array[1]**2)
    # ax.quiver(xx1_, xx2_, gradient_array[0], gradient_array[1], color_array, angles='xy', scale_units='xy', cmap = 'RdYlBu_r', edgecolor='none', alpha=0.8)
    ax.quiver(xx1_, xx2_, gradient_array[0], gradient_array[1],  angles='xy', scale_units='xy', cmap = 'RdYlBu_r', edgecolor='none', alpha=0.8)
    ax.set_xlabel(r'$x_1$');
    ax.set_ylabel(r'$x_2$')
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.set_aspect('equal');
    ax.grid(False)
    ax.set_xlim(xx1.min(), xx1.max());
    ax.set_ylim(xx2.min(), xx2.max())

    # 第三幅子图
    ax = fig.add_subplot(1, 3, 3)
    ax.contourf(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')
    # color_array = np.sqrt(gradient_array[0]**2 + gradient_array[1]**2)
    # ax.quiver(xx1_, xx2_, gradient_array[0], gradient_array[1], color_array, angles='xy', scale_units='xy', cmap = 'RdYlBu_r', edgecolor='none', alpha=0.8)
    ax.quiver(xx1_, xx2_, gradient_array[0], gradient_array[1],  angles='xy', scale_units='xy', cmap = 'RdYlBu_r', edgecolor='none', alpha=0.8)

    ax.set_xlabel(r'$x_1$');
    ax.set_ylabel(r'$x_2$')
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlim(xx1.min(), xx1.max());
    ax.set_ylim(xx2.min(), xx2.max())
    return

## 生成网格化数据
x1_array = np.linspace(-2, 2, 201)
x2_array = np.linspace(-2, 2, 201)
xx1, xx2 = np.meshgrid(x1_array, x2_array)

## 正定性
A = np.array([[1, 0],
              [0, 1]])
f2_array, gradient_array = fcn_n_grdnt(A, xx1, xx2)
visualize(xx1, xx2, f2_array, gradient_array)

## 半正定
A = np.array([[1, 0],
              [0, 0]])
f2_array, gradient_array = fcn_n_grdnt(A, xx1, xx2)
visualize(xx1,xx2,f2_array,gradient_array)


## 负定
A = np.array([[-1, 0],
              [0, -1]])
f2_array, gradient_array = fcn_n_grdnt(A, xx1, xx2)
visualize(xx1,xx2,f2_array,gradient_array)


## 半负定
A = np.array([[-1, 0],
              [0,  0]])
f2_array, gradient_array = fcn_n_grdnt(A, xx1, xx2)
visualize(xx1,xx2,f2_array,gradient_array)

## 不定
A = np.array([[0, 1],
              [1, 0]])
f2_array, gradient_array = fcn_n_grdnt(A, xx1, xx2)
visualize(xx1,xx2,f2_array,gradient_array)



#%%
import sympy
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as L

def mesh_circ(c1, c2, r, num):
    theta = np.linspace(0, 2*np.pi, num)
    r     = np.linspace(0,r, num)
    theta,r = np.meshgrid(theta,r)
    xx1 = np.cos(theta)*r + c1
    xx2 = np.sin(theta)*r + c2
    return xx1, xx2

def fcn_n_grdnt(A, xx1, xx2, xx1_, xx2_, t):
    x1, x2 = symbols('x1 x2')
    x = np.array([[x1, x2]]).T
    f_x = x.T@A@x
    f_x = f_x[0][0]
    # f_x = sqrt(f_x)
    # print(simplify(expand(f_x)))

    # take the gradient symbolically
    grad_f = [diff(f_x,var) for var in (x1,x2)]
    f_x_fcn = lambdify([x1, x2], f_x)
    ff_x = f_x_fcn(xx1,xx2)

    #turn into a bivariate lambda for numpy
    grad_fcn = lambdify([x1,x2],grad_f)
    # xx1_ = xx1[::20,::20]
    # xx2_ = xx2[::20,::20]
    V = grad_fcn(xx1_, xx2_)
    if isinstance(V[1], int):
        V[1] = np.zeros_like(V[0])

    elif isinstance(V[0], int):
        V[0] = np.zeros_like(V[1])

    # t = np.linspace(0, np.pi*2, 100)
    r = 2
    f_circle = f_x_fcn(2*np.cos(t), 2*np.sin(t))
    return ff_x, V, f_circle


# 可视化函数
def visualize(xx1, xx2, f2_array, xx1_, xx2_, gradient_array, t, f_circle):
    fig = plt.figure( figsize=(12, 6), constrained_layout = True)
    # 第一幅子图
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.plot_wireframe(xx1, xx2, f2_array, rstride=10, cstride=10, color = [0.8,0.8,0.8], linewidth = 0.25)
    ax.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')
    ax.plot(2*np.cos(t), 2*np.sin(t), f_circle, color = 'k')
    ax.set_xlabel(r'$x_1$');
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$f(x_1,x_2)$')
    ax.set_proj_type('ortho')
    # ax.set_xticks([]);
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.view_init(azim=-120, elev=30);
    ax.grid(False)
    ax.set_xlim(xx1.min(), xx1.max());
    ax.set_ylim(xx2.min(), xx2.max())

    # 第二幅子图
    ax = fig.add_subplot(1, 3, 2)
    ax.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')
    # xx1_ = xx1[::20,::20]
    # xx2_ = xx2[::20,::20]
    # color_array = np.sqrt(gradient_array[0]**2 + gradient_array[1]**2)
    # ax.quiver(xx1_, xx2_, gradient_array[0], gradient_array[1], color_array, angles='xy', scale_units='xy', cmap = 'RdYlBu_r', edgecolor='none', alpha=0.8)
    ax.quiver(xx1_, xx2_, gradient_array[0], gradient_array[1],  angles='xy', scale_units='xy', cmap = 'RdYlBu_r', edgecolor='none', alpha=0.8)
    ax.set_xlabel(r'$x_1$');
    ax.set_ylabel(r'$x_2$')
    # ax.set_xticks([]);
    # ax.set_yticks([])
    ax.set_aspect('equal');
    ax.grid(False)
    ax.set_xlim(xx1.min(), xx1.max());
    ax.set_ylim(xx2.min(), xx2.max())

    # 第三幅子图
    ax = fig.add_subplot(1, 3, 3)
    ax.contourf(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')
    # color_array = np.sqrt(gradient_array[0]**2 + gradient_array[1]**2)
    # ax.quiver(xx1_, xx2_, gradient_array[0], gradient_array[1], color_array, angles='xy', scale_units='xy', cmap = 'RdYlBu_r', edgecolor='none', alpha=0.8)
    ax.quiver(xx1_, xx2_, gradient_array[0], gradient_array[1],  angles='xy', scale_units='xy', cmap = 'RdYlBu_r', edgecolor='none', alpha=0.8)

    ax.set_xlabel(r'$x_1$');
    ax.set_ylabel(r'$x_2$')
    # ax.set_xticks([]);
    # ax.set_yticks([])
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlim(xx1.min(), xx1.max());
    ax.set_ylim(xx2.min(), xx2.max())
    return

## 生成网格化数据
r = 4
xx1, xx2 = mesh_circ(0, 0, r, 201)
x1_array = np.linspace(-r, r, 20)
x2_array = np.linspace(-r, r, 20)
xx1_, xx2_ = np.meshgrid(x1_array, x2_array)
mask_inside = (xx1_**2 + xx2_**2 <= r**2)
mask_idx = np.where(mask_inside == True)
xx1_ = xx1_[mask_idx]
xx2_ = xx2_[mask_idx]

t = np.linspace(0, np.pi*2, 100)

## 正定性
A = np.array([[1, 0],
              [0, 1]])
f2_array, gradient_array, f_circle = fcn_n_grdnt(A, xx1, xx2, xx1_, xx2_, t)
visualize(xx1, xx2, f2_array, xx1_, xx2_, gradient_array, t, f_circle)

## 半正定
A = np.array([[1, 0],
              [0, 0]])
f2_array, gradient_array, f_circle = fcn_n_grdnt(A, xx1, xx2, xx1_, xx2_, t)
visualize(xx1, xx2, f2_array, xx1_, xx2_, gradient_array, t, f_circle)


## 负定
A = np.array([[-1, 0],
              [0, -1]])
f2_array, gradient_array, f_circle  = fcn_n_grdnt(A, xx1, xx2, xx1_, xx2_, t)
visualize(xx1, xx2, f2_array, xx1_, xx2_, gradient_array, t, f_circle)


## 半负定
A = np.array([[-1, 0],
              [0,  0]])
f2_array, gradient_array, f_circle = fcn_n_grdnt(A, xx1, xx2, xx1_, xx2_, t)
visualize(xx1, xx2, f2_array, xx1_, xx2_, gradient_array, t, f_circle)

## 不定
A = np.array([[0, 1],
              [1, 0]])
f2_array, gradient_array, f_circle = fcn_n_grdnt(A, xx1, xx2, xx1_, xx2_, t)
visualize(xx1, xx2, f2_array, xx1_, xx2_,gradient_array, t, f_circle)


#%% BK_2_Ch25_06 平面Lp范数等高线
## 创建数据
p_values = [1, 1.5, 2, 4, 8, np.inf]
# 给定不同p值
x1 = np.linspace(-2.5, 2.5, num=101);
x2 = x1;
xx1, xx2 = np.meshgrid(x1,x2)
## 自定义Lp范数函数
def Lp_norm(p):
    # 计算范数
    if np.isinf(p):
        zz = np.maximum(np.abs(xx1),np.abs(xx2))
    else:
        zz = ((np.abs((xx1))**p) + (np.abs((xx2))**p))**(1./p)
    return zz
## 可视化
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(6, 9))
for p, ax in zip(p_values, axes.flat):
    # 计算范数
    zz = Lp_norm(p)
    # 绘制等高线
    ax.contourf(xx1, xx2, zz, 20, cmap='RdYlBu_r')
    # 绘制Lp norm = 1的等高线
    ax.contour (xx1, xx2, zz, [1], colors='k', linewidths = 2)

    # 装饰
    ax.axhline(y=0, color='k', linewidth = 0.25)
    ax.axvline(x=0, color='k', linewidth = 0.25)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    ax.set_title('p = ' + str(p))
    ax.set_aspect('equal', adjustable='box')


###   平面Lp范数等高线3D
## 创建数据
p_values = [1, 1.5, 2, 4, 8, np.inf]
# 给定不同p值
x1 = np.linspace(-2.5, 2.5, num=101);
x2 = x1;
xx1, xx2 = np.meshgrid(x1,x2)
## 自定义Lp范数函数
def Lp_norm(p):
    # 计算范数
    if np.isinf(p):
        zz = np.maximum(np.abs(xx1),np.abs(xx2))
    else:
        zz = ((np.abs((xx1))**p) + (np.abs((xx2))**p))**(1./p)
    return zz
## 可视化
# fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(12, 18), projection = '3d')
fig = plt.figure(figsize=(12, 18), constrained_layout = True)
for i, p in enumerate(p_values):
    ax = fig.add_subplot(3, 2, i+1, projection='3d')
    # 计算范数
    zz = Lp_norm(p)

    ## 4 plot_wireframe() 绘制网格曲面 + 三维等高线

    ax.plot_wireframe(xx1, xx2, zz, color = [0.5,0.5,0.5], linewidth = 0.25)

    # 三维等高线
    # colorbar = ax.contour(xx,yy, ff,20,  cmap = 'RdYlBu_r')
    # 三维等高线
    colorbar = ax.contour(xx1, xx2, zz, 20,  cmap = 'hsv')
    # fig.colorbar(colorbar, ax = ax, shrink=0.5, aspect=20)

    # 二维等高线
    ax.contour(xx1, xx2, zz, zdir = 'z', offset= zz.min(), levels = 20, linewidths = 2, cmap = "hsv")  # 生成z方向投影，投到x-y平面

    fig.colorbar(colorbar, ax=ax, shrink=0.5, aspect=20)
    ax.set_proj_type('ortho')

    # 3D坐标区的背景设置为白色
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xlabel(r'$\it{x_1}$')
    ax.set_ylabel(r'$\it{x_2}$')
    ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')
    ax.set_title(f"p = {p}")
    ax.set_xlim(x1.min(), x1.max())
    ax.set_ylim(x2.min(), x2.max())

    ax.view_init(azim=-135, elev=30)
    ax.grid(False)
plt.show()


#%% BK_2_Ch25_07 三维空间Lp范数
# 导入包
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

## 自定义函数展示隐函数
def plot_implicit(fn, X_plot, Y_plot, Z_plot, ax, bbox):
    # 等高线的起止范围
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    ax.set_proj_type('ortho')
    # 绘制三条参考线
    k = 1.5
    ax.plot((xmin * k, xmax * k), (0, 0), (0, 0), 'k')
    ax.plot((0, 0), (ymin * k, ymax * k), (0, 0), 'k')
    ax.plot((0, 0), (0, 0), (zmin * k, zmax * k), 'k')
    # 等高线的分辨率
    A = np.linspace(xmin, xmax, 100)
    # 产生网格数据
    A1,A2 = np.meshgrid(A,A)
    # 等高线的分割位置
    B = np.linspace(xmin, xmax, 20)
    # 绘制 XY 平面等高线
    if X_plot == True:
        for z in B:
            X,Y = A1,A2
            Z = fn(X,Y,z)
            cset = ax.contour(X, Y, Z+z, [z],
                              zdir='z',
                              linewidths = 0.25,
                              colors = '#0066FF',
                              linestyles = 'solid')
    # 绘制 XZ 平面等高线
    if Y_plot == True:
        for y in B:
            X,Z = A1,A2
            Y = fn(X,y,Z)
            cset = ax.contour(X, Y+y, Z, [y],
                              zdir='y',
                              linewidths = 0.25,
                              colors = '#88DD66',
                              linestyles = 'solid')
    # 绘制 YZ 平面等高线
    if Z_plot == True:
        for x in B:
            Y,Z = A1,A2
            X = fn(x,Y,Z)
            cset = ax.contour(X+x, Y, Z, [x],
                              zdir='x',
                              linewidths = 0.25,
                              colors = '#FF6600',
                              linestyles = 'solid')
    ax.set_zlim(zmin * k,zmax * k)
    ax.set_xlim(xmin * k,xmax * k)
    ax.set_ylim(ymin * k,ymax * k)
    ax.set_box_aspect([1,1,1])
    ax.view_init(azim=-120, elev=30)
    ax.axis('off')

def visualize_four_ways(fn, title, bbox=(-2.5,2.5)):
    fig = plt.figure(figsize=(12, 4), constrained_layout = True)

    ax = fig.add_subplot(1, 4, 1, projection='3d')
    plot_implicit(fn, True, False, False, ax, bbox)

    ax = fig.add_subplot(1, 4, 2, projection='3d')
    plot_implicit(fn, False, True, False, ax, bbox)

    ax = fig.add_subplot(1, 4, 3, projection='3d')
    plot_implicit(fn, False, False, True, ax, bbox)

    ax = fig.add_subplot(1, 4, 4, projection='3d')
    plot_implicit(fn, True, True, True, ax, bbox)

## 可视化
def vector_norm(x,y,z):
    p = 4
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1

visualize_four_ways(vector_norm, 'norm_1000', bbox = (-1,1))


#%% # 平面Lp范数等高线  # Bk4_Ch3_01.py  Bk2_Ch25
import numpy as np
import matplotlib.pyplot as plt
import os
from sympy import init_printing, symbols, diff, lambdify, expand, simplify, sqrt
# init_printing("mathjax")


## 创建数据
p_values = [1, 1.5, 2, 4, 8, np.inf]
# 给定不同p值
x1 = np.linspace(-2.5, 2.5, num=101);
x2 = x1;
xx1, xx2 = np.meshgrid(x1,x2)

## 自定义Lp范数函数
def Lp_norm(p):
    # 计算范数
    if np.isinf(p):
        zz = np.maximum(np.abs(xx1),np.abs(xx2))
    else:
        zz = ((np.abs((xx1))**p) + (np.abs((xx2))**p))**(1./p)
    return zz

## 可视化
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(6, 9))
for p, ax in zip(p_values, axes.flat):
    # 计算范数
    zz = Lp_norm(p)
    # 绘制等高线
    ax.contourf(xx1, xx2, zz, 20, cmap='RdYlBu_r')
    # 绘制Lp norm = 1的等高线
    ax.contour (xx1, xx2, zz, [1], colors='k', linewidths = 2)

    # 装饰
    ax.axhline(y=0, color='k', linewidth = 0.25)
    ax.axvline(x=0, color='k', linewidth = 0.25)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    ax.set_title('p = ' + str(p))
    ax.set_aspect('equal', adjustable='box')


#%% # 平面Lp范数等高线
## 创建数据
p_values = [1, 1.5, 2, 4, 8, np.inf]
# 给定不同p值
x1 = np.linspace(-2.5, 2.5, num=101);
x2 = x1;
xx1, xx2 = np.meshgrid(x1,x2)

## 自定义Lp范数函数
def Lp_norm(p):
    # 计算范数
    if np.isinf(p):
        zz = np.maximum(np.abs(xx1),np.abs(xx2))
    else:
        zz = ((np.abs((xx1))**p) + (np.abs((xx2))**p))**(1./p)
    return zz

## 可视化
# fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(12, 18), projection = '3d')
fig = plt.figure(figsize=(12, 18), constrained_layout = True)
for i, p in enumerate(p_values):
    ax = fig.add_subplot(3, 2, i+1, projection='3d')
    # 计算范数
    zz = Lp_norm(p)

    ## 4 plot_wireframe() 绘制网格曲面 + 三维等高线
    ax.plot_wireframe(xx1, xx2, zz, color = [0.5,0.5,0.5], linewidth = 0.25)

    # 三维等高线
    # colorbar = ax.contour(xx,yy, ff,20,  cmap = 'RdYlBu_r')
    # 三维等高线
    colorbar = ax.contour(xx1, xx2, zz, 20,  cmap = 'hsv')
    # fig.colorbar(colorbar, ax = ax, shrink=0.5, aspect=20)

    # 二维等高线
    ax.contour(xx1, xx2, zz, zdir = 'z', offset= zz.min(), levels = 20, linewidths = 2, cmap = "hsv")  # 生成z方向投影，投到x-y平面

    fig.colorbar(colorbar, ax=ax, shrink=0.5, aspect=20)
    ax.set_proj_type('ortho')

    # 3D坐标区的背景设置为白色
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xlabel(r'$\it{x_1}$')
    ax.set_ylabel(r'$\it{x_2}$')
    ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')
    ax.set_title(f"p = {p}")
    ax.set_xlim(x1.min(), x1.max())
    ax.set_ylim(x2.min(), x2.max())

    ax.view_init(azim=-135, elev=30)
    ax.grid(False)
plt.show()
























































