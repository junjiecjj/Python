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

































































