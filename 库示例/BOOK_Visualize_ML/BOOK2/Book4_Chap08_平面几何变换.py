#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:56:10 2025

@author: jack
"""



#%% Bk4_Ch8_01.py

import matplotlib.pyplot as plt
import numpy as np

def plot_shape(X,copy = False):
    if copy:
        fill_color = np.array([255,236,255])/255
        edge_color = np.array([255,0,0])/255
    else:
        fill_color = np.array([219,238,243])/255
        edge_color = np.array([0,153,255])/255

    plt.fill(X[:,0], X[:,1],
             color = fill_color,
             edgecolor = edge_color)

    plt.plot(X[:,0], X[:,1],marker = 'x',
             markeredgecolor = edge_color*0.5,
             linestyle = 'None')

X = np.array([[1,1],
              [0,-1],
              [-1,-1],
              [-1,1]])

# visualizations

fig, ax = plt.subplots()

plot_shape(X)      # plot original

# translation
t1 = np.array([3,2]);
Z = X + t1
plot_shape(Z,True) # plot copy

t2 = np.array([-3,-2]);
Z = X + t2
plot_shape(Z,True) # plot copy

t3 = np.array([-2,3]);
Z = X + t3
plot_shape(Z,True) # plot copy

t4 = np.array([3,-3]);
Z = X + t4
plot_shape(Z,True) # plot copy

# Decorations
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.axis('equal')
plt.axis('square')
plt.axhline(y=0, color='k', linewidth = 0.25)
plt.axvline(x=0, color='k', linewidth = 0.25)
plt.xticks(np.arange(-5, 6))
plt.yticks(np.arange(-5, 6))
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')


#%% Bk4_Ch8_02.py
import matplotlib.pyplot as plt
import numpy as np

def plot_shape(X,copy = False):
    if copy:
        fill_color = np.array([255, 236, 255])/255
        edge_color = np.array([255, 0, 0])/255
    else:
        fill_color = np.array([219, 238, 243])/255
        edge_color = np.array([0, 153, 255])/255

    plt.fill(X[:,0], X[:,1],
             color = fill_color,
             edgecolor = edge_color)

    plt.plot(X[:,0], X[:,1],marker = 'x',
             markeredgecolor = edge_color*0.5,
             linestyle = 'None')

X = np.array([[1,1],
              [0,-1],
              [-1,-1],
              [-1,1]]) + np.array([3,3])

# visualizations

thetas = np.linspace(30, 330, num=11)

for theta in thetas:
    fig, ax = plt.subplots()
    theta = theta/180*np.pi;
    # rotation
    R = np.array([[np.cos(theta),  np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])

    Z = X@R;
    plot_shape(Z, True) # plot copy

    plot_shape(X)      # plot original

    # Decorations
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    plt.axis('equal')
    plt.axis('square')
    plt.axhline(y=0, color='k', linewidth = 0.25)
    plt.axvline(x=0, color='k', linewidth = 0.25)
    plt.xticks(np.arange(-5, 6))
    plt.yticks(np.arange(-5, 6))
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 展示旋转 Bk_2_Ch12_11
import matplotlib.pyplot as plt
import numpy as np

# 定义函数
def plot_shape(X, copy = False):
    if copy:
        fill_color = np.array([255, 236, 255])/255
        edge_color = np.array([255, 0, 0])/255
    else:
        fill_color = np.array([219, 238, 243])/255
        edge_color = np.array([0, 153, 255])/255
    plt.fill(X[:,0], X[:,1], color = fill_color, edgecolor = edge_color, alpha = 0.5)
    plt.plot(X[:,0], X[:,1],marker = 'x', markeredgecolor = edge_color*0.5, linestyle = 'None')
X = np.array([[1,1],
              [0,-1],
              [-1,-1],
              [-1,1]])
X = X + [2, 2]

# 可视化
num = 24
thetas = np.linspace(360/num, 360, num, endpoint = False)
fig, ax = plt.subplots(figsize = (10, 10))
plot_shape(X)      # plot original
for theta in thetas:
    theta = theta/180*np.pi;
    # rotation
    R = np.array([[np.cos(theta),  np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    Z = X@R
    # 旋转
    plot_shape(Z,True) # plot copy

# 装饰
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.axhline(y=0, color='k', linewidth = 0.25)
plt.axvline(x=0, color='k', linewidth = 0.25)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])

ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  平面仿射变换 Bk_2_Ch12_13, 线性变换

# 导入包
import numpy as np
import matplotlib.pyplot as plt

# 产生网格数据
x1 = np.arange(-20, 20 + 1, step = 1)
x2 = np.arange(-20, 20 + 1, step = 1)

XX1, XX2 = np.meshgrid(x1,x2)
X = np.column_stack((XX1.ravel(), XX2.ravel()))

# 自定义可视化函数
def visualize_transform(XX1, XX2, ZZ1, ZZ2, cube, arrows, fig_name):
    colors = np.arange(len(XX1.ravel()))
    fig, ax = plt.subplots(figsize = (5,5))
    # 绘制原始网格
    ax.plot(XX1 ,XX2, color = [0.8,0.8,0.8], lw = 0.25)
    ax.plot(XX1.T, XX2.T, color = [0.8,0.8,0.8], lw = 0.25)
    # plt.scatter(XX1.ravel(), XX2.ravel(), c = colors, s = 10, cmap = 'plasma', zorder=1e3)

    #绘制几何变换后的网格
    ax.plot(ZZ1, ZZ2, color = '#0070C0', lw = 0.25)
    ax.plot(ZZ1.T, ZZ2.T, color = '#0070C0', lw = 0.25)

    ax.fill(cube[:,0], cube[:,1], color = '#92D050', alpha = 0.5)
    ax.quiver(0, 0, arrows[0,0], arrows[0,1], color = 'r', angles='xy', scale_units='xy', scale=1)
    ax.quiver(0, 0, arrows[1,0], arrows[1,1], color = 'g', angles='xy', scale_units='xy', scale=1)

    ax.axis('scaled')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.axhline(y = 0, color = 'k')
    ax.axvline(x = 0, color = 'k')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # fig.savefig('Figures/' + fig_name + '.svg', format='svg')

#>>>>>>>>>>> 原始网格
colors = np.arange(len(XX1.ravel()))
fig, ax = plt.subplots(figsize = (5,5))
cube = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
arrows = np.array([[1, 0], [0, 1]])

# 绘制原始网格
ax.plot(XX1, XX2, color = '#0070C0', lw = 0.25)
ax.plot(XX1.T, XX2.T, color = '#0070C0', lw = 0.25)
ax.fill(cube[:,0], cube[:,1], color = '#92D050', alpha = 0.5)
ax.quiver(0,0,arrows[0,0], arrows[0,1], color = 'r', angles='xy', scale_units='xy', scale=1)
ax.quiver(0,0,arrows[1,0], arrows[1,1], color = 'g', angles='xy', scale_units='xy', scale=1)
ax.scatter(XX1, XX2, c = 'red')

plt.axis('scaled')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

#>>>>>>>>>>> 旋转
# 绕原点，逆时针旋转30
theta = 30/180*np.pi
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
Z = X@R.T
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))
fig_name = '逆时针旋转30度'

cube_ = cube @ R.T;
arrows_ = arrows @ R.T;
visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)
#>>>>>>>>>>> 等比例放大
S = np.array([[2, 0], [0, 2]])
Z = X@S
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '等比例放大'
cube_ = cube @ S.T;
arrows_ = arrows @ S.T;

visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)

#>>>>>>>>>>> 等比例缩小
S = np.array([[0.4, 0], [0,   0.4]])
Z = X@S;

ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '等比例缩小'
cube_ = cube @ S.T;
arrows_ = arrows @ S.T;

visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)

#>>>>>>>>>>> 非等比例缩放
S = np.array([[2, 0], [0, 0.5]])
Z = X@S;

ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '非等比例缩放'
cube_ = cube @ S.T;
arrows_ = arrows @ S.T;

visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)

#>>>>>>>>>>> 先缩放，再旋转
Z = X@S.T@R.T;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先缩放，再旋转'
cube_ = cube @S.T@R.T;
arrows_ = arrows @S.T@R.T;

visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)

#>>>>>>>>>>> 先旋转，再放大
Z = X@R.T@S.T
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先旋转，再缩放'
cube_ = cube @R.T@S.T;
arrows_ = arrows @R.T@S.T;

visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)

#>>>>>>>>>>> 沿横轴剪切
T = np.array([[1, 1.5], [0, 1]])
Z = X@T.T
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '沿横轴剪切'
cube_ = cube @T.T;
arrows_ = arrows @T.T;

visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk_2_Ch26_02 平面网格的仿射变换

# 导入包
import numpy as np
import matplotlib.pyplot as plt

# 产生网格数据
x1 = np.arange(-5, 5 + 1, step=1) # (11,)
x2 = np.arange(-5, 5 + 1, step=1)

XX1, XX2 = np.meshgrid(x1,x2) # (11, 11)
X = np.column_stack((XX1.ravel(), XX2.ravel())) #  (121, 2)

# 自定义可视化函数
def visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name):
    colors = np.arange(len(XX1.ravel()))

    fig, ax = plt.subplots(figsize = (5,5), constrained_layout = True)
    # 绘制原始网格
    plt.plot(XX1, XX2, color = [0.8,0.8,0.8], lw = 0.25)
    plt.plot(XX1.T, XX2.T, color = [0.8,0.8,0.8], lw = 0.25)
    # plt.scatter(XX1.ravel(), XX2.ravel(), c = colors, s = 10, cmap = 'plasma', zorder=1e3)

    #绘制几何变换后 的网格
    plt.plot(ZZ1,ZZ2,color = '#0070C0', lw = 0.25)
    plt.plot(ZZ1.T, ZZ2.T,color = '#0070C0', lw = 0.25)
    plt.scatter(ZZ1.ravel(), ZZ2.ravel(), c = colors, s = 10, cmap = 'rainbow', zorder=1e3)

    plt.axis('scaled')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.axhline(y = 0, color = 'k')
    ax.axvline(x = 0, color = 'k')
    plt.xticks([])
    plt.yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

#>>>>>>>>>>>> 原始网格
colors = np.arange(len(XX1.ravel()))

fig, ax = plt.subplots(figsize = (5,5))
# 绘制原始网格
plt.plot(XX1, XX2,color = '#0070C0', lw = 0.25)
plt.plot(XX1.T, XX2.T,color = '#0070C0', lw = 0.25)
plt.scatter(XX1.ravel(), XX2.ravel(), c = colors, s = 10, cmap = 'rainbow', zorder=1e3)

plt.axis('scaled')
ax.set_xlim([-15, 15])
ax.set_ylim([-15, 15])
ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

#>>>>>>>>>>>> 平移
t = np.array([[4.5, -1.5]])
Z = X + t;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))
fig_name = '平移'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 旋转
# 绕原点，逆时针旋转30
theta = 30/180*np.pi
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
Z = X@R.T;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '逆时针旋转30度'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

# 先平移，再旋转
Z = (X + t)@R.T;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))
fig_name = '先平移，再逆时针旋转30度'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

# 先旋转，再平移
Z = X@R.T + t;

ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先逆时针旋转30度，再平移'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 等比例放大
S = np.array([[2, 0],
              [0, 2]])

Z = X@S;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '等比例放大'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 等比例缩小
S = np.array([[0.8, 0],
              [0,   0.8]])
Z = X@S;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '等比例缩小'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 非等比例缩放
S = np.array([[2, 0],
              [0, 1.5]])
Z = X@S;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '非等比例缩放'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 先缩放，再旋转
Z = X@S@R;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先缩放，再旋转'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 先旋转，再放大
Z = X@R@S;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先旋转，再缩放'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 横轴镜像
M = np.array([[1, 0],
              [0, -1]])
# 绕原点，逆时针旋转30
theta = 30/180*np.pi
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
S = np.array([[2, 0],
              [0, 1.5]])

# 先旋转，再放大，最后横轴镜像
Z = X@R@S@M;

ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先旋转，再放大，最后横轴镜像'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 纵轴镜像
M = np.array([[-1, 0],
              [0, 1]])

# 先旋转，再放大，最后纵轴镜像
Z = X@R@S@M;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先旋转，再放大，最后纵轴镜像'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 向横轴投影
P = np.array([[1, 0],
              [0, 0]])
Z = X@P;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '向横轴投影'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 向纵轴投影
P = np.array([[0, 0],
              [0, 1]])
Z = X@P;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '向纵轴投影'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 向特定过原点直线投影
theta = 30/180 * np.pi
# 过原点，和横轴夹角30度直线

v = np.array([[np.cos(theta)],
              [np.sin(theta)]])
Z = X@v@v.T;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '向特定过原点直线投影'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 沿横轴剪切
T = np.array([[1, 0],
              [1.5, 1]])
Z = X@T;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '沿横轴剪切'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 沿纵轴剪切
T = np.array([[1, 1.5],
              [0, 1]])
Z = X@T;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '沿纵轴剪切'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 绘制网格 BK_2_Ch08_10, 非线性变换
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import matplotlib

# colormap = cm.get_cmap("rainbow")
colormap = matplotlib.colormaps["rainbow"]
def plot_grid(xmin: float, xmax: float, ymin: float, ymax: float, n_lines: int, line_points: int, map_func, ax):
    lines = []
    # 水平线
    for y in np.linspace(ymin, ymax, n_lines):
        lines.append([map_func(x, y) for x in np.linspace(xmin, xmax, line_points)])
    # 竖直线
    for x in np.linspace(xmin, xmax, n_lines):
        lines.append([map_func(x, y) for y in np.linspace(ymin, ymax, line_points)])

    # 绘制所有线条
    for i, line in enumerate(lines):
        p = i / (len(lines) - 1)
        xs, ys = zip(*line)
        # 利用颜色映射
        # ax.plot(xs, ys, color = colormap(p))
        ax.plot(xs, ys, color = 'gray', lw = 0.7, alpha = 0.3)
        ax.scatter(xs, ys, c = 'r', s = 2)
# 各种映射
def identity(x, y):
    return x, y

def rotate_scale(x, y):
    return x + y, x - y

def shear(x, y):
    return x, x + y

def exp(x, y):
    return math.exp(x), math.exp(y)

def complex_sq(x, y):
    c = complex(x, y) ** 2
    return (c.real, c.imag)

def sin_cos(x: float, y: float):
    return x + math.sin(y * 2) * 0.2, y + math.cos(x * 2) * 0.3

def vortex(x: float, y: float):
    dst = (x - 2) ** 2 + (y - 2) ** 2
    ang = math.atan2(y - 2, x - 2)
    return math.cos(ang - dst * 0.1) * dst, math.sin(ang - dst * 0.1) * dst

# 原图
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
xmin = 0
xmax = 5
ymin = 0
ymax = 5
n_lines = 20
line_points = 20
plot_grid(0, 5, 0, 5, 20, 20, identity, ax)
ax.axis('off')

fig = plt.figure(figsize=(8, 12))
ax = fig.add_subplot(3, 2, 1)
plot_grid(0, 5, 0, 5, 20, 20, rotate_scale, ax)
ax.axis('off')

ax = fig.add_subplot(3, 2, 2)
plot_grid(0, 5, 0, 5, 20, 20, shear, ax)
ax.axis('off')

ax = fig.add_subplot(3, 2, 3)
plot_grid(0, 5, 0, 5, 20, 20, exp, ax)
ax.axis('off')

ax = fig.add_subplot(3, 2, 4)
plot_grid(0, 5, 0, 5, 20, 20, complex_sq, ax)
ax.axis('off')

ax = fig.add_subplot(3, 2, 5)
plot_grid(0, 5, 0, 5, 20, 20, sin_cos, ax)
ax.axis('off')

ax = fig.add_subplot(3, 2, 6)
plot_grid(0, 5, 0, 5, 20, 20, vortex, ax)
ax.axis('off')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BK_2_Ch10_05 线性、非线性几何变换
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
# 这段代码绘制了一个包含黑色轮廓线的图形， 其中轮廓线表示了X和Y坐标与其最近整数的差的绝对值为1的区域。注意这种技巧画网格,参见Book2_Chap24
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

X_,Y_ = 1/X, 1/Y
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

























































































































































































































































































