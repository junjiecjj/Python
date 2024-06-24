#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:32:47 2024

@author: jack
 Chapter 12 平面几何图形 | Book 2《可视之美》


# ◄ matplotlib.patches.Arc() 绘制弧线
# ◄ matplotlib.patches.Arrow() 绘制箭头
# ◄ matplotlib.patches.Circle() 绘制正圆
# ◄ matplotlib.patches.Ellipse() 绘制椭圆
# ◄ matplotlib.patches.FancyBboxPatch() 绘制 Fancy 矩形框
# ◄ matplotlib.patches.Polygon() 绘制多边形
# ◄ matplotlib.patches.Rectangle() 绘制长方形
# ◄ matplotlib.patches.RegularPolygon() 绘制正多边形
# ◄ matplotlib.pyplot.cm 提供各种预定义色谱方案，比如 matplotlib.pyplot.cm.rainbow
# ◄ matplotlib.pyplot.contour()绘制平面等高线
# ◄ matplotlib.pyplot.contourf() 绘制填充等高线图
# ◄ numpy.cos() 计算余弦值
# ◄ numpy.diag() 如果 A 为方阵， numpy.diag(A) 函数提取对角线元素，以向量形式输入结果；如果 a 为向量，
# numpy.diag(a) 函数将向量展开成方阵，方阵对角线元素为 a 向量元素
# ◄ numpy.dot() 计算向量标量积。值得注意的是，如果输入为一维数组， numpy.dot() 输出结果为标量积；如果输入为
# 矩阵， numpy.dot() 输出结果为矩阵乘积，相当于矩阵运算符@
# ◄ numpy.linalg.inv() 矩阵求逆
# ◄ numpy.linalg.norm() 计算范数
# ◄ numpy.meshgrid() 创建网格化数据
# ◄ numpy.sin() 计算正弦值
# ◄ numpy.sqrt() 计算平方根
# ◄ matplotlib.patches.Rectangle() 是 Matplotlib 中的一个图形对象，用于绘制矩形形状
# ◄ matplotlib.pyplot.axhspan() 函数用于在水平方向创建一个跨越指定 y 值范围的色块
# ◄ matplotlib.pyplot.axvspan() 函数用于在垂直方向创建一个跨越指定 x 值范围的色块
# ◄ matplotlib.pyplot.fill() 函数用于绘制多边形，并在其中填充颜色，创建一个封闭区域的图形效果
# ◄ matplotlib.pyplot.fill_between() 函数用于在两条曲线之间填充颜色，创建一个区域的图形效果
# ◄ matplotlib.pyplot.fill_betweenx()函数用于在两条垂直于 x 轴的水平线之间填充颜色，创建一个区域的图形效
# 果
# ◄ numpy.linspace() 在指定的间隔内,返回固定步长的数据



"""

#%% 使用 patches 绘制平面几何形状


import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.path as mpath

# Prepare the data for the PathPatch below.
Path = mpath.Path
codes, verts = zip(*[
    (Path.MOVETO, [0.018, -0.11]),
    (Path.CURVE4, [-0.031, -0.051]),
    (Path.CURVE4, [-0.115, 0.073]),
    (Path.CURVE4, [-0.03, 0.073]),
    (Path.LINETO, [-0.011, 0.039]),
    (Path.CURVE4, [0.043, 0.121]),
    (Path.CURVE4, [0.075, -0.005]),
    (Path.CURVE4, [0.035, -0.027]),
    (Path.CLOSEPOLY, [0.018, -0.11])])

artists = [
    mpatches.Circle((0, 0), 0.1, ec="none"),
    mpatches.Rectangle((-0.025, -0.05), 0.05, 0.1, ec="none"),
    mpatches.Wedge((0, 0), 0.1, 30, 270, ec="none"),
    mpatches.RegularPolygon((0, 0), 5, radius=0.1),
    mpatches.Ellipse((0, 0), 0.2, 0.1),
    mpatches.Arrow(-0.05, -0.05, 0.1, 0.1, width=0.1),
    mpatches.PathPatch(mpath.Path(verts, codes), ec="none"),
    mpatches.FancyBboxPatch((-0.025, -0.05), 0.05, 0.1, ec="none", boxstyle=mpatches.BoxStyle("Round", pad=0.02)),
    mlines.Line2D([-0.06, 0.0, 0.1], [0.05, -0.05, 0.05], lw=5),
]

axs = plt.figure(figsize=(12, 12)).subplots(3, 3)
for i, (ax, artist) in enumerate(zip(axs.flat, artists)):
    artist.set(color = mpl.cm.get_cmap('hsv')(i / len(artists)))
    ax.add_artist(artist)
    ax.set(title=type(artist).__name__, aspect=1, xlim=(-.2, .2), ylim=(-.2, .2))
    ax.set_axis_off()
plt.show()





#%% 利用patches绘制正圆，以及外切、内接正多边形

# 导入包
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
import numpy as np

# 可视化
fig, axs = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 5))

for num_vertices, ax in zip([4,5,6,8], axs.ravel()):

    hexagon_inner = RegularPolygon((0,0), numVertices=num_vertices,  radius=1, alpha=0.2, edgecolor='k')
    ax.add_patch(hexagon_inner) # 绘制正圆内接多边形

    hexagon_outer = RegularPolygon((0,0), numVertices=num_vertices,
                                   radius=1/np.cos(np.pi/num_vertices),
                                   alpha=0.2, edgecolor='k')

    # 绘制正圆外切多边形
    ax.add_patch(hexagon_outer)

    # 绘制正圆
    circle = Circle((0,0), radius=1, facecolor = 'none', edgecolor='k')
    ax.add_patch(circle)

    ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
    ax.set_aspect('equal', adjustable='box'); ax.axis('off')
plt.show()



#%%  正圆的生成艺术
import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(200)
delta_angle = 5 # degrees

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    width = 0.05 + i * 0.05
    point_of_rotation = np.array([-width/2, -width/2])
    rec = plt.Circle(point_of_rotation, radius=width,
                        fill = False, edgecolor = colors[i],
                        transform = Affine2D().rotate_deg_around(0, 0, deg) + ax.transData)
    ax.add_patch(rec)

plt.axis('off')

# fig.savefig('Figures/旋转正圆_A.svg', format='svg')
plt.show()


## 2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(100)
delta_angle = 3.6 # degrees

colors = plt.cm.hsv(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    width = 2
    point_of_rotation = np.array([0, -width])
    rec = plt.Circle(point_of_rotation, radius=width,
                        fill = False, edgecolor = colors[i],
                        transform=Affine2D().rotate_deg_around(0,0, deg)+ax.transData)
    ax.add_patch(rec)

plt.axis('off')
# fig.savefig('Figures/旋转正圆_B.svg', format='svg')
plt.show()

## 3
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.transforms import Affine2D
fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='b', marker='o', markersize=10)

range_array = np.arange(36)
delta_angle = 10 # degrees

colors = plt.cm.hsv(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    width = 2
    point_of_rotation = np.array([0, -width])
    rec = plt.Circle(point_of_rotation, radius=width,
                     fill = True, edgecolor = 'b',
                     alpha = 0.1,
                     transform=Affine2D().rotate_deg_around(0,0, deg)+ax.transData)
    ax.add_patch(rec)

plt.axis('off')
plt.show()

#%%  可视化线性、非线性变换
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os


# 绘制网格
# p = plt.rcParams
# p["font.sans-serif"] = ["Roboto"]
# p["font.weight"] = "light"
# p["ytick.minor.visible"] = True
# p["xtick.minor.visible"] = True
# p["axes.grid"] = True
# p["grid.color"] = "0.5"
# p["grid.linewidth"] = 0.5

colormap = cm.get_cmap("rainbow")

def plot_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n_lines: int,
    line_points: int,
    map_func,
):
    lines = []
    # 水平线
    for y in np.linspace(ymin, ymax, n_lines):
        lines.append([map_func(x, y) for x in np.linspace(xmin, xmax, line_points)])
    print(f"1 {len(lines)}, {len(lines[0])}")
    # 竖直线
    for x in np.linspace(xmin, xmax, n_lines):
        lines.append([map_func(x, y) for y in np.linspace(ymin, ymax, line_points)])
    print(f"2 {len(lines)}, {len(lines[0])}")
    # 绘制所有线条
    for i, line in enumerate(lines):
        p = i / (len(lines) - 1)
        xs, ys = zip(*line)
        # 利用颜色映射
        plt.plot(xs, ys, color=colormap(p))

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

## 1
fig = plt.figure(figsize=(4, 4))

# 原图
ax = fig.add_subplot(111)
plot_grid(0, 5, 0, 5, 20, 20, identity)

ax.axis('off')
# fig.savefig('Figures/原始网格.svg', format='svg')
plt.show()

## 2
fig = plt.figure(figsize=(8, 12))

ax = fig.add_subplot(3, 2, 1)
plot_grid(0, 5, 0, 5, 20, 20, rotate_scale)
ax.axis('off')

ax = fig.add_subplot(3, 2, 2)
plot_grid(0, 5, 0, 5, 20, 20, shear)
ax.axis('off')

ax = fig.add_subplot(3, 2, 3)
plot_grid(0, 5, 0, 5, 20, 20, exp)
ax.axis('off')

ax = fig.add_subplot(3, 2, 4)
plot_grid(0, 5, 0, 5, 20, 20, complex_sq)
ax.axis('off')

ax = fig.add_subplot(3, 2, 5)
plot_grid(0, 5, 0, 5, 20, 20, sin_cos)
ax.axis('off')

ax = fig.add_subplot(3, 2, 6)
plot_grid(0, 5, 0, 5, 20, 20, vortex)
ax.axis('off')

# fig.savefig('Figures/线性、非线性变换.svg', format='svg')
plt.show()











#%%  三角形的生成艺术，两组
import os

# 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
    # os.makedirs("Figures")

import matplotlib.pyplot as plt
import numpy as np


theta = 15 # rotation angle
t = [0, 0]
r = 1
points_x = [0, np.sqrt(3)/2 * r, -np.sqrt(3)/2 * r, 0]
points_y = [r, -1/2 * r, -1/2 * r, r]

X = np.column_stack([points_x,points_y])
theta = np.deg2rad(theta)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

X = X @ R + t
# X


def eq_l_tri(ax, r, theta, t, color = 'b', fill = False):
    points_x = [0, np.sqrt(3)/2 * r, -np.sqrt(3)/2 * r, 0]
    points_y = [r, -1/2 * r, -1/2 * r, r]

    X = np.column_stack([points_x,points_y])
    theta = np.deg2rad(theta)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    X = X @ R.T
    ax.plot(X[:,0], X[:,1], color = color)
    if fill:
        plt.fill(X[:,0], X[:,1], color = color, alpha = 0.1)
## 1
fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
eq_l_tri(ax, r, 10, t)
plt.show()

## 2
fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(20)
delta_angle = 2 # degrees

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    r = 0.05 + i * 0.05
    eq_l_tri(ax, r, deg, (0,0), colors[i])

plt.axis('off')
# fig.savefig('Figures/旋转三角形_A.svg', format='svg')
plt.show()


## 3
fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(20)
delta_angle = 5 # degrees

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    r = 0.05 + i * 0.05
    eq_l_tri(ax, r, deg, (0,0), colors[i])

plt.axis('off')
# fig.savefig('Figures/旋转三角形_B.svg', format='svg')
plt.show()






#%%  椭圆的生成艺术，两组
import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Ellipse



## 1
fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(200)
delta_angle = 10 # degrees

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    width = 0.05 + i * 0.05
    point_of_rotation = np.array([-width/2, -width/2])
    rec = Ellipse(point_of_rotation, width=width, height = width * 1.5,
                        fill = False, edgecolor = colors[i],
                        transform=Affine2D().rotate_deg_around(0,0, deg)+ax.transData)
    ax.add_patch(rec)

plt.axis('off')
# fig.savefig('Figures/旋转椭圆_A.svg', format='svg')
plt.show()



## 2
fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(200)
delta_angle = 60 # degrees

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    width = 0.05 + i * 0.05
    point_of_rotation = np.array([-width/2, -width/2])
    rec = Ellipse(point_of_rotation, width=width, height = width * 1.5,
                        fill = False, edgecolor = colors[i],
                        transform=Affine2D().rotate_deg_around(0,0, deg)+ax.transData)
    ax.add_patch(rec)

plt.axis('off')
# fig.savefig('Figures/旋转椭圆_B.svg', format='svg')
plt.show()





#%% 正方形的生成艺术，两组


## 1
import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D


fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(100)
delta_angle = 2.5 # degrees

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    width = 0.05 + i * 0.05
    point_of_rotation = np.array([-width/2, -width/2])
    rec = plt.Rectangle(point_of_rotation, width=width, height=width,
                        fill = False, edgecolor = colors[i],
                        transform=Affine2D().rotate_deg_around(0,0, deg)+ax.transData)
    ax.add_patch(rec)

plt.axis('off')
# fig.savefig('Figures/旋转正方形_A.svg', format='svg')
plt.show()


## 2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D


fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(200)
delta_angle = 5 # degrees

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    width = 0.05 + i * 0.05
    point_of_rotation = np.array([-width/2, -width/2])
    rec = plt.Rectangle((0,0), width=width, height=width,
                        fill = False, edgecolor = colors[i],
                        transform=Affine2D().rotate_deg_around(0,0, deg)+ax.transData)
    ax.add_patch(rec)

plt.axis('off')
# fig.savefig('Figures/旋转正方形_B.svg', format='svg')
plt.show()





























































































































































































































































































































































































