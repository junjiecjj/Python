#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:32:47 2024

@author: jack

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
























































































































































































































































































































































































