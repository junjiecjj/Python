#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:08:49 2025

@author: jack
"""

#%% Bk2_Ch19_01 # 斐波那契数列
# 导入包
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

# 使用递归函数生成 Fibonacci 数列
def fibonacci(n):
    # 如果 n 小于或等于 1，它将直接返回 n
    if n <= 1:
        return n
    # 否则，它将调用两次自己
    # 并将 n-1 和 n-2 作为参数传递给它们
    else:
        return fibonacci(n-1) + fibonacci(n-2)

## 生成数列
num = 9
idx_array = np.arange(num)
# 从0开始的斐波那契数列
fibonacci_array = np.array([fibonacci(i) for i in range(num)])
# 从1开始的斐波那契数列
fibonacci_array_from_1 = np.array([fibonacci(i) for i in np.arange(1, num + 1)])

## 准备数据
# 正方形 x 坐标
x_array = 0 + fibonacci_array_from_1 * (-1) ** (np.floor(idx_array/2) + 2)
x_array = np.insert(x_array, 0, 0)
x_array = np.cumsum(x_array)
# 正方形 y 坐标
y_array = 0 + fibonacci_array_from_1 * (-1) ** (np.floor((1 + idx_array)/2) + 1)
y_array = np.insert(y_array, 0, 0)
y_array = np.cumsum(y_array)
# 正方形旋转角度，逆时针为正
rotation = (idx_array - 1) * 90
fig, ax = plt.subplots(figsize = (8,8))

colors = plt.cm.Blues_r(np.linspace(0,1,num + 1))
for i in idx_array:
    rotate_i = rotation[i]
    side_idx = fibonacci_array_from_1[i]
    rec_loc_idx = np.array([x_array[i], y_array[i]])
    rec = plt.Rectangle(rec_loc_idx,
                        width =side_idx,
                        height=side_idx,
                        facecolor=colors[i + 1],
                        edgecolor = 'k',
                        transform=Affine2D().rotate_deg_around( x_array[i], y_array[i], rotate_i)  + ax.transData)
    ax.add_patch(rec)
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='x', markersize=10)
ax.axis('off')
plt.show()



#%% Bk2_Ch19_02 # 斐波那契数列曲线
# 导入包
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D


# 使用递归函数生成 Fibonacci 数列
def fibonacci(n):
    # 如果 n 小于或等于 1，它将直接返回 n
    if n <= 1:
        return n
    # 否则，它将调用两次自己
    # 并将 n-1 和 n-2 作为参数传递给它们
    else:
        return fibonacci(n-1) + fibonacci(n-2)

## 产生数据
num = 11
idx_array = np.arange(num)

fibonacci_array = np.array([fibonacci(i) for i in range(num)])
fibonacci_array_from_1 = np.array([fibonacci(i) for i in np.arange(1, num + 1)])
x_array = 0 + fibonacci_array_from_1 * (-1) ** (np.floor(idx_array/2) + 2)
x_array = np.insert(x_array, 0, 0)
x_array = np.cumsum(x_array)
y_array = 0 + fibonacci_array_from_1 * (-1) ** (np.floor((1 + idx_array)/2) + 1)
y_array = np.insert(y_array, 0, 0)
y_array = np.cumsum(y_array)
# # 逆时针为正
rotation = (idx_array - 1) * 90
x_arc_center = fibonacci_array * (4*np.abs((idx_array - 1)/4 - np.floor((idx_array - 1)/4 + 2/4)) - 1)
# 这一句用到了三角波的变形
x_arc_center = np.cumsum(x_arc_center) + 1
x_arc_center = np.insert(x_arc_center, 0, 1)

triangle_wave = 4*np.abs((idx_array - 1)/4 - np.floor((idx_array - 1)/4 + 2/4)) - 1
triangle_wave = np.insert(triangle_wave, 0, 0)
y_arc_center = fibonacci_array * triangle_wave[:-1]
y_arc_center = np.cumsum(y_arc_center) + 0
y_arc_center = np.insert(y_arc_center, 0, 0)
# 自定义函数，绘制45度弧形
def plot_arc(x_i, y_i, r_i, rotation_i):
    rotation_i = rotation_i/180 * np.pi
    angle_array = np.linspace(rotation_i, rotation_i + np.pi/2,  101, endpoint = True)
    x_arc_array = r_i * np.cos(angle_array) + x_i
    y_arc_array = r_i * np.sin(angle_array) + y_i
    plt.plot(x_arc_array, y_arc_array, c = 'k')
fig, ax = plt.subplots(figsize = (8,8))

colors = plt.cm.RdYlBu(np.linspace(0,1,num + 1))

for i in idx_array:
    rotate_i = rotation[i]
    side_idx = fibonacci_array_from_1[i]
    rec_loc_idx = np.array([x_array[i], y_array[i]])

    rec = plt.Rectangle(rec_loc_idx,
                        width =side_idx,
                        height=side_idx,
                        facecolor=colors[i + 1],
                        edgecolor = 'w',
                        transform=Affine2D().rotate_deg_around(
                           x_array[i], y_array[i], rotate_i)
                        + ax.transData)
    ax.add_patch(rec)
    # 绘制弧形
    plot_arc(x_arc_center[i], y_arc_center[i],
             fibonacci_array_from_1[i], 180 + i * 90)

ax.set_aspect('equal')
plt.axis('off')
plt.show()


#%% Bk2_Ch19_03 # 可视化巴都万数列
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

## 自定义函数
# 给定等边三角形的两个顶点坐标，计算第三个顶点坐标
def get_vertex(p1, p2):
    theta = 60
    theta = theta * np.pi/180
    # 旋转60度
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    v = p2 - p1
    p3 = p1 + R @ v
    return p3

# 绘制等边三角形
def draw_triangle(triangle, colour):
    v0, v1, v2 = triangle
    poly = Polygon((v0, v1, v2),
                   facecolor=colour,
                   edgecolor='k',
                   linewidth=2,
                   joinstyle='bevel')
    ax.add_patch(poly)
## 产生数列
num = 12
P = [1, 1, 1, 2]
for i in range(4, num):
    P.append(P[i-2] + P[i-3])
print(P)

## 等边三角形顶点位置
V0 = np.array([0, 0])
V1 = np.array([1, 0])
V2 = get_vertex(V0, V1)
V3 = get_vertex(V2, V1)
V4 = get_vertex(V2, V3)
V5 = get_vertex(V0, V4)
V6 = get_vertex(V0, V5)
V7 = get_vertex(V1, V6)

# 创建空数组，用来保存三角形顶点坐标
T = np.empty((num, 3, 2))
# num为三角形数量，3是顶点数量，2代表横纵坐标 (2维平面)

# 前六个等边三角形的顶点坐标
T[:6] = [(V0, V1, V2),
         (V2, V1, V3),
         (V2, V3, V4),
         (V0, V4, V5),
         (V0, V5, V6),
         (V1, V6, V7)]
# 从第7个三角形开始存在如下规律
for i in range(6, num):
    V0, V1 = T[i-4][1], T[i-1][2]
    T[i] = (V0, V1, get_vertex(V0, V1))
## 可视化
fig, ax = plt.subplots(figsize = (8,8))
colors = plt.cm.Blues_r(np.linspace(0,1,len(T) + 1))
for i in range(len(T)):
    draw_triangle(T[i], colors[i])
    ax.annotate(str(P[i]), np.mean(T[i], axis=0), va='center', ha='center')
ax.axis('equal')
xmax = np.max(T[:,:,0])
xmin = np.min(T[:,:,0])
ymax = np.max(T[:,:,1])
ymin = np.min(T[:,:,1])
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.axis('off')
plt.show()


#%% Bk2_Ch19_04 # 雷卡曼数列
import numpy as np
import matplotlib.pyplot as plt

# 自定义函数生成雷卡曼数列
def recaman(n):
    # Create an array to store terms
    arr = [0] * n

    # First term of the sequence
    # is always 0
    arr[0] = 0
    print(arr[0], end=", ")

    # Fill remaining terms using
    # recursive formula.
    for i in range(1, n):
        curr = arr[i-1] - i
        for j in range(0, i):
            # If arr[i-1] - i is
            # negative or already
            # exists.
            if ((arr[j] == curr) or curr < 0):
                curr = arr[i-1] + i
                break
        arr[i] = curr
        print(arr[i], end=", ")
    return arr
# 前1000项
num_iterations = 1000
recamans_sq = recaman(num_iterations)
fig, ax = plt.subplots() # facecolor='k'
ax.plot(np.arange(1, len(recamans_sq) + 1), recamans_sq)
plt.grid()
# fig.savefig('Figures/雷卡曼数列，1000.svg', format='svg')
plt.show()

import matplotlib
# 绘制圆弧
def add_to_plot(last_a, a, n):
    # 计算圆心、半径
    c, r = (a + last_a) / 2, abs(a - last_a) / 2
    y = np.linspace(-r, r, 1000)
    # 绘制半圆弧纵轴坐标
    x = np.sqrt(r**2 - y**2) * (-1)**n
    # 绘制半圆弧横轴坐标，(-1)**n 控制交替左右
    color = matplotlib.colormaps.get_cmap('hsv')(n/max_terms)
    # 颜色映射

    # 绘制半圆弧
    ax.plot(x, y+c, c=color, lw=1)

fig, ax = plt.subplots()
ax.axis('equal')
ax.axis('off')
cm = plt.cm.get_cmap('hsv')

seen = []
n = a = 0
lasta = None
max_terms = 60 # 前60项
while n < max_terms:
    last_a = a
    if (b := a - n) > 0 and b not in seen:
        a = b
    else:
        a = a + n
    seen = seen + [a]
    n += 1
    # 图中添加圆弧
    add_to_plot(last_a, a, n)
ax.set_yticks(np.arange(np.array(seen).max() + 1, step = 5))
# fig.savefig('Figures/雷卡曼数列，60项.svg', format='svg')
plt.show()


#%% Bk2_Ch19_05 # 数列求和极限，第一个例子

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

fig = plt.figure()
ax = fig.add_subplot(111)
num = 9

colors = plt.cm.Blues(np.linspace(0,1,num + 1))
for idx in np.arange(num):
    x_idx = (1/2)**(idx + 1)
    y_idx = (1/2)**(idx + 1)
    width_idx = (1/2)**(idx + 1)
    height_idx = (1/2)**(idx + 1)

    rect_idx = matplotlib.patches.Rectangle((x_idx,y_idx),
                                            width_idx, height_idx,
                                            facecolor=colors[idx + 1],
                                            edgecolor = 'k')
    ax.add_patch(rect_idx)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xticks(np.arange(0, 1.25, 0.25))
ax.set_yticks(np.arange(0, 1.25, 0.25))
ax.set_aspect('equal', adjustable='box')
# fig.savefig('Figures/极限值为三分之一.svg', format='svg')
plt.show()


#%% Bk2_Ch19_06 # 数列求和极限，第二个例子

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


fig = plt.figure()
ax = fig.add_subplot(111)
num = 15
colors = plt.cm.Blues(np.linspace(0,1,num + 1))
x_idx_0 = 0
y_idx_0 = 0
for idx in np.arange(num):
    x_idx = 1 - (1/2)**np.floor((idx + 1)/2)
    y_idx = 1 - (1/2)**np.floor(idx/2)
    width_idx = (1/2)**(np.ceil((idx + 1)/2))
    height_idx = (1/2)**np.floor((idx + 1)/2)

    ax.plot((x_idx_0, x_idx), (y_idx_0, y_idx), 'r')

    x_idx_0 = x_idx
    y_idx_0 = y_idx
    rect_idx = matplotlib.patches.Rectangle((x_idx,y_idx),
                                            width_idx, height_idx,
                                            facecolor=colors[idx + 1],
                                            edgecolor = 'k')
    ax.add_patch(rect_idx)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xticks(np.arange(0, 1.25, 0.25))
ax.set_yticks(np.arange(0, 1.25, 0.25))
ax.set_aspect('equal', adjustable='box')
# fig.savefig('Figures/日取其半，万世不竭.svg', format='svg')
plt.show()



















