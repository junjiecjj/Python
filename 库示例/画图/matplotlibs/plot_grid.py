#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 12:52:12 2024

@author: jack
"""


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





































