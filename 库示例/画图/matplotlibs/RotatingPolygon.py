#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:12:49 2024

@author: jack
"""


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

















