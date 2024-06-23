#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:48:02 2024

@author: jack
"""


#%% 导入包
import numpy as np
import matplotlib.pyplot as plt

import os

# 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
    # os.makedirs("Figures")



# 随机数数量
num = 100
r = 2 * np.random.rand(num)
theta = 2 * np.pi * np.random.rand(num)
area = 200 * r**2
# 散点面积

colors = theta


## 1
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

# fig.savefig('Figures/极坐标散点图_1.svg', format='svg')
plt.show()


## 2
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

ax.set_rorigin(-2)
# 改变极轴
# set_rorigin 是用于极坐标图 (polar plot) 的一个方法，用于设置极坐标轴的原点位置。
# 默认情况下，极坐标轴的原点位置位于图形中心。
# set_rorigin 方法可以用来改变极坐标轴原点的位置，
# 它接受一个参数 value，用于设置极坐标轴的半径的起点位置。
# 例如，如果将 value 设置为负数，那么极坐标轴原点将会移动到图形中心的下方，
# 而如果将 value 设置为正数，那么极坐标轴原点将会移动到图形中心的上方。

# fig.savefig('Figures/极坐标散点图_2.svg', format='svg')
plt.show()






#%% 极坐标中的生成艺术
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

def arc(r, angle_start, angle_arc):
    delta_radian = np.pi/720
    angle_array = np.arange(angle_start,
                            angle_start + angle_arc,
                            delta_radian)

    x_array = r * np.cos(angle_array)
    y_array = r * np.sin(angle_array)

    return x_array, y_array

cmap = mpl.cm.get_cmap('RdYlBu')
num = 200

r_array = np.random.uniform(0, 1, num)
angle_1_array = np.random.uniform(0, 2*np.pi, num)
angle_2_array = np.random.uniform(0, 2*np.pi, num)

fig, ax = plt.subplots()

for r_idx, a_1_idx, a_2_idx in zip(r_array, angle_1_array, angle_2_array):

    x_array, y_array = arc(r_idx, a_1_idx, a_2_idx)
    plt.plot(x_array,
             y_array,
             color = cmap(r_idx),
             lw = 0.2)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
ax.set_aspect('equal')
# fig.savefig('Figures/随机弧.svg', format='svg')
plt.show()


## 2
num = 600
r_array = np.random.uniform(0, 4, num)
r_array = 1 - np.exp(-r_array**2)

angle_array = np.random.uniform(0,1,num) * 2 * np.pi
x_array = np.cos(angle_array) * r_array
y_array = np.sin(angle_array) * r_array
area_array = np.random.rand(num)**3 * 200


fig, ax = plt.subplots()

plt.scatter(x_array, y_array, s=area_array, alpha = 0.2)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
ax.set_aspect('equal')
# fig.savefig('Figures/边缘散点.svg', format='svg')
plt.show()





























































































































































































































































































































