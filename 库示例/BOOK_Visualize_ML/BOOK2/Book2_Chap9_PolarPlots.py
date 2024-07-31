#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:49:20 2024

@author: jack
Chapter 9 极坐标绘图 | Book 2《可视之美》

"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 极坐标线图
# 导入包
import numpy as np
import matplotlib.pyplot as plt

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


# 产生数据

theta_array = np.linspace(0, 2 * np.pi, 1000)
# 极角
r_array = 2 + np.sin(6 * theta_array)
# 极径

# 可视化
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_array, r_array)
# fig.savefig('Figures/极坐标线图.svg', format='svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  更多极坐标线图
# 导入包
import numpy as np
import matplotlib.pyplot as plt

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

## 正圆
theta_array = np.linspace(0,2*np.pi, 200)
r_array     = np.ones_like(theta_array)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')
ax.plot(theta_array, r_array)
ax.set_rmax(2)
ax.set_rmin(0)
ax.set_rlabel_position(22.5)  # Move radial labels away from plotted line
ax.grid(True)
ax.set_yticklabels([])
# fig.savefig('Figures/正圆.svg', format='svg')


# 阿基米德螺线
r_array = np.arange(0, 4, 0.01)
theta_array = 2 * np.pi * r_array

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')
ax.plot(theta_array, r_array)
ax.set_rmin(r_array.min())
ax.set_rlabel_position(22.5)
ax.grid(True)
ax.set_yticklabels([])
# fig.savefig('Figures/阿基米德螺线.svg', format='svg')



# 心形曲线
theta_array = np.linspace(0, 2*np.pi, 2000)
r_array     = 2 + 2*np.cos(theta_array)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')

ax.plot(theta_array, r_array)
ax.set_rmin(r_array.min())

ax.set_rlabel_position(22.5)
ax.grid(True)
ax.set_yticklabels([])
# fig.savefig('Figures/心形曲线.svg', format='svg')


# 椭圆
theta_array = np.linspace(0,2*np.pi, 2000)
r_array     = 1/(1 + 0.5*np.cos(theta_array))

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')

ax.plot(theta_array, r_array)
ax.set_rmin(0)

ax.set_rlabel_position(22.5)
ax.grid(True)
ax.set_yticklabels([])
# fig.savefig('Figures/椭圆.svg', format='svg')



# 玫瑰线
theta_array = np.linspace(0,2*np.pi, 2000)
r_array     = 2*np.sin(6*theta_array)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')

ax.plot(theta_array, r_array)
ax.set_rmin(r_array.min())

ax.set_rlabel_position(22.5)
ax.grid(True)
ax.set_yticklabels([])
# fig.savefig('Figures/玫瑰线.svg', format='svg')


# 玫瑰线，有理数
theta_array = np.linspace(0,7*np.pi, 2000)
r_array     = 2*np.sin(11/3*theta_array)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')

ax.plot(theta_array, r_array)
ax.set_rmin(r_array.min())

ax.set_rlabel_position(22.5)
ax.grid(True)
ax.set_yticklabels([])
# fig.savefig('Figures/玫瑰线，有理数.svg', format='svg')


# 双纽线
a = 2
t = np.linspace(0, 2*np.pi, 200)
sint = np.sin(t)
cost = np.cos(t)
theta = np.arctan2(sint*cost, cost)
r = a*np.abs(cost) / np.sqrt(1 + sint**2)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')

ax.plot(theta, r)
ax.set_rmin(r_array.min())

ax.set_rlabel_position(22.5)
ax.grid(True)
ax.set_yticklabels([])
# fig.savefig('Figures/双纽线.svg', format='svg')

# 蝴蝶翼
theta_array = np.linspace(0,2*np.pi, 2000)
r_array = 1 - np.cos(theta_array)*np.sin(3*theta_array)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')

ax.plot(theta_array, r_array)
ax.set_rlabel_position(22.5)
ax.set_rmin(r_array.min())
ax.grid(True)
ax.set_yticklabels([])
# fig.savefig('Figures/蝴蝶翼.svg', format='svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 极坐标散点图
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




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 扇形散点图
import numpy as np
import matplotlib.pyplot as plt

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

# 随机数数量
num = 100
r = 2 * np.random.rand(num)
theta = 2 * np.pi * np.random.rand(num)
area = 200 * r**2
# 散点面积
colors = theta

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
ax.set_thetamin(0)
ax.set_thetamax(180)
# fig.savefig('极坐标，扇形，1.svg', format='svg')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
ax.set_thetamin(90)
ax.set_thetamax(360)
# fig.savefig('极坐标，扇形，2.svg', format='svg')
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 极坐标柱状图

# 导入包
import numpy as np
import matplotlib.pyplot as plt

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 柱状图柱子个数
num = 50
# 360度均分
theta = np.linspace(0.0, 2 * np.pi, num, endpoint = False)

# 随机数代表极轴长度
radii = 10 * np.random.rand(num)

# 宽度也是随机数
width = np.pi / 4 * np.random.rand(num)
colors = plt.cm.hsv(radii / 10.)

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.bar(theta, radii, width = width, bottom = 0.0, color = colors, alpha = 0.5)
# fig.savefig('Figures/极坐标柱状图.svg', format='svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 雷达图
import numpy as np
import matplotlib.pyplot as plt
import os

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

categories = ['A', 'B', 'C', 'D', 'E']
categories = [*categories, categories[0]]
Group_1 = [4, 4, 5, 4, 3]
Group_2 = [5, 5, 4, 5, 2]
Group_3 = [3, 4, 5, 3, 5]
Group_1 = [*Group_1, Group_1[0]]
Group_2 = [*Group_2, Group_2[0]]
Group_3 = [*Group_3, Group_3[0]]


fig = plt.figure()
ax = fig.add_subplot(projection='polar')
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(Group_1))
ax.plot(label_loc, Group_1, color = 'b', label='Group 1')
ax.fill(label_loc, Group_1, color = 'b', alpha=0.1)

ax.plot(label_loc, Group_2, color = 'r', label='Group 2')
ax.fill(label_loc, Group_2, color = 'r', alpha=0.1)

ax.plot(label_loc, Group_3, color = 'g', label='Group 3')
ax.fill(label_loc, Group_3, color = 'g', alpha=0.1)


lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
plt.legend()
# fig.savefig('Figures/雷达图.svg', format='svg')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 极坐标等高线
import matplotlib.pyplot as plt
import numpy as np

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

theta_array = np.linspace(0, 2*np.pi, 1001)
r_array = np.linspace(0, 3, 1001)
tt, rr = np.meshgrid(theta_array, r_array)

ff = np.cos(tt) * np.sin(2*rr)

# theta-r平面
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot( )
ax.contourf(tt, rr, ff, cmap = 'RdYlBu_r', levels=10)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$r$')
# fig.savefig('theta-r平面等高线.svg', format='svg')


# 3D
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(tt, rr, ff, color = [0.8,0.8,0.8], rstride=20, cstride=20, linewidth = 0.75)
ax.contour(tt, rr, ff, cmap = 'hsv', levels=10)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$r$')
# fig.savefig('theta-r平面等高线.svg', format='svg')


fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.contourf(tt, rr, ff, cmap = 'RdYlBu_r', levels=10)
# fig.savefig('极坐标等高线.svg', format='svg')

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.contour(tt, rr, ff, cmap = 'RdYlBu_r', levels=10)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 极坐标中的生成艺术
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

def arc(r, angle_start, angle_arc):
    delta_radian = np.pi/720
    angle_array = np.arange(angle_start, angle_start + angle_arc, delta_radian)
    x_array = r * np.cos(angle_array)
    y_array = r * np.sin(angle_array)
    return x_array, y_array

# cmap = mpl.cm.get_cmap('RdYlBu')
cmap = mpl.colormaps.get_cmap('RdYlBu')
#>>>>>>>>>>>>>>>  1
num = 200
r_array = np.random.uniform(0, 1, num)
angle_1_array = np.random.uniform(0, 2*np.pi, num)
angle_2_array = np.random.uniform(0, 2*np.pi, num)
fig, ax = plt.subplots()
for r_idx, a_1_idx, a_2_idx in zip(r_array, angle_1_array, angle_2_array):
    x_array, y_array = arc(r_idx, a_1_idx, a_2_idx)
    plt.plot(x_array, y_array, color = cmap(r_idx), lw = 1)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
ax.set_aspect('equal')
# fig.savefig('Figures/随机弧.svg', format='svg')
plt.show()


#>>>>>>>>>>>>>>>  2
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






































































































































































































































































































































































































































































































































































































































