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
# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.labelspacing'] = 0.2


# 产生数据
theta_array = np.linspace(0, 2 * np.pi, 1000)
# 极角
r_array = 2 + np.sin(6 * theta_array)
# 极径

# 可视化
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_array, r_array)
plt.show()
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  更多极坐标线图
# 导入包
import numpy as np
import matplotlib.pyplot as plt

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
plt.show()
plt.close()

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
plt.show()
plt.close()

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
plt.show()
plt.close()

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
plt.show()
plt.close()


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
plt.show()
plt.close()

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
plt.show()
plt.close()

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
plt.show()
plt.close()

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
plt.show()
plt.close()

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
plt.show()
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 扇形散点图
import numpy as np
import matplotlib.pyplot as plt

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

plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
ax.set_thetamin(90)
ax.set_thetamax(360)
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 极坐标柱状图
# 导入包
import numpy as np
import matplotlib.pyplot as plt

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 雷达图
import numpy as np
import matplotlib.pyplot as plt

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


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 极坐标等高线
import matplotlib.pyplot as plt
import numpy as np

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

# 3D
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(tt, rr, ff, color = [0.8,0.8,0.8], rstride=20, cstride=20, linewidth = 0.75)
ax.contour(tt, rr, ff, cmap = 'hsv', levels=10)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$r$')

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.contourf(tt, rr, ff, cmap = 'RdYlBu_r', levels=10)

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.contour(tt, rr, ff, cmap = 'RdYlBu_r', levels=10)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 极坐标中的生成艺术
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
plt.show()






































































































































































































































































































































































































































































































































































































































