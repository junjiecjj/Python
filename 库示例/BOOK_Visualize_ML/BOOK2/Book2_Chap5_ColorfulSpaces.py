#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 16:22:49 2024

@author: jack
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 平面散点展示RGB色彩空间

# 导入包
import matplotlib.pyplot as plt
import numpy as np
import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

######################################### 1. 蓝绿渐变
num_points = 21
# 定义每个维度上散点个数
# 定义绿色数组
Green_array = np.linspace(0, 1, num_points)

# 定义蓝色数组
Blue_array = np.linspace(0, 1, num_points)

# numpy.meshgrid() 常用来生成二维数据网格
GG, BB = np.meshgrid(Green_array, Blue_array) # (21, 21)

RR = np.zeros_like(GG)
# RR = np.ones_like(GG)
# 定义红色数组，红色为 0
# numpy.zeros_like(A) 生成和A形状相同的全0数组
# 类似函数还有 numpy.empty_like(), numpy.ones_like(), numpy.full_like()

RGB_colors = np.vstack([RR.ravel(), GG.ravel(), BB.ravel()]).T # (441, 3)
# ravel将多维数组降维成一维
# numpy.vstack() 返回竖直堆叠后的数组
# 类似函数还有 numpy.hstack(), numpy.column_stack(), numpy.row_stack()

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(GG, BB, c= RGB_colors, s=12)
# 绘制平面散点图 c为RGB颜色 s为散点大小
# 设置横纵轴标签
ax.set_xlabel('Green')
ax.set_ylabel('Blue')

# 设置横纵轴取值范围
ax.set_xlim(-0.05,1.05)
ax.set_ylim(-0.05,1.05)
# 设置横纵轴刻度
ax.set_xticks(np.linspace(0,1,6))
ax.set_yticks(np.linspace(0,1,6))
# 增加刻度网格，颜色为浅灰 (0.8,0.8,0.8)
plt.grid(color = (0.8,0.8,0.8))
# 将刻度网格置底
ax.set_axisbelow(True)

# 删除上下左右图脊
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# fig.savefig('Figures/蓝绿平面.svg', format='svg')
RGB_colors

######################################### 2. 蓝绿渐变 + 红 (1, 0, 0)
# 请大家自己补充注释
RR = np.ones_like(GG)

RGB_colors = np.vstack([RR.ravel(), GG.ravel(), BB.ravel()]).T

fig, ax = plt.subplots(figsize = (5,5))
ax.scatter(GG, BB, c = RGB_colors, s = 12, )

ax.set_xlabel('Green + (1, 0, 0)')
ax.set_ylabel('Blue + (1, 0, 0)')

ax.set_xlim(-0.05,1.05)
ax.set_ylim(-0.05,1.05)

ax.set_xticks(np.linspace(0,1,6))
ax.set_yticks(np.linspace(0,1,6))

plt.grid(color = (0.8,0.8,0.8))

ax.set_axisbelow(True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# fig.savefig('Figures/蓝绿_red_1.svg', format='svg')

#########################################  3. 红蓝渐变
# 请大家自己补充注释

XX, YY = np.meshgrid(np.linspace(0, 1, num_points),np.linspace(0, 1, num_points))

All_0s = XX*0

RGB_colors = np.vstack([XX.ravel(), All_0s.ravel(), YY.ravel()]).T

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(XX, YY, c = RGB_colors, s=12)
ax.set_xlim(-0.05,1.05)
ax.set_ylim(-0.05,1.05)

ax.set_xticks(np.linspace(0,1,6))
ax.set_yticks(np.linspace(0,1,6))

ax.set_xlabel('Red')
ax.set_ylabel('Blue')

plt.grid(color = (0.8,0.8,0.8))
ax.set_axisbelow(True)

plt.gca().invert_xaxis()
# 翻转横轴

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# fig.savefig('Figures/红蓝.svg', format='svg')


######################################### 4. 红蓝渐变 + 绿 (0, 1, 0)
# 请大家自己补充注释

XX,YY = np.meshgrid(np.linspace(0, 1, num_points),np.linspace(0, 1, num_points))

All_1s = XX*0 + 1

RGB_colors = np.vstack([XX.ravel(), All_1s.ravel(), YY.ravel()]).T

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(XX,YY, c=RGB_colors, s=12)
ax.set_xlim(-0.05,1.05)
ax.set_ylim(-0.05,1.05)

ax.set_xticks(np.linspace(0,1,6))
ax.set_yticks(np.linspace(0,1,6))

ax.set_xlabel('Red')
ax.set_ylabel('Blue')

plt.grid(color = (0.8,0.8,0.8))
ax.set_axisbelow(True)

plt.gca().invert_xaxis()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# fig.savefig('Figures/红蓝_green_1.svg', format='svg')

#########################################  5. 红绿渐变
RGB_colors = np.vstack([XX.ravel(), YY.ravel(), All_0s.ravel()]).T

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(XX,YY, c=RGB_colors, s=12)
ax.set_xlim(-0.05,1.05)
ax.set_ylim(-0.05,1.05)

ax.set_xticks(np.linspace(0,1,6))
ax.set_yticks(np.linspace(0,1,6))

ax.set_xlabel('Red')
ax.set_ylabel('Green')

plt.grid(color = (0.8,0.8,0.8))
ax.set_axisbelow(True)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# 删除上下左右图脊

# fig.savefig('Figures/红绿.svg', format='svg')


######################################### 6. 红绿渐变 + 蓝 (0, 0, 1)
RGB_colors = np.vstack([XX.ravel(), YY.ravel(), All_1s.ravel()]).T

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(XX,YY, c=RGB_colors, s=12)
ax.set_xlim(-0.05,1.05)
ax.set_ylim(-0.05,1.05)

ax.set_xticks(np.linspace(0,1,6))
ax.set_yticks(np.linspace(0,1,6))

ax.set_xlabel('Red')
ax.set_ylabel('Green')

plt.grid(color = (0.8,0.8,0.8))
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

# fig.savefig('Figures/红绿_blue_1.svg', format='svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 三维空间散点展示RGB色彩空间

# 导入包
import numpy as np
import matplotlib.pyplot as plt

import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 定义函数
# 自定义函数，生成三维网格数据
def color_cubic(num):
    x1 = np.linspace(0,1,num)
    x2 = x1
    x3 = x1

    # 生成三维数据网格
    xx1, xx2, xx3 = np.meshgrid(x1,x2,x3)

    # 将三维数组展成一维
    x1_ = xx1.ravel()
    x2_ = xx2.ravel()
    x3_ = xx3.ravel()

    # 将一维数组作为列堆叠成二维数组
    colors_all     = np.column_stack([x1_, x2_, x3_])
    # 利用面具 (mask) 做筛选 仅仅保留立方体三个朝外的立面： 颜色相对较为鲜亮
    colors_bright  = colors_all[np.any(colors_all == 1, axis=1)]
    # 仅仅保留立方体三个朝内的立面： 颜色相对较为暗沉
    colors_dark    = colors_all[np.any(colors_all == 0, axis=1)]
    return colors_all, colors_bright, colors_dark

#>>>>>>>>>>>>>>>>>>>>>>>  完全填充立方体，最艳丽的三个立面
colors_all, colors_bright, colors_dark = color_cubic(101)
# 使用自定义函数生成RGB颜色色号，单一维度上有101个点

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
# 增加三维轴

# 用三维散点图绘可视化立方体外侧三个鲜亮的侧面
ax.scatter(colors_bright[:,0], # x 坐标
           colors_bright[:,1], # y 坐标
           colors_bright[:,2], # z 坐标
           c = colors_bright,  # 颜色色号
           s = 1,              # 散点大小
           alpha = 1)          # 透明度，1 代表完全不透

# 设定横纵轴标签
ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{x_3}$')

## 设定观察视角
# ax.view_init(azim=38, elev=34)
# ax.view_init(azim=60, elev=30)

## 设定 x、y、z 取值范围
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# ax.set_zlim(0,1)

## 不显示刻度
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
## 不显示轴背景
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False

## 图脊设为白色
# ax.xaxis.pane.set_edgecolor('w')
# ax.yaxis.pane.set_edgecolor('w')
# ax.zaxis.pane.set_edgecolor('w')

## 不显示网格
# ax.grid(False)

## 正交投影
ax.set_proj_type('ortho')

# 等比例成像
ax.set_box_aspect(aspect = (1,1,1))

# # Transparent spines
# ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# # Transparent panes
# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

## fig.savefig('Figures/完全填充立方体，最艳丽的三个立面.svg', format='svg')
## fig.savefig('Figures/完全填充立方体，最艳丽的三个立面.png', format='png')

#>>>>>>>>>>>>>>>>>>>>>>> 整个色彩空间采样，散点稀疏
colors_all, colors_bright, colors_dark = color_cubic(21)

# 定义三根线的坐标
line1_x = [1,1]
line1_y = [1,1]
line1_z = [1,0]

line2_x = [1,1]
line2_y = [1,0]
line2_z = [1,1]

line3_x = [1,0]
line3_y = [1,1]
line3_z = [1,1]

# 请大家自己补充注释
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(colors_all[:,0], colors_all[:,1], colors_all[:,2], c = colors_all, s = 1, alpha = 1)

# ax.view_init(azim=30, elev=30)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# 设定横纵轴标签
ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{x_3}$')
# 绘制三根线
ax.plot(line1_x, line1_y, line1_z, alpha = 1, linewidth = 1, color='k')
# ax.plot(line2_x, line2_y, line2_z, alpha = 1, linewidth = 1, color='k')
# ax.plot(line3_x, line3_y, line3_z, alpha = 1, linewidth = 1, color='k')

# ax.grid(False)
ax.set_proj_type('ortho')
ax.set_box_aspect(aspect = (1,1,1))
# ax.view_init(azim=38, elev=34)
# fig.savefig('Figures/整个色彩空间采样，稀疏.svg', format='svg')

#>>>>>>>>>>>>>>>>>>>>>>> 立方体最艳丽的三个立面
# 颗粒度降低
colors_all, colors_bright, colors_dark = color_cubic(51)
# 每个维度散点数降低为 51

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(colors_bright[:,0],
           colors_bright[:,1],
           colors_bright[:,2],
           c = colors_bright,
           s = 1,
           alpha = 1)

# ax.view_init(azim=38, elev=34)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
# ax.set_xticks([0,1])
# ax.set_yticks([0,1])
# ax.set_zticks([0,1])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# ax.grid(False)
ax.set_proj_type('ortho')
ax.set_box_aspect(aspect = (1,1,1))

# fig.savefig('Figures/立方体最艳丽的三个立面.svg', format='svg')


#>>>>>>>>>>>>>>>>>>>>>>> 立方体最暗淡的三个立面
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(colors_dark[:,0],
           colors_dark[:,1],
           colors_dark[:,2],
           c = colors_dark,
           s = 1,
           alpha = 1)

# ax.view_init(azim=30, elev=30)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
# ax.set_xticks([0,1])
# ax.set_yticks([0,1])
# ax.set_zticks([0,1])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
# ax.grid(False)
ax.set_proj_type('ortho')
ax.set_box_aspect(aspect = (1,1,1))
# ax.view_init(azim=38, elev=34)

# fig.savefig('Figures/立方体最暗淡的三个立面.svg', format='svg')


#>>>>>>>>>>>>>>>>>>>>>>>  绘制六个侧面，三维散点
colors_all, colors_bright, colors_dark = color_cubic(21)
# 每个维度的颗粒度进一步降低至 25

# 通过面具 (mask) 筛选得到六个侧面散点, 将六个侧面的散点放在 list 里
facets_6 = [colors_all[colors_all[:,0] == 1], # colors_all[:,0] == 1 代表面具 (mask) 条件为：第一列 (红色R) 为 1
            colors_all[colors_all[:,1] == 1], # colors_all[:,1] == 1 代表面具 (mask) 条件为：第二列 (绿色G) 为 1
            colors_all[colors_all[:,2] == 1],
            colors_all[colors_all[:,0] == 0],
            colors_all[colors_all[:,1] == 0],
            colors_all[colors_all[:,2] == 0]]

# 利用 for 循环可视化六个侧面，每个侧面单独绘图, 在 for 循环中使用 enumerate()，在遍历时，同时给出索引 idx
for idx, colors_one_facet in enumerate(facets_6):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    ax.plot(line1_x, line1_y, line1_z, alpha = 1, linewidth = 1, color='k')
    ax.plot(line2_x, line2_y, line2_z, alpha = 1, linewidth = 1, color='k')
    ax.plot(line3_x, line3_y, line3_z, alpha = 1, linewidth = 1, color='k')

    ax.scatter(colors_one_facet[:,0], colors_one_facet[:,1], colors_one_facet[:,2], c = colors_one_facet, s = 3, alpha = 1)

    # 设定横纵轴标签
    ax.set_xlabel('$\it{x_1}$')
    ax.set_ylabel('$\it{x_2}$')
    ax.set_zlabel('$\it{x_3}$')

    # ax.view_init(azim=30, elev=30)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    # ax.set_xticks([0,1])
    # ax.set_yticks([0,1])
    # ax.set_zticks([0,1])

    ax.grid(False)
    ax.set_proj_type('ortho')
    ax.set_box_aspect(aspect = (1,1,1))
    # ax.view_init(azim=38, elev=34)
    # fig.savefig('Figures/六个立面_' + str(idx + 1) + '.svg', format='svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 切豆腐展示RGB色彩空间内部
import numpy as np
import matplotlib.pyplot as plt

import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 定义函数
# 自定义函数，生成三维网格数据
def color_cubic(num):
    x1 = np.linspace(0,1,num)
    x2 = x1
    x3 = x1

    # 生成三维数据网格
    xx1, xx2, xx3 = np.meshgrid(x1,x2,x3)

    # 将三维数组展成一维
    x1_ = xx1.ravel()
    x2_ = xx2.ravel()
    x3_ = xx3.ravel()

    # 将一维数组作为列堆叠成二维数组
    colors_all     = np.column_stack([x1_, x2_, x3_])
    # 利用面具 (mask) 做筛选 仅仅保留立方体三个朝外的立面： 颜色相对较为鲜亮
    colors_bright  = colors_all[np.any(colors_all == 1, axis=1)]
    # 仅仅保留立方体三个朝内的立面： 颜色相对较为暗沉
    colors_dark    = colors_all[np.any(colors_all == 0, axis=1)]
    return colors_all, colors_bright, colors_dark

# 定义三根线的坐标
line1_x = [1,1]
line1_y = [1,1]
line1_z = [1,0]

line2_x = [1,1]
line2_y = [1,0]
line2_z = [1,1]

line3_x = [1,0]
line3_y = [1,1]
line3_z = [1,1]


# 整个色彩空间采样，散点稀疏
colors_all, colors_bright, colors_dark = color_cubic(21)

#>>>>>>>>>>>>>>>>>>>>>>>  红色渐变切片,# 将红色置 0
colors_one_facet = colors_all[colors_all[:,0] == 0]

# 定义每个切片红色的色号
red_levels = np.linspace(0, 1, 6)

fig = plt.figure(figsize=(5, 30))
for idx, Red_level in enumerate(red_levels):
    # 获得数组副本
    # 用 numpy.copy() 复制时，不改变原数据
    colors_one_facet_idx = np.copy(colors_one_facet)

    # 给切片红色赋值
    colors_one_facet_idx[:,0] = colors_one_facet_idx[:,0] + Red_level

    # 三维散点可视化
    ax = fig.add_subplot(len(red_levels), 1, # 增加子图，6列、1行
                         idx + 1,            # 子图索引
                         projection = '3d')  # 子图为3D轴
    # 设定横纵轴标签
    ax.set_xlabel('$\it{x_1}$')
    ax.set_ylabel('$\it{x_2}$')
    ax.set_zlabel('$\it{x_3}$')

    ax.plot(line1_x, line1_y, line1_z, alpha = 1, linewidth = 1, color='k')
    ax.plot(line2_x, line2_y, line2_z, alpha = 1, linewidth = 1, color='k')
    ax.plot(line3_x, line3_y, line3_z, alpha = 1, linewidth = 1, color='k')

    ax.scatter(colors_one_facet_idx[:,0], colors_one_facet_idx[:,1], colors_one_facet_idx[:,2], c = colors_one_facet_idx, s = 3, alpha = 1)

    ax.view_init(azim=30, elev=30)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])

    # ax.grid(False)
    ax.set_proj_type('ortho')
    ax.set_box_aspect(aspect = (1,1,1))
    ax.view_init(azim=38, elev=34)

# fig.savefig('Figures/红色渐变切片，稀疏.svg', format='svg')


#>>>>>>>>>>>>>>>>>>>>>>>  绿色渐变切片 # 将绿色置 0
colors_one_facet = colors_all[colors_all[:,1] == 0]

# 定义每个切片绿色的色号
green_levels = np.linspace(0, 1, 6)

fig = plt.figure(figsize=(5, 30))
for idx, green_level in enumerate(green_levels):
    colors_one_facet_idx = np.copy(colors_one_facet)

    colors_one_facet_idx[:,1] = colors_one_facet_idx[:,1] + green_level

    ax = fig.add_subplot(len(red_levels), 1, idx + 1, projection = '3d')
    ax.plot(line1_x, line1_y, line1_z, alpha = 1, linewidth = 1, color='k')
    ax.plot(line2_x, line2_y, line2_z, alpha = 1, linewidth = 1, color='k')
    ax.plot(line3_x, line3_y, line3_z, alpha = 1, linewidth = 1, color='k')

    ax.scatter(colors_one_facet_idx[:,0], colors_one_facet_idx[:,1], colors_one_facet_idx[:,2], c = colors_one_facet_idx, s = 3, alpha = 1)

    ax.view_init(azim=30, elev=30)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])

    # ax.grid(False)
    ax.set_proj_type('ortho')
    ax.set_box_aspect(aspect = (1,1,1))
    ax.view_init(azim=38, elev=34)

# fig.savefig('Figures/绿色渐变切片，稀疏.svg', format='svg')


#>>>>>>>>>>>>>>>>>>>>>>>  蓝色渐变切片
colors_all, colors_bright, colors_dark = color_cubic(25)
colors_one_facet = colors_all[colors_all[:,2] == 0]
blue_levels = np.linspace(0, 1, 6)
fig = plt.figure(figsize=(5, 30))
for idx, blue_level in enumerate(blue_levels):
    colors_one_facet_idx = np.copy(colors_one_facet)
    colors_one_facet_idx[:,2] = colors_one_facet_idx[:,2] + blue_level
    ax = fig.add_subplot(len(red_levels), 1, idx + 1, projection = '3d')

    ax.plot(line1_x, line1_y, line1_z, alpha = 1, linewidth = 1, color='k')
    ax.plot(line2_x, line2_y, line2_z, alpha = 1, linewidth = 1, color='k')
    ax.plot(line3_x, line3_y, line3_z, alpha = 1, linewidth = 1, color='k')

    ax.scatter(colors_one_facet_idx[:,0], colors_one_facet_idx[:,1], colors_one_facet_idx[:,2], c = colors_one_facet_idx, s = 3, alpha = 1)

    ax.view_init(azim=30, elev=30)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])

    # ax.grid(False)
    ax.set_proj_type('ortho')
    ax.set_box_aspect(aspect = (1,1,1))
    ax.view_init(azim=38, elev=34)

# fig.savefig('Figures/蓝色渐变切片，稀疏.svg', format='svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 用指定色号和平面散点图可视化HSV色彩空间，极坐标

# 导入包
import numpy as np
import colorsys
import matplotlib.pyplot as plt
import math
import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 1. 定义函数用来绘制HSV色盘
def polar_circles(num_r, num_n):
    # 极径 [0, 1] 分成若干等份
    r_array = np.linspace(0, 1, num_r)
    # 极角 [0, 2*pi] 分成若干等份
    # HSV 色号三个值最后也会转化成 [0, 1] 之间的数值
    t_array = np.linspace(0, 2*np.pi, num_n, endpoint=False)

    # 生成极坐标网格数据
    rr, tt = np.meshgrid(r_array, t_array)

    # rr.ravel() 将二维数组展开成一维数组
    # numpy.column_stack() 则一维数组按列方向堆叠构成一个二维数组
    circles = np.column_stack([rr.ravel(), tt.ravel()])
    return circles

def plot_HSV_polar(value = 1, num_r = 10, num_n = 20):
    # 自定义函数生成散点极坐标
    circles = polar_circles(num_r, num_n)

    RHO = circles[:,0] # 极径 (2520,)
    PHI = circles[:,1] # 极角 (2520,)

    # 色调取值转换为 [0, 1] 区间
    h_ = (PHI-PHI.min()) / (PHI.max()-PHI.min())

    # 饱和度
    s_ = RHO
    # 明暗度为定值, 自定义函数中，默认为 value = 1
    v_ = np.ones_like(RHO)*value

    # 绘制极坐标
    fig = plt.figure(figsize = (3,3))
    ax = fig.add_subplot(projection='polar')

    # colorsys.hsv_to_rgb() 完成 HSV 色号向 RGB 色号转换
    h,s,v = h_.flatten().tolist(), s_.flatten().tolist(), v_.flatten().tolist()
    c = [colorsys.hsv_to_rgb(*x) for x in zip(h,s,v)]

    c = np.array(c) # c.shape

    ax.scatter(PHI, RHO, c=c, s = 3, alpha = 1)
    # ax.axis('off')
    ax.set_rlim(0, 1)
    ax.set_xticks(np.linspace(0,  2*np.pi, 12, endpoint=False))
    ax.tick_params("both", grid_linewidth=0.5)
    ax.set_rlabel_position(0)
    ax.set_axisbelow(False)

    # fig.savefig('Figures/HSV色盘_极坐标网格_V_' + str(value) + '.svg', format='svg')
    plt.show()


# 2. 绘制HSV色盘，颗粒度高
plot_HSV_polar(value = 1, num_r = 51, num_n = 780)

# 3. 绘制HSV色盘，明暗度 Value = 1.0
plot_HSV_polar(value = 1)

# 4. 绘制HSV色盘，明暗度 Value = 0.8
plot_HSV_polar(value = 0.8)

# 5. 绘制HSV色盘，明暗度 Value = 0.6
plot_HSV_polar(value = 0.6)

### 6. 绘制HSV色盘，明暗度 Value = 0.4
plot_HSV_polar(value = 0.4)

### 7. 绘制HSV色盘，明暗度 Value = 0.2
plot_HSV_polar(value = 0.2)

### 8. 绘制HSV色盘，明暗度 Value = 0.0
plot_HSV_polar(value = 0)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 用指定色号和平面散点图可视化HSV色彩空间，均匀散点

# 导入包
import numpy as np
import colorsys
import matplotlib.pyplot as plt
import math
import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 1. 定义函数用来绘制HSV色盘
def circle_points(num_r, num_n):
    r = np.linspace(0,1,num_r)
    # 极轴 [0, 1] 分成若干等份
    n = r*num_n
    # 每一层散点数 n 和半径成正比
    n = n.astype(int)
    # 将 n 转化为整数
    circles = np.empty((0,2))
    # 创建空数组

    # 用 for 循环产生每一层散点对应的极坐标
    for r_i, n_i in zip(r, n):
        t_i = np.linspace(0, 2*np.pi, n_i, endpoint=False)
        r_i = np.ones_like(t_i)*r_i

        circle_i = np.c_[r_i, t_i]
        circles = np.append(circles, circle_i, axis=0)
        # 拼接极坐标点

    return circles

def plot_HSV_even(value = 1, num_r = 10, num_n = 20):
    circles = circle_points(num_r, num_n)
    # 调用自定义函数

    RHO = circles[:,0] # 散点极角
    PHI = circles[:,1] # 散点极径

    # 色调取值转换为 [0, 1] 区间
    h_ = (PHI-PHI.min()) / (PHI.max()-PHI.min())
    # 饱和度
    s_ = RHO
    # 明暗度为定值
    v_ = np.ones_like(RHO)*value

    fig = plt.figure(figsize = (3,3))
    # 绘制极坐标
    ax = fig.add_subplot(projection='polar')

    h,s,v = h_.flatten().tolist(), s_.flatten().tolist(), v_.flatten().tolist()
    c = [colorsys.hsv_to_rgb(*x) for x in zip(h,s,v)]
    # colorsys.hsv_to_rgb() 完成 HSV 色号向 RGB 色号转换
    c = np.array(c)

    ax.scatter(PHI, RHO, c=c, s = 3, alpha = 1) # c 为散点颜色色号 s 为散点大小 alpha 为透明度；0 对应完全透明
    # ax.axis('off')
    ax.set_rlim(0,1)
    ax.set_xticks(np.linspace(0,  2*np.pi, 12, endpoint=False))
    ax.tick_params("both", grid_linewidth=0.5)
    ax.set_rlabel_position(0)
    ax.set_axisbelow(False)

    # fig.savefig('Figures/HSV色盘_散点均匀_V_' + str(value) + '.svg', format='svg')
    plt.show()

### 2. 绘制HSV色盘，颗粒度高
plot_HSV_even(value = 1, num_r = 51, num_n = 780)

### 3. 绘制HSV色盘，明暗度 Value = 1.0
plot_HSV_even(value = 1)

## 4. 绘制HSV色盘，明暗度 Value = 0.8
plot_HSV_even(value = 0.8)

### 5. 绘制HSV色盘，明暗度 Value = 0.6
plot_HSV_even(value = 0.6)

### 6. 绘制HSV色盘，明暗度 Value = 0.4
plot_HSV_even(value = 0.4)

### 7. 绘制HSV色盘，明暗度 Value = 0.2
plot_HSV_even(value = 0.2)

### 8. 绘制HSV色盘，明暗度 Value = 0.0
plot_HSV_even(value = 0)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%













































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































