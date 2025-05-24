#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 16:22:49 2024

@author: jack
"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 颜色映射
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import cm # Colormaps

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


cmaps = plt.colormaps()
cmaps
len(cmaps) # 166

# fig, ax = plt.subplots(figsize = (5,5))
# for cp in cmaps:
#     plt.get_cmap(cp)
#     plt.show()

plt.get_cmap('RdYlBu_r')
plt.get_cmap('rainbow')
plt.get_cmap('hsv')
plt.get_cmap('jet')
#>>>>>>>>>>>>>>>>>>>>>>  生成颜色映射
values = np.linspace(0,1,11)
cmap   = plt.get_cmap('RdYlBu_r')
colors = plt.get_cmap('RdYlBu_r')(values) # (11, 4)
rgba_color = cmap(values[0])
rgba_color
# (0.19215686274509805, 0.21176470588235294, 0.5843137254901961, 1.0)

#>>>>>>>>>>>>>>>>>>>>>>  生成颜色映射
cmap1 = matplotlib.colormaps.get_cmap('RdYlBu_r')
cmap1 = matplotlib.colormaps['RdYlBu_r']
cmap1
colors1 = matplotlib.colormaps.get_cmap('RdYlBu_r')(values) # (11, 4)
colors1 = matplotlib.colormaps['RdYlBu_r'](values) # (11, 4)
rgba_color1 = cmap1(values[0])
rgba_color1

plt.cm.RdYlBu_r(0)
# (0.19215686274509805, 0.21176470588235294, 0.5843137254901961, 1.0)
plt.cm.RdYlBu_r(0.1)
# (0.2690503652441369, 0.45397923875432533, 0.7034986543637063, 1.0)
plt.cm.RdYlBu_r(0.9)

#>>>>>>>>>>>>>>>>>>>>>>  生成颜色映射
cmap2   = matplotlib.cm.RdYlBu_r
colors2 = matplotlib.cm.RdYlBu_r(values)
rgba_color2 = matplotlib.cm.RdYlBu_r(values[0])
# (0.19215686274509805, 0.21176470588235294, 0.5843137254901961, 1.0)
# >>>>>>>>>>>>>>>>>>>>>>  生成颜色映射
# 生成数据
data = np.round(np.linspace(0, 1, 11).reshape(-1, 1),1)
# (11, 1)

# 用seaborn heatmap展示颜色，并在色块上打印RGB色号
fig, ax = plt.subplots(figsize = (8,8))
sns.heatmap(data, xticklabels = False, yticklabels = False, cmap = 'RdYlBu_r', annot=True, cbar=False, fmt='', ax=ax)
# 打印RGB色号
for i, value in enumerate(data):
    rgba_color = cmap1(value[0])
    rgb_color = tuple(int(255 * c) for c in rgba_color[:3])
    ax.text(0, i + 0.5, f'RGB: {rgb_color}', color='black', ha='left', va='center')
# plt.savefig('color mapping, 1.svg')
plt.show()

#>>>>>>>>>>>>>>>>>>>>>> 生成颜色映射
colormap = matplotlib.colormaps.get_cmap('RdYlBu_r')
# 生成数据
data = np.round(np.linspace(0, 1, 21).reshape(-1, 1),2)
# 用seaborn heatmap展示颜色，并在色块上打印RGB色号
fig, ax = plt.subplots(figsize = (8,8))
sns.heatmap(data, xticklabels = False, yticklabels = False, cmap=colormap, annot=True, cbar=False, fmt='', ax=ax)
# 打印RGB色号
for i, value in enumerate(data):
    rgba_color = colormap(value[0])
    rgb_color = tuple(int(255 * c) for c in rgba_color[:3])
    ax.text(0, i + 0.5, f'RGB: {rgb_color}', color='black', ha='left', va='center')
# plt.savefig('color mapping, 2.svg')
plt.show()

#>>>>>>>>>>>>>>>>>>>>>>  数值到颜色映射
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
# 生成一组随机数值
random_values = np.random.randint(low = 0, high = 20, size = 20)
# 定义颜色映射
colormap = matplotlib.colormaps.get_cmap('RdYlBu_r')
# 使用Normalize将随机数值映射到[0, 1]范围
norm = Normalize(vmin=random_values.min(), vmax=random_values.max())
normalized_values = norm(random_values)
# 用seaborn heatmap展示颜色
sns.heatmap(normalized_values.reshape(-1, 1), cmap=colormap, annot=True, xticklabels = False, yticklabels = False, cbar=False)
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 用指定色谱绘制三维网格曲面

# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex
from sympy.abc import x, y
# 导入符号变量
import os

from matplotlib import cm
# 导入色谱模块

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

num = 301; # number of mesh grids
x_array = np.linspace(-3,3,num)
y_array = np.linspace(-3,3,num)
xx,yy = np.meshgrid(x_array,y_array)

# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x, y], f_xy)
# 将符号函数表达式转换为Python函数
ff = f_xy_fcn(xx, yy)

# 查看函数
f_xy
#>>>>>>>>>>>>>>>>>>>>>> 2. 用plot_surface() 绘制二元函数曲面
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
#  正交投影模式
ax.set_proj_type('ortho')

# 使用 RdYlBu 色谱, 请大家试着调用其他色谱
surf = ax.plot_surface(xx, yy, ff, cmap=cm.RdYlBu, linewidth=0, antialiased=False)
# 设定横纵轴标签
ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')
# 设定横、纵轴取值范围
ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(y_array.min(), y_array.max())
# 设定观察视角
ax.view_init(azim=-135, elev=30)
# 删除网格
ax.grid(False)
# 修改字体、字号
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"
# 增加色谱条
fig.colorbar(surf, shrink=0.5, aspect=20)

# fig.savefig('Figures/用plot_surface()绘制二元函数曲面.svg', format='svg')

#>>>>>>>>>>>>>>>>>>>>>> 3. 翻转色谱
# 使用 RdYlBu_r 色谱
# RdYlBu_r 是 RdYlBu 的调转
# 请大家自行补充注释

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.set_proj_type('ortho')
surf = ax.plot_surface(xx,yy,ff, cmap='RdYlBu_r', linewidth=0, antialiased=False)

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(y_array.min(), y_array.max())

ax.view_init(azim=-135, elev=30)

ax.grid(False)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

fig.colorbar(surf, shrink=0.5, aspect=20)
# fig.savefig('Figures/翻转色谱.svg', format='svg')

#>>>>>>>>>>>>>>>>>>>>>> 4. 只保留网格线
# 同样使用 plot_surface()，不同的是只保留彩色网格
# 请大家自行补齐注释
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

norm_plt = plt.Normalize(ff.min(), ff.max())
colors = cm.RdYlBu_r(norm_plt(ff))

surf = ax.plot_surface(xx,yy,ff, facecolors = colors, linewidth = 1, shade = False) # 删除阴影
# 网格面填充为空
surf.set_facecolor((0,0,0,0))

# 白色背板
# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(y_array.min(), y_array.max())
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)

ax.grid(False)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"
# fig.savefig('Figures/只保留网格线.svg', format='svg')

#>>>>>>>>>>>>>>>>>>>>>> 5. plot_wireframe() 绘制网格曲面 + 三维等高线
# 在网格曲面基础上，叠加三维等高线
# 请大家补齐注释

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff, color = [0.5, 0.5, 0.5], linewidth = 0.25)

colorbar = ax.contour(xx,yy, ff,20, cmap = 'RdYlBu_r')
# 三维等高线

fig.colorbar(colorbar, ax=ax, shrink=0.5, aspect=20)
ax.set_proj_type('ortho')

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(y_array.min(), y_array.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/plot_wireframe() 绘制网格曲面 + 三维等高线.svg', format='svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 用指色谱绘制等高线
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex
from sympy.abc import x, y
# 导入符号变量
import os

from matplotlib import cm
# 导入色谱模块

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 1. 定义函数
num = 301;
# 数列元素数量

x_array = np.linspace(-3,3,num)
y_array = np.linspace(-3,3,num)
xx,yy = np.meshgrid(x_array,y_array)
# 产生网格数据

# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x,y],f_xy)
# 将符号函数表达式转换为Python函数
ff = f_xy_fcn(xx,yy)

# 2. 平面等高线，填充
# 四种不同色谱
cmap_arrays = ['RdYlBu_r', 'Blues_r', 'rainbow', 'viridis']

# 定义等高线高度, 几幅图采用完全一样的等高线高度
levels = np.linspace(-10, 10, 21)

# for 循环绘制四张图片
for cmap_idx in cmap_arrays:
    fig, ax = plt.subplots()
    # 绘制平面填充等高线
    colorbar = ax.contourf(xx,yy, ff, levels = levels, cmap = cmap_idx)
    ax.contour(xx,yy, ff, levels = levels, colors = 'k')
    cbar = fig.colorbar(colorbar, ax=ax)
    cbar.set_ticks([-10, -5, 0, 5, 10])
    cbar.ax.set_title(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)', fontsize = 8)
    # 增加色谱条，并指定刻度

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    # 控制横轴、纵轴取值范围

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.gca().set_aspect('equal', adjustable='box')
    # 横纵轴比例尺1:1

    title = 'Colormap = ' + str(cmap_idx)
    plt.title(title)
    # 给图像加标题

# 3. 平面等高线，非填充
# 请大家自行补充注释
for cmap_idx in cmap_arrays:
    fig, ax = plt.subplots()
    # 绘制平面等高线，非填充
    colorbar = ax.contour(xx,yy, ff, levels = levels, cmap = cmap_idx)

    cbar = fig.colorbar(colorbar, ax=ax)
    cbar.set_ticks([-10, -5, 0, 5, 10])
    cbar.ax.set_title(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)',fontsize=8)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.gca().set_aspect('equal', adjustable='box')

    title = 'Colormap = ' + str(cmap_idx)
    plt.title(title)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 可视化色谱在RGB色彩空间位置
# 导入包
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. 了解 RdYlBu 的具体颜色
mpl.colormaps['RdYlBu']

# 如果大家对某个色谱中的具体色号感兴趣的话，可以用如下办法查看
RdYlBu = mpl.colormaps['RdYlBu']
# 请大家自己分析 rainbow 和其他色谱的颜色特点

color_codes = RdYlBu(np.linspace(0, 1, 100)) # 将色谱分为 100 份

print(color_codes) # 打印每个颜色 RGBA值

# 2. 色谱在RGB空间中的位置
# color_codes 前三列无非就是三维直角坐标系中 100 个散点坐标点
# 我们可以用三维散点可视化 RdYlBu 色谱在 RGB 空间位置
fig = plt.figure(figsize = (6,6))

azim_array = [38, 0, -90, -90]
elev_array = [34, 0, 0,  90]
# 指定三种不同观察三维空间的视角

for idx, angles in enumerate(zip(azim_array, elev_array)):
    ax = fig.add_subplot(2,2,idx+1, projection = '3d')
    # 2 X 2 子图布置方案,每个子图展示一个视角
    # 绘制三维散点
    ax.scatter(color_codes[:,0],  # 色谱颜色的R值 (x坐标）
               color_codes[:,1],  # 色谱颜色的G值 (y坐标）
               color_codes[:,2],  # 色谱颜色的B值 (z坐标）
               c = color_codes,   # 指定每个点的RGB色号
               s = 4,             # 散点大小
               alpha = 1)         # 透明度
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')

    # 白色图脊
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # 白色背板
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # 格子颜色为黑色
    plt.rcParams['grid.color'] = "k"

    # 正交投影
    ax.set_proj_type('ortho')

    # 等比例尺
    ax.set_box_aspect(aspect = (1,1,1))

    # 采用指定视角
    ax.view_init(azim=angles[0], elev=angles[1])

# 3. 创建自定义函数
# 根据之前分析，构造一个可视化函数
# 函数输入为matplotlib中一个色谱的名称
# 函数可视化该色谱在RGB空间的具体位置
# 请大家自己补充注释
def visualize_cm_in_RGB(cm_name_str):
    cm_name = mpl.colormaps[cm_name_str]
    color_codes = cm_name(np.linspace(0, 1, 200)) # (200, 4)
    # print(color_codes.shape)
    fig = plt.figure(figsize = (6,6))
    azim_array = [38, 0, -90, -90]
    elev_array = [34, 0, 0,  90]

    for idx, angles in enumerate(zip(azim_array, elev_array)):
        ax = fig.add_subplot(2, 2,idx + 1, projection = '3d')
        ax.scatter(color_codes[:,0], color_codes[:,1], color_codes[:,2], c = color_codes, s = 4, alpha = 1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_zticks([0, 1])
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')

        # Transparent spines
        # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        # ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # Transparent panes
        # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # ax.grid()
        plt.rcParams['grid.color'] = "k"
        ax.set_proj_type('ortho')
        ax.set_box_aspect(aspect = (1,1,1))
        ax.view_init(azim=angles[0], elev=angles[1])
        # 注意，越靠近1，颜色越饱满，明亮

# 4. 可视化几个常用色谱
cm_list = ['RdYlBu', 'viridis', 'Blues', 'cool', 'rainbow', 'jet', 'turbo', 'hsv']
mpl.colormaps['RdYlBu']
visualize_cm_in_RGB('RdYlBu')

mpl.colormaps['viridis']
visualize_cm_in_RGB('viridis')

mpl.colormaps['Blues']
visualize_cm_in_RGB('Blues')


mpl.colormaps['cool']
visualize_cm_in_RGB('cool')


mpl.colormaps['rainbow']
visualize_cm_in_RGB('rainbow')

mpl.colormaps['jet']
visualize_cm_in_RGB('jet')


mpl.colormaps['turbo']
visualize_cm_in_RGB('turbo')


mpl.colormaps['hsv']
visualize_cm_in_RGB('hsv')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 可视化色谱在HSV色彩空间位置
# 导入包
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
# import colorsys

# 自定义函数
def visualize_cm_in_HSV(cm_name):
    fig = plt.figure(figsize = (6,6))

    azim_array = [38, 0, -90, -90]
    elev_array = [34, 0, 0,  90]
    # 指定三种不同观察三维空间的视角

    x = np.linspace(0.0, 1.0, 500)
    # 提取色谱 RGB 色号
    rgb = mpl.colormaps[cm_name](x)[np.newaxis, :, :3] # (1, 500, 3)

    theta_array = np.linspace(0, 2*np.pi, 100)
    xline = np.sin(theta_array)
    yline = np.cos(theta_array)

    # 将圆柱坐标转换为三维直角坐标系坐标
    HSV = rgb_to_hsv(rgb) # # (1, 500, 3)
    print(HSV.shape)
    HSV = HSV[0]
    HSV[:,0] = HSV[:,0] * 2 * np.pi
    HSV_xyz = np.copy(HSV)
    HSV_xyz[:,0] = HSV[:,1] * np.sin(HSV[:,0])
    HSV_xyz[:,1] = HSV[:,1] * np.cos(HSV[:,0])

    for idx, angles in enumerate(zip(azim_array, elev_array)):
        ax = fig.add_subplot(2,2,idx+1, projection = '3d')
        ax.plot(xline, yline, theta_array*0 + 0, 'k')
        ax.plot(xline, yline, theta_array*0 + 1, 'k')
        ax.plot((0,0),(0,0),(0,1), color = 'k')
        # 绘制两个单位正圆，z 的高度分别为 0、1
        # 代表 V = 0, V = 1
        # 绘制散点图
        ax.scatter(HSV_xyz[:,0], HSV_xyz[:,1], HSV_xyz[:,2], c = rgb[0], s = 10, alpha = 1)

        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(0,1)
        ax.set_xticks([-1, 1]); ax.set_yticks([-1, 1]); ax.set_zticks([0, 1])
        ax.set_zlabel('V')

        # ax.grid()
        plt.rcParams['grid.color'] = "k"
        ax.set_proj_type('ortho')
        ax.set_box_aspect(aspect = (1,1,1))
        ax.view_init(azim=angles[0], elev=angles[1])
    # fig.savefig('Figures/HSV_' + cm_name + '.svg', format='svg')

# 可视化八个色谱，HSV色彩空间
cm_list = ['RdYlBu', 'Blues', 'viridis', 'cool', 'rainbow', 'jet', 'turbo', 'hsv']
for idx, cm_idx in enumerate(cm_list):
    visualize_cm_in_HSV(cm_idx)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 自定义色谱
# 导入包
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
# matplotlib.colors.LinearSegmentedColormap() 可以用来产生连续色谱
# 函数输入为list；list 内可以是RGB/RGBA色号，也可以是色彩名称
# RGBA中的A是alpha (透明度)
# 参考
# https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LinearSegmentedColormap.html

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 函数内容来自上一话题
def visualize_cm_in_RGB(continuous_cmap, fig_name):
    color_codes = continuous_cmap(np.linspace(0, 1, 500))
    list_colors_RGB = [colors.to_rgb(rgb_idx) for rgb_idx in list_colors]
    fig = plt.figure(figsize = (6,6))
    azim_array = [38, 0, -90, -90]
    elev_array = [34, 0, 0,  90]
    for idx, angles in enumerate(zip(azim_array, elev_array)):
        ax = fig.add_subplot(2,2,idx+1, projection = '3d')
        ax.scatter(color_codes[:,0], color_codes[:,1], color_codes[:,2], c = color_codes, s = 4, alpha = 1)
        for color_idx in list_colors_RGB:
            ax.plot(color_idx[0], color_idx[1], color_idx[2], marker = 'x', markersize = 10, color = 'k')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_zticks([0, 1])
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')

        # Transparent spines
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # Transparent panes
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # ax.grid()
        plt.rcParams['grid.color'] = "k"
        ax.set_proj_type('ortho')
        ax.set_box_aspect(aspect = (1,1,1))
        ax.view_init(azim=angles[0], elev=angles[1])

    # fig.savefig('Figures/' + fig_name + '.svg', format='svg')
# 查看几个颜色名称RGB色号
print('darkblue = ')
print(colors.to_rgb('darkblue'))
print(colors.to_hex('darkblue'))

print('skyblue = ')
print(colors.to_rgb('skyblue'))
print(colors.to_hex('skyblue'))

print('white = ')
print(colors.to_rgb('white'))
print(colors.to_hex('white'))

print('pink = ')
print(colors.to_rgb('pink'))
print(colors.to_hex('pink'))

print('magenta = ')
print(colors.to_rgb('magenta'))
print(colors.to_hex('magenta'))

### 两个端点
list_nodes  = [0.0, 1.0]
# 自定义色谱
# 0， 1 为色谱的两个端点
list_colors = ['darkblue','magenta']
# 两个端点对应的色彩名称，也可以是 RGB/RGBA 色号

continuous_cmap_two_nodes = LinearSegmentedColormap.from_list("camp_name", list(zip(list_nodes, list_colors)))
continuous_cmap_two_nodes
visualize_cm_in_RGB(continuous_cmap_two_nodes, 'darkblue_0_magenta_1')


# 三个节点，均匀
list_nodes  = [0.0, 0.5, 1.0]
# 三个颜色，对应三个节点
list_colors = ['darkblue','white','magenta']
continuous_cmap_three_even_nodes = LinearSegmentedColormap.from_list("", list(zip(list_nodes, list_colors)))
continuous_cmap_three_even_nodes
visualize_cm_in_RGB(continuous_cmap_three_even_nodes, 'darkblue_0_white_.5_magenta_1')


# 三个节点，不均匀
list_nodes  = [0.0, 0.75, 1.0]
# 三个颜色，对应三个节点
list_colors = ['darkblue','white','magenta']
continuous_cmap_three_uneven_nodes = LinearSegmentedColormap.from_list("", list(zip(list_nodes, list_colors)))
continuous_cmap_three_uneven_nodes
visualize_cm_in_RGB(continuous_cmap_three_uneven_nodes, 'darkblue_0_white_.75_magenta_1')


# 五个节点，均匀
list_nodes  = [0.0, 0.25, 0.5, 0.75, 1.0]
list_colors = ['darkblue','skyblue','white','pink','magenta']
continuous_cmap_five_even_nodes = LinearSegmentedColormap.from_list("Chinese_plum_blossom", list(zip(list_nodes, list_colors)))
continuous_cmap_five_even_nodes
# 给这个色谱取了个名字——梅花盛开
visualize_cm_in_RGB(continuous_cmap_five_even_nodes, 'Chinese_plum_blossom')


list_nodes  = [0.0, 0.25, 0.5, 0.75, 1.0]
list_colors = ['darkblue','skyblue','yellow','pink','magenta']
continuous_cmap_five_even_nodes_mid_y = LinearSegmentedColormap.from_list("Chinese_plum_blossom_yl", list(zip(list_nodes, list_colors)))
continuous_cmap_five_even_nodes_mid_y
visualize_cm_in_RGB(continuous_cmap_five_even_nodes_mid_y, 'Chinese_plum_blossom_yl')


# 五个节点，不均匀
list_nodes  = [0.0, 0.1, 0.5, 0.9, 1.0]
list_colors = ['darkblue','skyblue','white','pink','magenta']
continuous_cmap_five_uneven_nodes = LinearSegmentedColormap.from_list("Chinese_plum_blossom", list(zip(list_nodes, list_colors)))
continuous_cmap_five_uneven_nodes
visualize_cm_in_RGB(continuous_cmap_five_uneven_nodes, 'five_nodes_uneven')


### RGB色谱，不循环
list_nodes  = [0.0, 1/2, 1.0]
list_colors = ['r',[0, 1, 0],'b']
# 注意，'green' 或者 'g' 对应 [0, 0.5, 0]
# 圆心很简单 [0, 1, 0] 在 RGB 中过于鲜亮刺眼
# 请大家试着调转三个基色顺序，改变节点位置
RGB_cmap_3_nodes = LinearSegmentedColormap.from_list("", list(zip(list_nodes, list_colors)))
RGB_cmap_3_nodes
visualize_cm_in_RGB(RGB_cmap_3_nodes, 'RGB_cmap_3_nodes')

print('green = ')
print(colors.to_rgb('g'))
print(colors.to_hex('g'))


# RGB色谱，循环
list_nodes  = [0.0, 1/3, 2/3, 1.0]
list_colors = ['r',[0, 1, 0],'b', 'r']
# 注意，'green' 或者 'g' 对应 [0, 0.5, 0]
# 圆心很简单 [0, 1, 0] 在 RGB 中过于鲜亮刺眼
# 请大家试着调转三个基色顺序，改变节点位置
RGB_cmap_cyclic = LinearSegmentedColormap.from_list("", list(zip(list_nodes, list_colors)))
RGB_cmap_cyclic
visualize_cm_in_RGB(RGB_cmap_cyclic, 'RGB_cmap_3_nodes_cyclic')

# CMY色谱，不循环
list_nodes  = [0.0, 1/2, 1.0]
list_colors = ['cyan','magenta','yellow']
# 请大家试着调转三个基色顺序，改变节点位置
CMY_cmap_3_nodes = LinearSegmentedColormap.from_list("", list(zip(list_nodes, list_colors)))
CMY_cmap_3_nodes
visualize_cm_in_RGB(CMY_cmap_3_nodes, 'CMY_cmap_3_nodes')

## CMY色谱，循环
list_nodes  = [0.0, 1/3, 2/3, 1.0]
list_colors = ['cyan','magenta','yellow', 'cyan']
# 用CMY基色构造色谱，循环cyclic
CMY_cmap_cyclic = LinearSegmentedColormap.from_list("", list(zip(list_nodes, list_colors)))
CMY_cmap_cyclic
visualize_cm_in_RGB(CMY_cmap_cyclic, 'CMY_cmap_cyclic')


## 仿制HSV色谱
import matplotlib as mpl
mpl.colormaps['hsv']
# Matplotlib中的hsv色谱

list_nodes  = [0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0]
list_colors = ['red','yellow', [0, 1, 0], 'cyan', 'blue', 'magenta','red']
# 仿制hsv色谱
HSV_cmap_cyclic = LinearSegmentedColormap.from_list("", list(zip(list_nodes, list_colors)))
HSV_cmap_cyclic
visualize_cm_in_RGB(HSV_cmap_cyclic, 'HSV_cmap_cyclic')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 使用自定义色谱绘制热图
# 导入包

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 生成满足标准正态分布随机数
data = np.random.randn(20, 20)

#>>>>>>>>>>>>>>>>>>>  1. 连续色谱，均匀
continuous_cmap_even = LinearSegmentedColormap.from_list('', ['darkblue','skyblue','white','pink','magenta'])
# 采用默认均匀分布节点，线性插值点
continuous_cmap_even


fig, ax = plt.subplots()
# 利用 seaborn.heatmap() 热图可视化随机数
HM = sns.heatmap(data, ax = ax,
                 vmin = -3,            # 色谱对应的最小值
                 vmax = 3,             # 色谱对应的最大值
                 cmap=continuous_cmap_even, # 使用自定义离散色谱
                 xticklabels=False,    # 删除横轴刻度标签
                 yticklabels=False,    # 删除纵轴刻度标签
                 linecolor = 'grey',   # 热图网格颜色
                 linewidths = 0.1)     # 热图网格线宽
# 在数轴上，vmin (-3) 和 vmax (3) 关于原点对称
# 这样可以保证，纯白色用来可视化 0
# 删除横、纵轴刻度
HM.tick_params(left=False, bottom=False)

#>>>>>>>>>>>>>>>>>>>  2. 连续色谱，不均匀
list_nodes  = [0.0, 0.1, 0.5, 0.9, 1.0]
list_colors = ['darkblue','skyblue','white','pink','magenta']
# 定义颜色节点位置，数值在 [0, 1] 之间
# 每个值对应一个颜色
# 'darkblue' 位于 0.0
# 'skyblue'  位于 0.1
# 'white'    位于 0.5
# 'skyblue' 到 'white' 的线性过度占据色谱的 40% （0.5 - 0.1）
# 'pink'     位于 0.9
# 'white'  到 'pink'   的线性过度占据色谱的 40% （0.9 - 0.5）
# 'magenta'  位于 1.0
continuous_cmap_uneven = LinearSegmentedColormap.from_list("", list(zip(list_nodes, list_colors)))
continuous_cmap_uneven


# 采用默认线性插值点
# 较深颜色位于色谱的两端尾部 (0.0 ~ 0.1, 0.9 ~ 1.0)
# 使用这个色谱，更容易发现离群值
fig, ax = plt.subplots()
# 利用 seaborn.heatmap() 热图可视化随机数
HM = sns.heatmap(data, ax = ax,
                 vmin = -3,            # 色谱对应的最小值
                 vmax = 3,             # 色谱对应的最大值
                 cmap=continuous_cmap_uneven, # 使用自定义离散色谱
                 xticklabels=False,    # 删除横轴刻度标签
                 yticklabels=False,    # 删除纵轴刻度标签
                 linecolor = 'grey',   # 热图网格颜色
                 linewidths = 0.1)     # 热图网格线宽

HM.tick_params(left=False, bottom=False)
# 删除横、纵轴刻度

#>>>>>>>>>>>>>>>>>>>  3. 离散色谱
from matplotlib.colors import ListedColormap
# matplotlib.colors.ListedColormap() 可以用来产生离散色谱
# 函数输入为list；list 内可以是RGB/RGBA色号，也可以是色彩名称
# RGBA中的A是alpha (透明度)
# 参考
# https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html

import seaborn as sns
discrete_cmap = ListedColormap(['darkblue','skyblue','white','white', 'pink','magenta'])
# 两个白色
discrete_cmap
fig, ax = plt.subplots()
# 利用 seaborn.heatmap() 热图可视化随机数
HM = sns.heatmap(data, ax = ax,
                 vmin = -3,          # 色谱对应的最小值
                 vmax = 3,           # 色谱对应的最大值
                 cmap=discrete_cmap, # 使用自定义离散色谱
                 xticklabels=False,  # 删除横轴刻度标签
                 yticklabels=False,  # 删除纵轴刻度标签
                 linecolor = 'grey', # 热图网格颜色
                 linewidths = 0.1)   # 热图网格线宽
# 删除横、纵轴刻度
HM.tick_params(left=False, bottom=False)

# fig.savefig('Figures/热图_离散色谱.svg', format='svg')
# [2, 3] 用品红
# [1, 2] 用粉色
# [-1, 1] 用白色
# [-2, -1] 用天蓝
# [-3, -2] 用深蓝






























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































