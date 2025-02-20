#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 13:58:26 2023

@author: jack
"""


import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import art3d
# matplotlib.use('Agg')
from mpl_toolkits import mplot3d

filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
fonte1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=24)

fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Light Nerd Font Complete Mono.otf", size=20)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove SemiLight Nerd Font Complete.otf", size=20)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Regular Nerd Font Complete Mono.otf", size=20)

def add_borders(ax, edgecolor=(0, 0, 0, 1), linewidth=0.57, scale=1.021):
    xlims = ax.get_xlim3d()
    xoffset = (xlims[1] - xlims[0]) * scale
    xlims = np.array([xlims[1] - xoffset, xlims[0] + xoffset])
    ylims = ax.get_ylim3d()
    yoffset = (ylims[1] - ylims[0]) * scale
    ylims = np.array([ylims[1] - yoffset, ylims[0] + yoffset])
    zlims = ax.get_zlim3d()
    zoffset = (zlims[1] - zlims[0]) * scale
    zlims = np.array([zlims[1] - zoffset, zlims[0] + zoffset])
    verts1 = np.array([[xlims[0], ylims[1], zlims[1]],
                       [xlims[0], ylims[0], zlims[1]]])
    verts2 = np.array([[xlims[0], ylims[1], zlims[1]],
                       [xlims[1], ylims[1], zlims[1]]])
    verts3 = np.array([[xlims[1], ylims[1], zlims[0]],
                       [xlims[1], ylims[1], zlims[1]]])
    p = art3d.Line3DCollection([verts1,  verts2,  verts3], colors=edgecolor, linewidths=linewidth, linestyles='--')
    ax.add_collection3d(p)
    return True



#求向量积(outer()方法又称外积)
x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
#矩阵转置
y = x.copy().T
#数据z
z = np.cos(x ** 2 + y ** 2)
#绘制曲面图
fig = plt.figure()
ax = plt.axes(projection='3d')
#调用plot_surface()函数
ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
font2  = {'family':'Times New Roman','style':'normal','size':12, 'color':'#00FF00'}
ax.set_xlabel(r'X',  fontdict = font2, labelpad = 0.5)
ax.set_ylabel(r'Y', fontproperties=font3, labelpad = 0.5)
ax.set_zlabel(r'Z(cos(x))', fontproperties=font3, labelpad = 0.5)
ax.set_title('Surface plot')

# 设置坐标轴线宽
ax.xaxis.line.set_lw(.6)
ax.yaxis.line.set_lw(.6)
ax.zaxis.line.set_lw(.6)

# ax.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
# ax.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
# ax.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
# ax.spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

# 刻度间隔
x0 = MultipleLocator(1)  # x轴每10一个刻度
y0 = MultipleLocator(1)
z0 = MultipleLocator(0.5)
ax.xaxis.set_major_locator(x0)
ax.yaxis.set_major_locator(y0)
ax.zaxis.set_major_locator(z0)

# ax.tick_params(direction='in', axis='both',  labelsize=16, width=3, pad = -1)
# 刻度设置
ax.tick_params(axis='z', direction='in', labelsize=10, pad=-3, )
ax.tick_params(axis='x', direction='in', labelsize=10, pad=-5, )
ax.tick_params(axis='y', direction='in', labelsize=10, pad=-5, )

labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(14) for label in labels]  # 刻度值字号

ax.xaxis._axinfo['tick']['outward_factor'] = 0
ax.xaxis._axinfo['tick']['inward_factor'] = 0.4
ax.yaxis._axinfo['tick']['outward_factor'] = 0
ax.yaxis._axinfo['tick']['inward_factor'] = 0.4
ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
ax.zaxis._axinfo['tick']['inward_factor'] = 0.0

# 设置网格线形
ax.xaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": '--', "color": 'k'})
ax.yaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": '--', "color": 'k'})
ax.zaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": '--', "color": 'k'})

# 设置网格背景色
ax.xaxis.set_pane_color((0, 0, 0, 0))
ax.yaxis.set_pane_color((0, 0, 0, 0))
ax.zaxis.set_pane_color((0, 0, 0, 0))

# 更改z轴位置
ax.zaxis._axinfo['juggled'] = (1, 2, 7)
 # 坐标轴范围
# ax.set_xlim([84, 0]) # 调换了x轴的大小
# ax.set_ylim([0.0, 150])
 # 添加框线
add_borders(ax, edgecolor=(0, 0, 0, 1), linewidth=0.3)  # 添加边框


out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'plotfig.eps',bbox_inches = 'tight')
plt.show()
















































































































































































































































