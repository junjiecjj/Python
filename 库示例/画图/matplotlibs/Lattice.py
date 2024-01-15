#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:41:39 2023

@author: jack

pip install hexalattice


hex_centers, h_ax = create_hex_grid(
     nx              = 4,        # Number of horizontal hexagons in rectangular grid, [nx * ny]
     ny              = 5,        # Number of vertical hexagons in rectangular grid, [nx * ny]
     min_diam        = 1.,       # 每个六边形的最小直径
     n               = 0,        # 创建矩形网格的另一种方法。最终网格中的六边形可能会减少
     align_to_origin = True,     # 移动网格，使中央瓦片以原点为中心
     face_color      = None,     # 提供 RGB 三连字符、有效缩写（如 "k"）或 RGB+alpha
     edge_color      = 'k',      # 提供 RGB 三连字符、有效缩写（如 "k"）或 RGB+alpha
     plotting_gap    = 0.,       # 相邻瓷砖边缘之间的间隙，单位为 min_diam 的分数
     crop_circ       = 0.,       # 如果为 0 则禁用。 如果 >0，将保留一圈中心瓷砖，半径为 r=crop_circ
     do_plot         = False,    # 将六边形添加到轴上。如果没有提供 h_ax，将打开一个新的图形。
     rotate_deg      = 0.,       # 将网格围绕中心磁砖中心旋转，旋转度数为 rotate_deg
     h_ax            = None      # 轴的句柄。如果提供，网格将被添加到其中；如果不提供，将打开一个新的图形。
)



"""

import matplotlib
# matplotlib.get_backend()
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"


# from hexgrid import HexagonalGrid

import  hexalattice.hexalattice  as  lattice


# grid = HexagonalGrid(scale = 50, grid_width=6, grid_height = 6)

# grid.draw()
fig, axs = plt.subplots(1, 1, figsize=(10, 16))

hex_centers, axs = lattice.create_hex_grid(nx = 5, ny = 5, min_diam = 2, crop_circ = 0, edge_color = 'b',  do_plot = True, h_ax = axs  )

axs.scatter(hex_centers[:,0], hex_centers[:,1], color = 'red', s = 4)
axs.scatter(0, 0, color = 'green', s = 30)


# font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
font2  = {'family':'Times New Roman','style':'normal','size':17,  }
axs.set_xlabel(r'X', fontproperties=font2, labelpad = 2.5) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。)
axs.set_ylabel(r'Y', fontproperties=font2,  labelpad = 2.5)
# axs[0].set_title('sin and tan 函数', fontproperties=font1)



out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
out_fig .savefig(filepath2+'plotfig.eps',bbox_inches = 'tight')
plt.close()








# lattice.create_hex_grid(nx=50,
#                 ny=50,
#                 min_diam=1,
#                 rotate_deg=5,
#                 crop_circ=20,
#                 do_plot=True,
#                 h_ax=h_ax)





