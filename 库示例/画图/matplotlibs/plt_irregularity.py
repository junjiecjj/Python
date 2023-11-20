#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:39:20 2023

@author: jack
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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch


# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom', x_ratio = 0.05, y_ratio = 0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    # for yi in y:
        # axins.plot(x, yi, color='b', linestyle = '-.',  linewidth = 4, alpha=0.8, label='origin')
    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left], [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom], color = 'k', lw = 1, )

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_left,ylim_bottom)
        xyA_2, xyB_2 = (xlim_right,ylim_top), (xlim_right,ylim_bottom)
    elif  linked == 'top':
        xyA_1, xyB_1 = (xlim_left,ylim_bottom), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_right,ylim_top)
    elif  linked == 'left':
        xyA_1, xyB_1 = (xlim_right,ylim_top), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_left,ylim_bottom)
    elif  linked == 'right':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_right,ylim_top)
        xyA_2, xyB_2 = (xlim_left,ylim_bottom), (xlim_right,ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1,xyB=xyB_1,coordsA="data", coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2,xyB=xyB_2,coordsA="data",  coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)

    return
"""

##==============================================================================================
##  1
##==============================================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
mlt.rcParams['axes.unicode_minus']=False # 显示负号


# -----基础数据-----
t = np.linspace(-2*np.pi, 2*np.pi, 50, endpoint=False)

x1 = np.exp(t/3) * np.sin(t/2 + np.pi/5)
x2 = np.exp(t / 2) * np.cos(t + 1)
x3 = np.sin(t)
x4 = np.cos(3*t + np.pi/3)
x5 = np.cos(t)
x6 = np.sin(3*t + np.pi/3)

# -----基础设置-----
fig = plt.figure(figsize=(18, 12), dpi=300)
# 设置整个图的标题
fontt = FontProperties(fname=fontpath+"simsun.ttf", size=25)
fig.suptitle("使用subplot2grid绘制不规则画布的图", fontproperties=fontt,)


##==============================  1 =============================================
# 构建2×3的区域，在起点(0, 0)处开始，跨域1行2列的位置区域绘图
ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=2)
ax1.plot(t, x1, color="b", linestyle="-.", linewidth=1.0, marker="d")

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
ax1.set_title("图 01", fontproperties=font3)

font3  = {'family':'Times New Roman','style':'normal','size':25}
# axs[0,0].set_xlabel(r'time (s)', fontproperties=font3)
ax1.set_xlabel("x", fontproperties=font3)
ax1.set_ylabel("y", fontproperties=font3)
ax1.text(-2, -2, r"$f(x) = \mathrm{e}^{\frac{x}{3}} \mathrm{sin}{\frac{x}{2} + \frac{\pi}{5}}$", fontproperties=font3)
ax1.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )



# font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
# legend1 = ax1.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


ax1.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax1.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
ax1.spines['left'].set_color('b')  ### 设置边框线颜色
ax1.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax1.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细
# ax.spines['top'].set_color('m')  ### 设置边框线颜色
ax1.spines['right'].set_color('r')  ### 设置边框线颜色

ax1.tick_params(direction='in', axis='both', top=True, right=True, labelsize=20, width=2, labelcolor = "red", colors='blue', rotation=25,)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(20) for label in labels] #刻度值字号

##==============================  2 =============================================
# 构建2×3的区域，在起点(0, 2)处开始，跨域2行1列的位置区域绘图
ax2 = plt.subplot2grid((2, 3), (0, 2), rowspan=2, colspan=1)
ax2.plot(t, x2, color="g", linestyle="-", linewidth=1.5, marker="p")

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
ax2.set_title("图 02", fontproperties=font3)
font3  = {'family':'Times New Roman','style':'normal','size':25}
ax2.set_xlabel("$x$", fontproperties=font3)
ax2.set_ylabel("$y$", fontproperties=font3)
ax2.text(-4, 5, r"$f(x) = \mathrm{e}^{\frac{x}{2}} \mathrm{cos}{x+1}$",fontproperties=font3)
ax2.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

# font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
# legend1 = ax2.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


ax2.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax2.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
# ax2.spines['left'].set_color('b')  ### 设置边框线颜色
ax2.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax2.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细
# ax.spines['top'].set_color('m')  ### 设置边框线颜色
# ax2.spines['right'].set_color('r')  ### 设置边框线颜色

ax2.tick_params(direction='in', axis='both', top=True, right=True, labelsize=20, width=4, labelcolor = "blue", colors='red', rotation=25,)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(20) for label in labels] #刻度值字号

##==============================  3 =============================================
# 构建2×3的区域，在起点(1, 0)处开始，跨域1行1列的位置区域绘图
ax3 = plt.subplot2grid((2, 3), (1, 0), rowspan=1, colspan=1)
ax3.plot(t, x3, color="k", linestyle="none", marker="o", markeredgecolor="orange", label=r"$y=\mathrm{sin}{x}$")
ax3.plot(t, x4, color="r", linestyle="-.", linewidth=1.0, label=r"$y=\mathrm{cos}{3x + \frac{\pi}{3}}$")
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
ax3.set_title("图 02", fontproperties=font3)
font3  = {'family':'Times New Roman','style':'normal','size':25}
ax3.set_xlabel("$x$", fontproperties=font3)
ax3.set_ylabel("$y$", fontproperties=font3)


## legend
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
legend1 = ax3.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

## linewidth
ax3.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax3.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
# ax2.spines['left'].set_color('b')  ### 设置边框线颜色
ax3.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax3.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细
# ax.spines['top'].set_color('m')  ### 设置边框线颜色
# ax2.spines['right'].set_color('r')  ### 设置边框线颜色


## tick_params
ax3.tick_params(direction='in', axis='both', top=True, right=True, labelsize=20, width=4,  )
labels = ax3.get_xticklabels() + ax3.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(20) for label in labels] #刻度值字号
##==============================  4 =============================================
# 构建2×3的区域，在起点(1, 1)处开始，跨域1行1列的位置区域绘图
ax4 = plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=1)
ax4.plot(t, x5, color="c", linestyle="--", linewidth=0.5, marker="h", label=r"$y=\mathrm{cos}{x}$")
ax4.plot(t, x6, color="b", linestyle=":", linewidth=1.0, marker="x", label=r"$y=\mathrm{sin}{3x + \frac{\pi}{3}}$")
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
ax4.set_title("图 04", fontproperties=font3)

font3  = {'family':'Times New Roman','style':'normal','size':25}
ax4.set_xlabel("$x$", fontproperties=font3)
ax4.set_ylabel("$y$", fontproperties=font3)
# ax4.legend(fontsize=15)


## legend
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
legend1 = ax4.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

## linewidth
ax4.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax4.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
# ax4.spines['left'].set_color('b')  ### 设置边框线颜色
ax4.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax4.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细
# ax4.spines['top'].set_color('m')  ### 设置边框线颜色
# ax4.spines['right'].set_color('r')  ### 设置边框线颜色


## tick_params
ax4.tick_params(direction='in', axis='both', top=True, right=True, labelsize=20, width=4,  )
labels = ax4.get_xticklabels() + ax4.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(20) for label in labels] #刻度值字号


##=========================================================

filepath2 = '/home/jack/snap/'
# out_fig = plt.gcf()
plt.savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight' )
plt.show()
plt.close()





##==============================================================================================
##    2
##==============================================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
mlt.rcParams['axes.unicode_minus']=False # 显示负号


# -----基础数据-----
t = np.linspace(-2*np.pi, 2*np.pi, 50, endpoint=False)

x1 = np.exp(t/3) * np.sin(t/2 + np.pi/5)
x2 = np.exp(t / 2) * np.cos(t + 1)
x3 = np.sin(t)
x4 = np.cos(3*t + np.pi/3)
x5 = np.cos(t)
x6 = np.sin(3*t + np.pi/3)

# -----基础设置-----
fig = plt.figure(figsize=(12, 12), dpi=300)
# 设置整个图的标题
fontt = FontProperties(fname=fontpath+"simsun.ttf", size=25)
fig.suptitle("使用subplot2grid绘制不规则画布的图", fontproperties=fontt,)


##==============================  1 =============================================
# 构建2×3的区域，在起点(0, 0)处开始，跨域1行2列的位置区域绘图
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=2)
ax1.plot(t, x1, color="b", linestyle="-.", linewidth=1.0, marker="d")

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
ax1.set_title("图 01", fontproperties=font3)

font3  = {'family':'Times New Roman','style':'normal','size':25}
# axs[0,0].set_xlabel(r'time (s)', fontproperties=font3)
ax1.set_xlabel("x", fontproperties=font3)
ax1.set_ylabel("y", fontproperties=font3)
ax1.text(-2, -2, r"$f(x) = \mathrm{e}^{\frac{x}{3}} \mathrm{sin}{\frac{x}{2} + \frac{\pi}{5}}$", fontproperties=font3)
ax1.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )



# font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
# legend1 = ax1.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


ax1.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax1.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
ax1.spines['left'].set_color('b')  ### 设置边框线颜色
ax1.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax1.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细
# ax.spines['top'].set_color('m')  ### 设置边框线颜色
ax1.spines['right'].set_color('r')  ### 设置边框线颜色

ax1.tick_params(direction='in', axis='both', top=True, right=True, labelsize=20, width=2, labelcolor = "red", colors='blue', rotation=25,)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(20) for label in labels] #刻度值字号

##==============================  2 =============================================
# 构建2×3的区域，在起点(0, 2)处开始，跨域2行1列的位置区域绘图
ax2 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax2.plot(t, x2, color="g", linestyle="-", linewidth=1.5, marker="p")

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
ax2.set_title("图 02", fontproperties=font3)
font3  = {'family':'Times New Roman','style':'normal','size':25}
ax2.set_xlabel("$x$", fontproperties=font3)
ax2.set_ylabel("$y$", fontproperties=font3)
ax2.text(-4, 5, r"$f(x) = \mathrm{e}^{\frac{x}{2}} \mathrm{cos}{x+1}$",fontproperties=font3)
ax2.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

# font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
# legend1 = ax2.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


ax2.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax2.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
# ax2.spines['left'].set_color('b')  ### 设置边框线颜色
ax2.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax2.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细
# ax.spines['top'].set_color('m')  ### 设置边框线颜色
# ax2.spines['right'].set_color('r')  ### 设置边框线颜色

ax2.tick_params(direction='in', axis='both', top=True, right=True, labelsize=20, width=4, labelcolor = "blue", colors='red', rotation=25,)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(20) for label in labels] #刻度值字号

##==============================  3 =============================================
# 构建2×3的区域，在起点(1, 0)处开始，跨域1行1列的位置区域绘图
ax3 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
ax3.plot(t, x3, color="k", linestyle="none", marker="o", markeredgecolor="orange", label=r"$y=\mathrm{sin}{x}$")
ax3.plot(t, x4, color="r", linestyle="-.", linewidth=1.0, label=r"$y=\mathrm{cos}{3x + \frac{\pi}{3}}$")
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
ax3.set_title("图 03", fontproperties=font3)
font3  = {'family':'Times New Roman','style':'normal','size':25}
ax3.set_xlabel("$x$", fontproperties=font3)
ax3.set_ylabel("$y$", fontproperties=font3)


## legend
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
legend1 = ax3.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

## linewidth
ax3.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax3.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
# ax2.spines['left'].set_color('b')  ### 设置边框线颜色
ax3.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax3.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细
# ax.spines['top'].set_color('m')  ### 设置边框线颜色
# ax2.spines['right'].set_color('r')  ### 设置边框线颜色


## tick_params
ax3.tick_params(direction='in', axis='both', top=True, right=True, labelsize=20, width=4,  )
labels = ax3.get_xticklabels() + ax3.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(20) for label in labels] #刻度值字号


##=========================================================

filepath2 = '/home/jack/snap/'
# out_fig = plt.gcf()
plt.savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight' )
plt.show()

"""


##==============================================================================================
##    3
##==============================================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
mlt.rcParams['axes.unicode_minus']=False # 显示负号


# -----基础数据-----
t = np.linspace(-2*np.pi, 2*np.pi, 50, endpoint=False)

x1 = np.exp(t/3) * np.sin(t/2 + np.pi/5)
x2 = np.exp(t / 2) * np.cos(t + 1)
x3 = np.sin(t)
x4 = np.cos(3*t + np.pi/3)
x5 = np.cos(t)
x6 = np.sin(3*t + np.pi/3)

# -----基础设置-----
fig = plt.figure(figsize=(16, 14), constrained_layout=True )
# 设置整个图的标题
fontt = FontProperties(fname=fontpath+"simsun.ttf", size=25)
fig.suptitle("使用subplot2grid绘制不规则画布的图", fontproperties=fontt,)


##==============================  1 =============================================
# 构建2×3的区域，在起点(0, 0)处开始，跨域1行2列的位置区域绘图
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1)
axins = ax1.inset_axes((0.3, 0.2, 0.25, 0.2))

ax1.plot(t, x1, color="b", linestyle="-", linewidth=1.0, marker="d", markevery = 6)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
ax1.set_title("图 01", fontproperties=font3)

font3  = {'family':'Times New Roman','style':'normal','size':25}
# axs[0,0].set_xlabel(r'time (s)', fontproperties=font3)
ax1.set_xlabel("x", fontproperties=font3)
ax1.set_ylabel("y", fontproperties=font3)
ax1.text(-4, 2, r"$f(x) = \mathrm{e}^{\frac{x}{3}} \mathrm{sin}{\frac{x}{2} + \frac{\pi}{5}}$", fontproperties=font3)
ax1.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )



ax1.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax1.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
ax1.spines['left'].set_color('b')  ### 设置边框线颜色
ax1.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax1.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细
# ax.spines['top'].set_color('m')  ### 设置边框线颜色
ax1.spines['right'].set_color('r')  ### 设置边框线颜色

ax1.tick_params(direction='in', axis='both', top=True, right=True, labelsize=20, width=2, labelcolor = "red", colors='blue', rotation=25,)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(20) for label in labels] #刻度值字号

###  son
axins.plot(t, x1, color="b", linestyle="--", linewidth=1.0, marker = "o", markevery = 6)

##==================== mother and son ==================================
## 局部显示并且进行连线,方法3
zone_and_linked(ax1, axins, 40, 45, t , [x1], 'bottom', x_ratio = 0.2, y_ratio = 0.2)
## linewidth
bw = 1
axins.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
axins.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
axins.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
axins.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

axins.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize=16, width = 1)
labels = axins.get_xticklabels() + axins.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(16) for label in labels] #刻度值字号

##==============================  2 =============================================
# 构建2×3的区域，在起点(0, 2)处开始，跨域2行1列的位置区域绘图
ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax2.plot(t, x2, color="g", linestyle="-", linewidth=1.5, marker="p")

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
ax2.set_title("图 02", fontproperties=font3)
font3  = {'family':'Times New Roman','style':'normal','size':25}
ax2.set_xlabel("$x$", fontproperties=font3)
ax2.set_ylabel("$y$", fontproperties=font3)
ax2.text(-4, 5, r"$f(x) = \mathrm{e}^{\frac{x}{2}} \mathrm{cos}{x+1}$",fontproperties=font3)
ax2.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

# font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
# legend1 = ax2.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


ax2.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax2.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
# ax2.spines['left'].set_color('b')  ### 设置边框线颜色
ax2.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax2.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细
# ax.spines['top'].set_color('m')  ### 设置边框线颜色
# ax2.spines['right'].set_color('r')  ### 设置边框线颜色

ax2.tick_params(direction='in', axis='both', top=True, right=True, labelsize=20, width=4, labelcolor = "blue", colors='red', rotation=25,)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(20) for label in labels] #刻度值字号

##==============================  3 =============================================
# 构建2×3的区域，在起点(1, 0)处开始，跨域1行1列的位置区域绘图
ax3 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
ax3.plot(t, x3, color="k", linestyle="none", marker="o", markeredgecolor="orange", label=r"$y=\mathrm{sin}{x}$")
ax3.plot(t, x4, color="r", linestyle="-.", linewidth=1.0, label=r"$y=\mathrm{cos}{3x + \frac{\pi}{3}}$")
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
ax3.set_title("图 03", fontproperties=font3)
font3  = {'family':'Times New Roman','style':'normal','size':25}
ax3.set_xlabel("$x$", fontproperties=font3)
ax3.set_ylabel("$y$", fontproperties=font3)


## legend
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
legend1 = ax3.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

## linewidth
ax3.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax3.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
# ax2.spines['left'].set_color('b')  ### 设置边框线颜色
ax3.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax3.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细
# ax.spines['top'].set_color('m')  ### 设置边框线颜色
# ax2.spines['right'].set_color('r')  ### 设置边框线颜色


## tick_params
ax3.tick_params(direction='in', axis='both', top=True, right=True, labelsize=20, width=4,  )
labels = ax3.get_xticklabels() + ax3.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(20) for label in labels] #刻度值字号


##=========================================================

filepath2 = '/home/jack/snap/'
# out_fig = plt.gcf()
plt.savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight' )
plt.show()
















































































































































