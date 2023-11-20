#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:47:58 2023

@author: jack

https://juejin.cn/post/6844904183548608520
https://zhuanlan.zhihu.com/p/136574534
https://blog.csdn.net/weixin_45826022/article/details/113486448

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
# font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"




def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom', x_ratio=0.05, y_ratio=0.05):
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

    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left], [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black")

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




def awgn(x, snr, seed=7):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    np.random.seed(seed)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)   #信号功率
    npower = xpower / snr         # 噪声功率，对于均值为0的正态分布，等于噪声的方差,因为D(X) = E(X^2) - E(X)^2 = E(X^2)
    noise = np.random.randn(len(x)) * np.sqrt(npower)   #np.random.randn()
    return x + noise

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

##===============================================================================
x = np.arange(0, 3, 0.001)
y = np.sin(2*np.pi*x)
snr = 33

y1 = y + wgn(x, snr)  # 增加了6dBz信噪比噪声的信号
y2 = awgn(y, snr)


fig, axs = plt.subplots(1,1, figsize=(8, 6), constrained_layout=True)

axs.plot(x, y, 'r--',   lw=4, label = 'origin')
axs.plot(x, y1, 'g-',  lw=1, label = 'awgn1')
axs.plot(x, y2, 'b-.', lw=0.5, label = 'awgn2')


font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs.set_xlabel('Epoch',fontproperties=font)
axs.set_ylabel('Training loss',fontproperties=font)
# axs.set_title(label, fontproperties=font)
#font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)


font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
# legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

#  edgecolor='black',
# facecolor = 'y', legend背景颜色, none背景透明
# edgecolor 图例legend边框线颜色
# labelcolor 字体颜色
# borderaxespad: legend边框距离画布边框的距离，一般为1
# handletextpad: legend的文字和线段的距离
# handlelength: legend的线段的长度
# labelspacing: 相邻legend的垂直距离
legend1 = axs.legend(loc='upper left', prop=font1, bbox_to_anchor=(0.5, -0.1), labelspacing = 0.2, facecolor = 'none', edgecolor = 'b', labelcolor = 'r', borderaxespad=0.2, handletextpad = 0, handlelength = 1, framealpha = 0.2, )
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs.spines['left'].set_color('b')  ### 设置边框线颜色
axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3, labelcolor = "red", colors='blue', rotation=25,)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号




##========================== ax in ===============================================
# axins = inset_axes(ax, width="40%", height="30%", loc='lower left', bbox_to_anchor=(0.3, 0.1, 1, 1), bbox_transform=ax.transAxes)
# 参数解释如下：
    # ax：父坐标系
    # width, height：子坐标系的宽度和高度（百分比形式或者浮点数个数）
    # loc：子坐标系的位置; 固定坐标系的宽度和高度以及边界框，分别设置loc为左上、左下、右上（默认）、右下和中间，
    # bbox_to_anchor：边界框，四元数组（x0, y0, width, height）
    # bbox_transform：从父坐标系到子坐标系的几何映射
    # axins：子坐标系

# 另外有一种更加简洁的子坐标系嵌入方法：
    # axins = ax.inset_axes((0.2, 0.2, 0.4, 0.3))
    # ax为父坐标系，后面四个参数同样是（x0, y0, width, height），
    # 上述代码的含义是：以父坐标系中的x0=0.2*x，y0=0.2*y为左下角起点，嵌入一个宽度为0.2*x，高度为0.3*y的子坐标系，其中x和y分别为父坐标系的坐标轴范围。效果如下图所示：

# axins = inset_axes(axs, width = "40%", height = "30%", loc = 'lower left', bbox_to_anchor = (1.1, 0.8, 1, 1),  bbox_transform = axs.transAxes)
axins = axs.inset_axes((1.1, 1, 0.4, 0.3))
axins.plot(x, y, color='b', linestyle = '-.',  linewidth = 4, alpha=0.8, label='origin')
axins.plot(x, y1, color='r', linestyle = '--',  linewidth = 1, alpha=0.8, label='noise1')
axins.plot(x, y2, color='g', linestyle = '-',  linewidth = 0.3, alpha=0.8, label='noise2')





# # 设置放大区间
# zone_left = 1000
# zone_right = 1200

# # 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 0  # x轴显示范围的扩展比例
# y_ratio = 0.05  # y轴显示范围的扩展比例

# # X轴的显示范围
# xlim0 = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
# xlim1 = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

# # Y轴的显示范围
# y = np.hstack((y[zone_left:zone_right], y[zone_left:zone_right] ))
# ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
# ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio

# # 调整子坐标系的显示范围
# axins.set_xlim(xlim0, xlim1)
# axins.set_ylim(ylim0, ylim1)

# # 原图中画方框
# tx0 = xlim0
# tx1 = xlim1
# ty0 = ylim0
# ty1 = ylim1
# sx = [tx0,tx1,tx1,tx0,tx0]
# sy = [ty0,ty0,ty1,ty1,ty0]
# axs.plot(sx,sy,"black")

## 建立父坐标系与子坐标系的连接线,方法一
# con = ConnectionPatch(xyA = (xlim0, ylim0),xyB = (xlim0, ylim0), coordsA = "data", coordsB = "data", axesA = axs, axesB = axins)
# axins.add_artist(con)
# con = ConnectionPatch(xyA = (xlim1,ylim0), xyB = (xlim1,ylim0), coordsA = "data", coordsB = "data", axesA = axs, axesB = axins)
# axins.add_artist(con)

## 建立父坐标系与子坐标系的连接线,方法二
# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
# mark_inset(axs, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size = 12)
legend1 = axins.legend(loc='best',  prop=font1,  facecolor = 'none', edgecolor = 'r', labelcolor = 'b', borderaxespad=0, framealpha = 1)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


### 局部显示并且进行连线,方法3
zone_and_linked(axs, axins, 1000, 1200, x , [y,y1,y2], 'right')
##===============================================================================

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()































































































































































































































































































































































