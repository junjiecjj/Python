#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 21:58:46 2025

@author: jack
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
# import torch
from matplotlib.font_manager import FontProperties

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300      # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22

# 获取当前系统用户目录
home = os.path.expanduser('~')
savedir = home + '/FL_DQ/Figures/MNIST'

fontpath = "/usr/share/fonts/truetype/windows/"

mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', '$\\clubsuit$', '$\\bowtie$', '$\\boxdot$', '$\\boxtimes$', '$\\boxminus$',  '$\\boxplus$', '$\\bigodot$','$\\bigoplus$', '$\\bigotimes$', '$\\bigstar$', '$\\blacksquare$', '$\circledR$', '$\circledS$',  '$\\bigtriangledown$','$\\bigtriangleup$','$\\blacktriangleleft$', '$\\blacktriangleright$', '$\\diamondsuit',   '$\heartsuit$', '$666$', '$\\varrho$', '$\\Omega$', '$\\nabla$', '$\\Upsilon$', '$\\mho$', '$\\Phi$',  '$\\divideontimes$', '$\\otimes$', '$\\odot$', '$\\ominus$', '$\\oplus$', '$\\oslash$', '$\\otimes$']

colors = ['#FF0000','#0000FF','#00FF00','#1E90FF','#4ea142','#FF00FF','#FFA500','#800080','#EE82EE','#00FFFF','#9932CC','#FF6347','#00CED1','#CD5C5C',  '#7B68EE','#808000']

# colors = ['#FF7F00', '#9BDCFC', '#CAC8EF', '#8370FE', '#A020EF', '#0000FE', '#01008A', '#1E90FF', '#228B22',  '#C9EFBE', '#F0CFEA', '#FE0000', '#00ADEF', '#63BA45', '#FF0000','#1f77b4', '#2ca02c', '#9467bd', '#4ea142','#1E90FF','#FF00FF','#FFA500','#800080','#EE82EE','#00FFFF','#9932CC','#FF6347','#00CED1','#CD5C5C',  '#7B68EE','#808000']


# colors = plt.cm.tab10(np.linspace(0, 1, 6)) # colormap

lsty = [(0, (3, 10, 1, 10, 1, 10)), (0, (1, 1)), (0, (1, 2)), (0, (5, 1)), (0, (1, 10)), (0, (1, 2)),  (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),  '-', ':', '--', '-.', ]
alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom', x_ratio = 0.1, y_ratio = 0.1):
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

def MNIST_IID_841bit():
    fig, axs = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)
    axins = axs.inset_axes((0.62, 0.4, 0.3, 0.32))
    L = 300

    rootdir = f"{home}/FL_DQ/MNIST_IID/"
    data = np.load(os.path.join(rootdir, "MNIST_IID_epoch1_Perfect_adam_0.01_U100+10_bs64_2025-12-12-11:41:59/TraRecorder.npy"))[:L]
    Y1 = data[:,1]
    axs.plot(data[:,0], Y1 , color = 'k', linestyle= '-',lw = 2, label = '完美传输',)
    axins.plot(data[:,0], Y1, color = 'k', linestyle = '-', linewidth = 2)

    i = 0
    data = np.load(os.path.join(rootdir, "MNIST_IID_epoch1_8bits_sr_erf_adam_0.01_U100+10_bs64_2025-12-12-11:42:20/TraRecorder.npy"))[:L]
    Y2 = data[:,1]
    axs.plot(data[:,0], Y2, color = colors[i], lw = 2, linestyle='--', marker = 'o', ms = 14, markevery=30, mfc='white', mew = 2, label = r'$\mathrm{8bit}$'+', 无错传输',)
    axins.plot(data[:,0], Y2, color = colors[i], ls = '--', lw = 2, marker = 'o', ms = 14, markevery=10, mfc='white', mew = 2,)
    i += 1

    data = np.load(os.path.join(rootdir, "MNIST_IID_epoch1_DQ_sr_erf_adam_0.01_U100+10_bs64_2025-12-12-23:16:04/TraRecorder.npy"))[:L]
    Y3 = data[:,1]
    axs.plot(data[:,0], Y3, color = colors[i], lw = 2, linestyle='--', marker = '*', ms = 12, markevery=30, label = r'$\mathrm{DQ}$'+', 无错传输',)
    axins.plot(data[:,0], Y3, color = colors[i], linestyle = '--', linewidth = 2, marker = '*', ms = 12, markevery=10,  )
    i += 1

    data = np.load(os.path.join(rootdir, "MNIST_IID_epoch1_4bits_sr_erf_adam_0.01_U100+10_bs64_2025-12-12-15:50:46/TraRecorder.npy"))[:L]
    Y4 = data[:,1]
    axs.plot(data[:,0], Y4, color = colors[i], lw = 2, linestyle='--', label = r'$\mathrm{4bit}$'+', 无错传输',)
    axins.plot(data[:,0], Y4, color = colors[i], linestyle = '--', linewidth = 2)
    i += 1

    data = np.load(os.path.join(rootdir, "MNIST_IID_epoch1_1bits_sr_erf_adam_0.01_U100+10_bs64_2025-12-12-21:20:50/TraRecorder.npy"))[:L]
    Y5 = data[:,1]
    axs.plot(data[:,0], Y5, color = colors[i], lw = 2, linestyle='--', label = r'$\mathrm{1bit}$'+', 无错传输',)
    axins.plot(data[:,0], Y5, color = colors[i], linestyle = '--', linewidth = 2)
    i += 1

    ###########
    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    font2 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
    axs.set_xlabel( "通信轮数", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    axs.set_ylabel('学习精度', fontproperties=font2, )

    font2 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2, borderpad = 0.1, labelspacing = 0.1)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')                         # 设置图例legend背景透明

    # x_major_locator = MultipleLocator(5)               # 把x轴的刻度间隔设置为1，并存在变量里
    # axs.xaxis.set_major_locator(x_major_locator)       # 把x轴的主刻度设置为1的倍数
    axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 25, width=3,)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels]  # 刻度值字号

    # axs.set_xlim(-0.2, 2)  #拉开坐标轴范围显示投影
    axs.set_ylim(0.5, 1.01)  #拉开坐标轴范围显示投影

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    ###==================== mother and son ==================================
    ### 局部显示并且进行连线,方法3
    zone_and_linked(axs, axins, L-50, L-20, data[:, 0] , [Y1, Y2, Y3, Y4, Y5], 'bottom', x_ratio = 0.3, y_ratio = 0.2)
    ## linewidth
    bw = 1
    axins.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axins.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axins.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axins.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    axins.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 22,  width = 1)
    labels = axins.get_xticklabels() + axins.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # [label.set_fontsize(16) for label in labels] #刻度值字号

    out_fig = plt.gcf()
    out_fig.savefig(f'{savedir}/Fig_MNIST_IID_DQ841bit.pdf' )
    plt.show()
    plt.close()
    return

def DynamicBitWidth():
    fig, axs = plt.subplots(4, 1, figsize=(8, 12), constrained_layout=True, sharex=True, sharey=True)
    font2 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
    fig.text(-0.04, 0.5, '量化比特数', va = 'center', rotation = 'vertical', fontproperties=font2,)
    L = 300
    i = 0
    rootdir = f"{home}/FL_DQ/MNIST_IID/"

    data = np.load(os.path.join(rootdir, "MNIST_IID_epoch1_DQ_sr_erf_adam_0.01_U100+10_bs64_2025-12-12-23:16:04/TraRecorder.npy"))[:L]
    Y1 = data[:,4]
    axs[0].plot(data[:,0], Y1, color = colors[i], lw = 2, linestyle='-', label = r'$\mathrm{DQ}$'+',无错传输',)
    i += 1

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    font2 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
    # axs[0].set_xlabel( "通信轮数", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    # axs[0].set_ylabel('量化比特数', fontproperties=font2, )

    font2 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    legend1 = axs[0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2, borderpad = 0.1, labelspacing = 0.1)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')                         # 设置图例legend背景透明

    # x_major_locator = MultipleLocator(5)               # 把x轴的刻度间隔设置为1，并存在变量里
    # axs.xaxis.set_major_locator(x_major_locator)       # 把x轴的主刻度设置为1的倍数
    axs[0].tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 25, width=3,)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels]  # 刻度值字号

    # axs.set_xlim(-0.2, 2)  #拉开坐标轴范围显示投影
    # axs.set_ylim(0.5, 1.01)  #拉开坐标轴范围显示投影

    axs[0].grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs[0].spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs[0].spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs[0].spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs[0].spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    #########
    data = np.load(os.path.join(rootdir, "MNIST_IID_epoch1_DQ_sr_flip0.1_adam_0.01_U100+10_bs64_2025-12-13-00:11:24/TraRecorder.npy"))[:L]
    Y1 = data[:,4]
    axs[1].plot(data[:,0], Y1, color = colors[i], lw = 2, linestyle='-', label = r'$\mathrm{DQ, BER=0.1}$',)
    i += 1

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    font2 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
    # axs[1].set_xlabel( "通信轮数", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    # axs[1].set_ylabel('量化比特数', fontproperties=font2, )

    font2 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    legend1 = axs[1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2, borderpad = 0.1, labelspacing = 0.1)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')                         # 设置图例legend背景透明

    # x_major_locator = MultipleLocator(5)               # 把x轴的刻度间隔设置为1，并存在变量里
    # axs.xaxis.set_major_locator(x_major_locator)       # 把x轴的主刻度设置为1的倍数
    axs[1].tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 25, width=3,)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels]  # 刻度值字号

    # axs.set_xlim(-0.2, 2)  #拉开坐标轴范围显示投影
    # axs.set_ylim(0.5, 1.01)  #拉开坐标轴范围显示投影

    axs[1].grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs[1].spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs[1].spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs[1].spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs[1].spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    ##########
    data = np.load(os.path.join(rootdir, "MNIST_IID_epoch1_DQ_sr_flip0.2_adam_0.01_U100+10_bs64_2025-12-13-00:39:45/TraRecorder.npy"))[:L]
    Y1 = data[:,4]
    axs[2].plot(data[:,0], Y1, color = colors[i], lw = 2, linestyle='-', label = r'$\mathrm{DQ, BER=0.2}$',)
    i += 1

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    font2 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
    # axs[2].set_xlabel( "通信轮数", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    # axs[2].set_ylabel('量化比特数', fontproperties=font2, )

    font2 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    legend1 = axs[2].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2, borderpad = 0.1, labelspacing = 0.1)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')                         # 设置图例legend背景透明

    # x_major_locator = MultipleLocator(5)               # 把x轴的刻度间隔设置为1，并存在变量里
    # axs.xaxis.set_major_locator(x_major_locator)       # 把x轴的主刻度设置为1的倍数
    axs[2].tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 25, width=3,)
    labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels]  # 刻度值字号

    # axs.set_xlim(-0.2, 2)  #拉开坐标轴范围显示投影
    # axs.set_ylim(0.5, 1.01)  #拉开坐标轴范围显示投影

    axs[2].grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs[2].spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs[2].spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs[2].spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs[2].spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    ###########
    data = np.load(os.path.join(rootdir, "MNIST_IID_epoch1_DQ_sr_flip0.3_adam_0.01_U100+10_bs64_2025-12-13-00:39:55/TraRecorder.npy"))[:L]
    Y1 = data[:,4]
    axs[3].plot(data[:,0], Y1, color = colors[i], lw = 2, linestyle='-', label = r'$\mathrm{DQ, BER=0.3}$',)
    i += 1

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    font2 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
    axs[3].set_xlabel( "通信轮数", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    # axs[3].set_ylabel('量化比特数', fontproperties=font2, )

    font2 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    legend1 = axs[3].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2, borderpad = 0.1, labelspacing = 0.1)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')                         # 设置图例legend背景透明

    # x_major_locator = MultipleLocator(5)               # 把x轴的刻度间隔设置为1，并存在变量里
    # axs.xaxis.set_major_locator(x_major_locator)       # 把x轴的主刻度设置为1的倍数
    axs[3].tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 25, width=3,)
    labels = axs[3].get_xticklabels() + axs[3].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels]  # 刻度值字号

    # axs.set_xlim(-0.2, 2)  #拉开坐标轴范围显示投影
    # axs.set_ylim(0.5, 1.01)  #拉开坐标轴范围显示投影

    axs[3].grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs[3].spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs[3].spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs[3].spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs[3].spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    # [label.set_fontsize(16) for label in labels] #刻度值字号

    out_fig = plt.gcf()
    out_fig.savefig(f'{savedir}/Fig_MNIST_IID_DQbw.pdf' )
    plt.show()
    plt.close()
    return

def CommOverHead():
    fig, axs = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)
    lw = 2
    L = 500
    V = 21880
    rootdir = f"{home}/FL_DQ/MNIST_IID/"
    i = 0
    data = np.load(os.path.join(rootdir, "MNIST_IID_epoch1_DQ_sr_erf_adam_0.01_U100+10_bs64_2025-12-12-23:16:04/TraRecorder.npy"))[:L]
    Y1 = np.cumsum(data[:,4]*V)
    axs.plot(data[:,0], Y1, color = colors[i],  ls='-', lw = lw,  marker = mark[i], markersize = 10, markevery=100, label = 'DQ, 无错传输',)
    i += 1

    data1 = np.load(os.path.join(rootdir, "MNIST_IID_epoch1_DQ_sr_flip0.1_adam_0.01_U100+10_bs64_2025-12-13-00:11:24/TraRecorder.npy"))[:L]
    Y1 = np.cumsum(data1[:,4]*V)
    axs.plot(data1[:,0], Y1, color = colors[i],  ls='-', lw = lw,  marker = mark[i], markersize = 10, markevery=100, label = 'DQ, '+ r'$\mathrm{BER=0.1}$',)
    i += 1

    data = np.load(os.path.join(rootdir, "MNIST_IID_epoch1_DQ_sr_flip0.2_adam_0.01_U100+10_bs64_2025-12-13-00:39:45/TraRecorder.npy"))[:L]
    Y1 = np.cumsum(data[:,4]*V)
    axs.plot(data[:,0], Y1, color = colors[i],  ls='-', lw = lw,  marker = mark[i], markersize = 10, markevery=100, label = 'DQ, '+ r'$\mathrm{BER=0.2}$',)
    i += 1

    data = np.load(os.path.join(rootdir, "MNIST_IID_epoch1_DQ_sr_flip0.3_adam_0.01_U100+10_bs64_2025-12-13-00:39:55/TraRecorder.npy"))[:L]
    Y1 = np.cumsum(data[:,4]*V)
    axs.plot(data[:,0], Y1, color = colors[i],  ls='-', lw = lw,  marker = mark[i], markersize = 10, markevery=100, label = 'DQ, '+ r'$\mathrm{BER=0.2}$',)
    i += 1

    axs.plot(data[:,0], data[:, 0]*V*8, color = colors[i],  ls='-', lw = 2, marker = mark[i], markersize = 10, markevery=100, label = r'$\mathrm{8bit}$',)
    i += 1

    axs.plot(data[:,0], data[:, 0]*V*4, color = colors[i], ls='-', lw = 2, marker = mark[i], markersize = 10, markevery=100, label = r'$\mathrm{4bit}$',)
    i += 1

    axs.plot(data[:,0], data[:, 0]*V*1, color = colors[i], ls='-', lw = 2, marker = mark[i], markersize = 10, markevery=100, label = r'$\mathrm{1bit}$',)
    i += 1

    ###########
    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    font2 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
    axs.set_xlabel( "通信轮数", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    axs.set_ylabel('累计通信负载(bits)', fontproperties=font2, )

    font2 = FontProperties(fname=fontpath+"simsun.ttf", size=24)
    legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2, borderpad = 0.1, labelspacing = 0.1)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')                         # 设置图例legend背景透明

    # x_major_locator = MultipleLocator(5)               # 把x轴的刻度间隔设置为1，并存在变量里
    # axs.xaxis.set_major_locator(x_major_locator)       # 把x轴的主刻度设置为1的倍数
    axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 25, width=3,)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels]  # 刻度值字号

    # axs.set_xlim(-0.2, 2)  #拉开坐标轴范围显示投影
    # axs.set_ylim(0.5, 1.01)  #拉开坐标轴范围显示投影

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    # [label.set_fontsize(16) for label in labels] #刻度值字号

    out_fig = plt.gcf()
    out_fig.savefig(f'{savedir}/CommOverHead.pdf' )
    plt.show()
    plt.close()
    return


MNIST_IID_841bit()
DynamicBitWidth()
CommOverHead()
#































































































































































































































































































