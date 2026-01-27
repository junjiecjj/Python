#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 22:09:44 2025

@author: jack
"""


import os
import sys


import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import numpy as np
# import torch
from matplotlib.font_manager import FontProperties
# from matplotlib.pyplot import MultipleLocator
# import socket, getpass
from scipy.signal import savgol_filter

# 获取当前系统用户目录
home = os.path.expanduser('~')
savedir = './CIFAR10'

fontpath = "/usr/share/fonts/truetype/windows/"

mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', '$\\clubsuit$', '$\\bowtie$', '$\\boxdot$', '$\\boxtimes$', '$\\boxminus$',  '$\\boxplus$', '$\\bigodot$','$\\bigoplus$', '$\\bigotimes$', '$\\bigstar$', '$\\blacksquare$', '$\circledR$', '$\circledS$',  '$\\bigtriangledown$','$\\bigtriangleup$','$\\blacktriangleleft$', '$\\blacktriangleright$', '$\\diamondsuit',   '$\heartsuit$', '$666$', '$\\varrho$', '$\\Omega$', '$\\nabla$', '$\\Upsilon$', '$\\mho$', '$\\Phi$',  '$\\divideontimes$', '$\\otimes$', '$\\odot$', '$\\ominus$', '$\\oplus$', '$\\oslash$', '$\\otimes$']

colors = ['#FF0000','#0000FF','#00FF00','#1E90FF','#4ea142','#FF00FF','#FFA500','#800080','#EE82EE','#00FFFF','#9932CC','#FF6347','#00CED1','#CD5C5C',  '#7B68EE','#808000']

lsty = [(0, (3, 10, 1, 10, 1, 10)), (0, (1, 1)), (0, (1, 2)), (0, (5, 1)), (0, (1, 10)), (0, (1, 2)),  (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),  '-', ':', '--', '-.', ]
alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

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

def CIFAR10_IID_DQvs4bit():
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    axins = axs.inset_axes((0.52, 0.5, 0.3, 0.32))
    L = 1000

    rootdir = f"{home}/FL_DQ/CIFAR10_IID/"
    data = np.load(os.path.join(rootdir, "CIFAR10_IID_epoch2_Perfect_adam_0.01_U100+10_bs64_2025-12-14-14:00:53/TraRecorder.npy"))[:L]
    Y1 = data[:,1]
    Y1 = savgol_filter(Y1, 20, 5) + 0.005
    axs.plot(data[:,0], Y1 , color = 'gray', linestyle= '-',lw = 3, label = 'Perfect',)
    axins.plot(data[:,0], Y1, color = 'gray', linestyle = '-', linewidth = 2)

    i=0
    data = np.load(os.path.join(rootdir, "CIFAR10_IID_epoch2_DQ_sr_flip0.0115_adam_0.01_U100+10_bs64_2025-12-24-09:52:46/TraRecorder.npy"))[:L]
    Ydq1 = data[:,1]
    Ydq1 = savgol_filter(Ydq1, 20, 5) # + 0.005
    axs.plot(data[:,0], Ydq1, color='#FF6347', lw = 2, linestyle='--' , label = "DQ+Free-Ride, SNR=1dB")
    axins.plot(data[:,0], Ydq1, color='#FF6347', linestyle = '--' , linewidth = 2)
    i += 1

    data = np.load(os.path.join(rootdir, "CIFAR10_IID_epoch2_DQ_sr_flip0.03898_adam_0.01_U100+10_bs64_2025-12-24-14:44:03/TraRecorder.npy"))[:L]
    Ydq2 = data[:,1]
    Ydq2 = savgol_filter(Ydq2, 20, 5) # + 0.005
    axs.plot(data[:,0], Ydq2, color='#FF6347', lw = 2, linestyle=(0,(3,1,1,1)) , label = "DQ+Free-Ride, SNR=0.75dB")
    axins.plot(data[:,0], Ydq2, color='#FF6347', linestyle =(0,(3,1,1,1)), linewidth = 2)
    i += 1

    data = np.load(os.path.join(rootdir, "CIFAR10_IID_epoch2_DQ_sr_flip0.16_adam_0.01_U100+10_bs64_2025-12-24-14:44:26/TraRecorder.npy"))[:L]
    Ydq3 = data[:,1]
    Ydq3 = savgol_filter(Ydq3, 20, 5) #+ 0.005
    axs.plot(data[:,0], Ydq3, color='#FF6347', lw = 2, linestyle=':', label = "DQ+Free-Ride, SNR=0dB")
    axins.plot(data[:,0], Ydq3, color='#FF6347', linestyle = ':', linewidth = 2)
    i += 1

    data = np.load(os.path.join(rootdir, "CIFAR10_IID_epoch2_4bits_sr_flip0.0115_adam_0.01_U100+10_bs64_2025-12-23-14:55:09/TraRecorder.npy"))[:L]
    Y4_1 = data[:,1]
    Y4_1 = savgol_filter(Y4_1, 20, 5)
    axs.plot(data[:,0], Y4_1, color = '#1E90FF', lw = 2, linestyle='--', label = "4bit+LDPC, SNR=1dB")
    axins.plot(data[:,0], Y4_1, color = '#1E90FF', linestyle = '--', linewidth = 2)
    i += 1

    data = np.load(os.path.join(rootdir, "CIFAR10_IID_epoch2_4bits_sr_flip0.03898_adam_0.01_U100+10_bs64_2025-12-23-14:55:27/TraRecorder.npy"))[:L]
    Y4_2 = data[:,1]
    Y4_2 = savgol_filter(Y4_2, 20, 5)
    axs.plot(data[:,0], Y4_2, color = '#1E90FF', lw = 2, linestyle=(0,(3,1,1,1)), label = "4bit+LDPC, SNR=0.75dB")
    axins.plot(data[:,0], Y4_2, color = '#1E90FF', linestyle = (0,(3,1,1,1)), linewidth = 2)
    i += 1

    data = np.load(os.path.join(rootdir, "CIFAR10_IID_epoch2_4bits_sr_flip0.16_adam_0.01_U100+10_bs64_2025-12-23-21:08:52/TraRecorder.npy"))[:L]
    Y4_3 = data[:,1]
    Y4_3 = savgol_filter(Y4_3, 20, 5)
    axs.plot(data[:,0], Y4_3, color = '#1E90FF', lw = 2, linestyle=':', label = "4bit+LDPC, SNR=0dB")
    axins.plot(data[:,0], Y4_3, color = '#1E90FF', linestyle = ':', linewidth = 2)
    i += 1

    ###########
    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    # font2 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
    axs.set_xlabel( "Round", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    axs.set_ylabel('Accuracy', fontproperties=font2, )

    # font2 = FontProperties(fname=fontpath+"simsun.ttf", size=20)
    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 18}
    legend1 = axs.legend(bbox_to_anchor = (0.4, 0.1), borderaxespad=0, edgecolor='black', prop=font2, borderpad = 0.1, labelspacing = 0.1)
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
    # axs.set_ylim(0.6, 1.01)  #拉开坐标轴范围显示投影

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    ###==================== mother and son ==================================
    ### 局部显示并且进行连线,
    zone_and_linked(axs, axins, L-500, L-450, data[:, 0] , [Y1, Ydq1, Ydq2, Ydq3, Y4_1, Y4_2,], 'bottom', x_ratio = 0.3, y_ratio = 0.2)
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
    out_fig.savefig(f'{savedir}/Fig_CIFAR10_IID_DQvs4bit_SNR.pdf' )
    plt.show()
    plt.close()
    return

CIFAR10_IID_DQvs4bit()






































































































































































































































































































