#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:12:38 2024

@author: jack
"""


import scipy
import numpy as np
# import scipy
# import cvxpy as cp
import matplotlib.pyplot as plt
# import math
import matplotlib
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
from matplotlib.pyplot import MultipleLocator
# 使用Savitzky-Golay 滤波器后得到平滑图线
from scipy.signal import savgol_filter

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

import socket, getpass
from scipy.signal import savgol_filter


filepath2 = '/home/jack/snap/'
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

def Cifar10_IID_1bit_flip_acc():
    # %% 画图
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    axins = axs.inset_axes((0.62, 0.46, 0.3, 0.32))
    L = 1000
    ## erf
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_sgd_0.01_U100+6_bs64_2025-01-14-14:15:05/TraRecorder.npy")[:L]
    Y1 = savgol_filter(data[:,1], 10, 3)
    axs.plot(data[:,0], Y1, color = 'k', lw = 1.3, linestyle='-', marker = 'o', ms = 18, mfc = 'white', markevery = 100, label = 'Perfect',)
    axins.plot(data[:,0], Y1, color = 'k', linestyle = '-', linewidth = 2)

    ## 1-bit erf
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_erf_sgd_0.01_U100+6_bs64_2025-01-14-22:30:24/TraRecorder.npy")[:L]
    Y2 = savgol_filter(data[:,1], 10, 3)
    axs.plot(data[:,0], Y2, color = '#E918B5', lw = 2, linestyle='--', label = '1-bit Error-free',)
    axins.plot(data[:,0], Y2, color = '#E918B5', linestyle = '--', linewidth = 2)

    ## 1-bit, 0.01ber
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_flip0.001_sgd_0.01_U100+6_bs64_2025-01-21-20:19:27/TraRecorder.npy")[:L]
    Y3 = savgol_filter(data[:,1], 10, 3)
    axs.plot(data[:,0], Y3, color = '#556B2F', lw = 2, linestyle='-', label = '1-bit, BER=10$^{-2}$',)
    axins.plot(data[:,0], Y3, color = '#556B2F', linestyle = '-', linewidth = 2)

    ## 1-bit, 0.1ber
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_flip0.1_sgd_0.01_U100+6_bs64_2025-01-15-09:44:54/TraRecorder.npy")[:L]
    Y4 = savgol_filter(data[:,1], 10, 3)
    axs.plot(data[:,0], Y4, color = 'b', lw = 2, linestyle='--', label = '1-bit, BER=0.1',)
    axins.plot(data[:,0], Y4, color = 'b', linestyle = '--', linewidth = 2)

    ## 1-bit, 0.2ber
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_flip0.2_sgd_0.01_U100+6_bs64_2025-01-15-09:44:58/TraRecorder.npy")[:L]
    Y5 = savgol_filter(data[:,1], 10, 3)
    axs.plot(data[:,0], Y5, color = 'g', lw = 2, linestyle='--', label = '1-bit, BER=0.2',)
    axins.plot(data[:,0], Y5, color = 'g', linestyle = '--', linewidth = 2)

    # ## 1-bit 0.3ber
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_flip0.3_sgd_0.01_U100+6_bs64_2025-01-15-11:20:42/TraRecorder.npy")[:L]
    Y6 = savgol_filter(data[:,1], 10, 3)
    axs.plot(data[:,0], Y6, color = '#CD853F', lw = 2, linestyle='--',  label = '1-bit, BER=0.3',)
    axins.plot(data[:,0], Y6, color = '#CD853F', linestyle = '--', linewidth = 2)

    # ## 1-bit 0.4ber
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_flip0.4_sgd_0.01_U100+6_bs64_2025-01-15-11:21:01/TraRecorder.npy")[:L]
    Y7 = savgol_filter(data[:,1], 10, 3)
    axs.plot(data[:,0], Y7, color = '#00BFFF', lw = 2, linestyle='--',  label = '1-bit, BER=0.4',)
    axins.plot(data[:,0], Y7, color = '#00BFFF', linestyle = '--', linewidth = 2)

    # ## 1-bit 0.5ber
    data1 = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_flip0.5_sgd_0.01_U100+6_bs64_2025-01-15-13:30:37/TraRecorder.npy")[:500]
    Y8 = savgol_filter(data1[:,1], 10, 3)
    axs.plot(data1[:,0], Y8, color = '#778899', lw = 2, linestyle='--',  label = '1-bit, BER=0.5',)
    # axins.plot(data[:,0], data[:,1], color = '#778899', linestyle = '--', linewidth = 2)

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    axs.set_xlabel( "Communication round", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    axs.set_ylabel('Test accuracy', fontproperties=font2, )
    # axs.set_title("CNN, IID", fontproperties=font2)

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 21}
    legend1 = axs.legend(loc='lower left', bbox_to_anchor=(0.15, 0.06), borderaxespad=0, edgecolor='black', prop=font2, borderpad = 0.1, labelspacing = 0.1)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')                         # 设置图例legend背景透明

    # x_major_locator = MultipleLocator(5)               # 把x轴的刻度间隔设置为1，并存在变量里
    # axs.xaxis.set_major_locator(x_major_locator)       # 把x轴的主刻度设置为1的倍数
    axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 25, width=3,)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(24) for label in labels]  # 刻度值字号

    # axs.set_xlim(-0.2, 2)  #拉开坐标轴范围显示投影
    # axs.set_ylim(0.1, 1.06)  #拉开坐标轴范围显示投影

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    ##==================== mother and son ==================================
    ## 局部显示并且进行连线,方法3
    zone_and_linked(axs, axins, 850, 900, data[:, 0] , [Y1, Y2, Y3, Y4, Y5], 'bottom', x_ratio = 0.3, y_ratio = 0.2)
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
    # out_fig.savefig('../Figures/Cifar10_IID_1bit_bitflip_acc.eps' )
    out_fig.savefig('../Figures/Cifar10_IID_1bit_bitflip_acc.pdf' )
    plt.show()
    return

def Cifar10_IID_1bit_flip_loss():
    # %% 画图
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    L = 1000
    ## erf
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_sgd_0.01_U100+6_bs64_2025-01-14-14:15:05/TraRecorder.npy")[:L]
    Y1 = data[:, 1]
    axs.plot(data[:,0], data[:,2], color = 'k', lw = 1.2, linestyle='-', marker = 'o', ms = 18, mfc = 'white', markevery = 100, label = 'Perfect',)

    ## 1-bit erf
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_erf_sgd_0.01_U100+6_bs64_2025-01-14-22:30:24/TraRecorder.npy")[:L]
    Y2 = data[:, 1]
    axs.plot(data[:,0], data[:,2], color = '#E918B5', lw = 2, linestyle='-', label = '1-bit Error-free',)

    ## 1-bit, 0.1ber
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_flip0.1_sgd_0.01_U100+6_bs64_2025-01-15-09:44:54/TraRecorder.npy")[:L]
    Y3 = data[:, 1]
    axs.plot(data[:,0], data[:,2], color = 'b', lw = 2, linestyle='--', label = '1-bit, BER=0.1',)

    ## 1-bit, 0.2ber
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_flip0.2_sgd_0.01_U100+6_bs64_2025-01-15-09:44:58/TraRecorder.npy")[:L]
    Y4 = data[:, 1]
    axs.plot(data[:,0], data[:,2], color = 'g', lw = 2, linestyle='--', label = '1-bit, BER=0.2',)

    # ## 1-bit 0.3ber
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_flip0.3_sgd_0.01_U100+6_bs64_2025-01-15-11:20:42/TraRecorder.npy")[:L]
    Y5 = data[:, 1]
    axs.plot(data[:,0], data[:,2], color = '#CD853F', lw = 2, linestyle='--',  label = '1-bit, BER=0.3',)

    # ## 1-bit 0.4ber
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_flip0.4_sgd_0.01_U100+6_bs64_2025-01-15-11:21:01/TraRecorder.npy")[:L]
    Y6 = data[:, 1]
    axs.plot(data[:,0], data[:,2], color = '#00BFFF', lw = 2, linestyle='--',  label = '1-bit, BER=0.4',)

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    axs.set_xlabel( "Communication round", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    axs.set_ylabel('Training loss', fontproperties=font2, )
    # axs.set_title("CNN, IID", fontproperties=font2)

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2, borderpad = 0.1, labelspacing = 0.1)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')                         # 设置图例legend背景透明

    # x_major_locator = MultipleLocator(5)               # 把x轴的刻度间隔设置为1，并存在变量里
    # axs.xaxis.set_major_locator(x_major_locator)       # 把x轴的主刻度设置为1的倍数
    axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 25, width=3,)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(24) for label in labels]  # 刻度值字号

    # axs.set_xlim(-0.2, 2)  #拉开坐标轴范围显示投影
    # axs.set_ylim(0.1, 1.06)  #拉开坐标轴范围显示投影

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    out_fig = plt.gcf()
    # out_fig.savefig('../Figures/Cifar10_IID_1bit_bitflip_loss.eps' )
    out_fig.savefig('../Figures/Cifar10_IID_1bit_bitflip_loss.pdf' )
    plt.show()
    return

def CIFAR10_IID_14bit_erf_acc():
    # %% 画图
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    axins = axs.inset_axes((0.52, 0.46, 0.3, 0.32))
    L = 1000

    ## erf
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_sgd_0.01_U100+6_bs64_2025-01-14-14:15:05/TraRecorder.npy")[:L]
    Y1 = savgol_filter(data[:,1], 10, 3)
    axs.plot(data[:,0], Y1 , color = 'k', linestyle= '-',lw = 1.2, marker = 'o', ms = 14, mfc = 'white',mew = 2, markevery = 100,  label = 'Perfect',)
    axins.plot(data[:,0], Y1, color = 'k', linestyle = '-', linewidth = 2)

    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_erf_sgd_0.01_U100+6_bs64_2025-01-14-22:30:24/TraRecorder.npy")[:L]
    Y2 = savgol_filter(data[:,1], 10, 3)
    axs.plot(data[:,0], Y2, color = '#E918B5', lw = 2, linestyle='--', label = '1-bit, Error-free',)
    axins.plot(data[:,0], Y2, color = '#E918B5', linestyle = '--', linewidth = 2)

    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_4bits_sr_erf_sgd_0.01_U100+6_bs64_2025-01-14-15:40:27/TraRecorder.npy")[:L]
    Y3 = savgol_filter(data[:,1], 10, 3)
    axs.plot(data[:,0], Y3, color = 'b' , lw = 2, linestyle='--', label = '4-bit, Error-free',)
    axins.plot(data[:,0], Y3, color = 'b', linestyle = '--', linewidth = 2)

    ###########
    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    axs.set_xlabel( "Communication round", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    axs.set_ylabel('Test accuracy', fontproperties=font2, )
    # axs.set_title("CNN, IID", fontproperties=font2)

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
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
    axs.set_ylim(0.3, 0.88)  #拉开坐标轴范围显示投影

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    ###==================== mother and son ==================================
    ### 局部显示并且进行连线,方法3
    zone_and_linked(axs, axins, 535, 550, data[:, 0] , [Y1, Y2, Y3, ], 'bottom', x_ratio = 0.3, y_ratio = 0.2)
    ## linewidth
    bw = 1
    axins.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axins.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axins.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axins.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    axins.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 26,  width = 1)
    labels = axins.get_xticklabels() + axins.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # [label.set_fontsize(16) for label in labels] #刻度值字号

    out_fig = plt.gcf()
    # out_fig.savefig('../Figures/Cifar10_IID_14bit_erf_acc.eps' )
    out_fig.savefig('../Figures/Cifar10_IID_14bit_erf_acc.pdf' )
    plt.show()

def Cifar10_BatchIID_K0_1bit_flip_acc():
    # %% 画图
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    axins = axs.inset_axes((0.62, 0.42, 0.3, 0.32))
    L = 1000
    ## erf
    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_flip0.1_sgd_0.01_U100+2_bs64_2025-01-15-22:02:03/TraRecorder.npy")[:L]
    Y1 = savgol_filter(data[:,1], 10, 3)
    axs.plot(data[:,0], Y1, color = 'k', linestyle= '-',lw = 2,   label = r'1-bit, BER=0.1, K$_0$=2',)
    axins.plot(data[:,0], Y1, color = 'k', linestyle = '-', linewidth = 2)

    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_flip0.1_sgd_0.01_U100+6_bs64_2025-01-15-09:44:54/TraRecorder.npy")[:L]
    Y2 = savgol_filter(data[:,1], 10, 3)
    axs.plot(data[:,0], Y2, color = '#E918B5', lw = 3, linestyle='--', label = r'1-bit, BER=0.1, K$_0$=6',)
    axins.plot(data[:,0], Y2, color = '#E918B5', linestyle = '--', linewidth = 2)

    data = np.load("/home/jack/FL_1bitJoint/CIFAR10_resnet20_IID/CIFAR10_IID_diff_epoch2_1bits_sr_flip0.1_sgd_0.01_U100+12_bs64_2025-01-15-22:02:17/TraRecorder.npy")[:L]
    Y3 = savgol_filter(data[:,1], 10, 3)
    axs.plot(data[:,0], Y3, color = 'b' , lw = 3, linestyle='--', label = r'1-bit, BER=0.1, K$_0$=12',)
    axins.plot(data[:,0], Y3, color = 'b', linestyle = '--', linewidth = 2)

    ###########
    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    axs.set_xlabel( "Communication round", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    axs.set_ylabel('Test accuracy', fontproperties=font2, )
    # axs.set_title("CNN, IID", fontproperties=font2)

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
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
    # axs.set_ylim(0.2, 1.06)  #拉开坐标轴范围显示投影

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    ##==================== mother and son ==================================
    ### 局部显示并且进行连线,方法3
    zone_and_linked(axs, axins, 850, 900, data[:, 0] , [Y1, Y2, Y3, ], 'bottom', x_ratio = 0.3, y_ratio = 0.2)
    ## linewidth
    bw = 1
    axins.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axins.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axins.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axins.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    axins.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 26,  width = 1)
    labels = axins.get_xticklabels() + axins.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # [label.set_fontsize(16) for label in labels] #刻度值字号

    out_fig = plt.gcf()
    # out_fig.savefig('../Figures/Cifar10_IID_1bit_flip_K0_acc.eps' )
    out_fig.savefig('../Figures/Cifar10_IID_1bit_flip_K0_acc.pdf' )
    plt.show()


Cifar10_IID_1bit_flip_acc()
Cifar10_IID_1bit_flip_loss()
CIFAR10_IID_14bit_erf_acc()
Cifar10_BatchIID_K0_1bit_flip_acc()
plt.close('all')





































