#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:45:56 2024

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

mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', '$\\clubsuit$', '$\\bowtie$', '$\\boxdot$', '$\\boxtimes$', '$\\boxminus$',  '$\\boxplus$', '$\\bigodot$','$\\bigoplus$', '$\\bigotimes$', '$\\bigstar$', '$\\blacksquare$', '$\circledR$', '$\circledS$',  '$\\bigtriangledown$','$\\bigtriangleup$','$\\blacktriangleleft$', '$\\blacktriangleright$', '$\\diamondsuit',   '$\heartsuit$', '$666$', '$\\varrho$', '$\\Omega$', '$\\nabla$', '$\\Upsilon$', '$\\mho$', '$\\Phi$',  '$\\divideontimes$', '$\\otimes$', '$\\odot$', '$\\ominus$', '$\\oplus$', '$\\oslash$', '$\\otimes$']

color = ['#1E90FF','#FF6347','#00FF00','#0000FF','#4ea142','#FF00FF','#FFA500','#800080','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE','#808000']
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


def LocalBatchIID():
    # %% 画图
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    axins = axs.inset_axes((0.62, 0.5, 0.3, 0.32))
    L = 800
    ## erf
    data = np.load("/home/jack/FL_1bitJoint/NN/MNIST_IID_diff_batchs3_sgd_0.01_U100+6_bs128_2024-12-16-15:26:49/TraRecorder.npy")[:L]
    Y1 = data[:, 1]
    axs.plot(data[:,0], data[:,1], color = 'k', linestyle= '-', marker = 'o', ms = 18, mfc = 'white',mew = 2, markevery = 100, label = 'Error-free',)
    axins.plot(data[:,0], data[:,1], color = 'k', linestyle = '-', linewidth = 2)

    ## 1-bit erf
    data = np.load("/home/jack/FL_1bitJoint/NN/MNIST_IID_diff_batchs3_1bits_sr_erf_sgd_0.01_U100+6_bs128_2024-12-16-15:27:31/TraRecorder.npy")[:L]
    Y2 = data[:, 1]
    axs.plot(data[:,0], data[:,1], color = 'r', lw = 3, linestyle='--', label = '1-bit Error-free',)
    axins.plot(data[:,0], data[:,1], color = 'r', linestyle = '--', linewidth = 2)

    ## 1-bit, 0.1ber
    data = np.load("/home/jack/FL_1bitJoint/NN/MNIST_IID_diff_batchs3_1bits_sr_flip0.1_sgd_0.01_U100+6_bs128_2024-12-16-15:27:59/TraRecorder.npy")[:L]
    Y3 = data[:, 1]
    axs.plot(data[:,0], data[:,1], color = 'b' , lw = 3, linestyle='--', label = '1-bit, BER=0.1',)
    axins.plot(data[:,0], data[:,1], color = 'b', linestyle = '--', linewidth = 2)

    # ## 1-bit 0.2ber
    data = np.load("/home/jack/FL_1bitJoint/NN/MNIST_IID_diff_batchs3_1bits_sr_flip0.2_sgd_0.01_U100+6_bs128_2024-12-16-16:13:05/TraRecorder.npy")[:L]
    Y3 = data[:, 1]
    axs.plot(data[:,0], data[:,1], color = 'g', lw = 3, linestyle='--', label = '1-bit, BER=0.2',)
    axins.plot(data[:,0], data[:,1], color = 'g', linestyle = '--', linewidth = 2)

    # ## 1-bit 0.3ber
    data = np.load("/home/jack/FL_1bitJoint/NN/MNIST_IID_diff_batchs3_1bits_sr_flip0.3_sgd_0.01_U100+6_bs128_2024-12-16-16:13:16/TraRecorder.npy")[:L]
    Y4 = data[:, 1]
    axs.plot(data[:,0], data[:,1], color = '#CD853F', lw = 3, linestyle='--',  label = '1-bit, BER=0.3',)
    axins.plot(data[:,0], data[:,1], color = '#CD853F', linestyle = '--', linewidth = 2)

    # ## 1-bit 0.4ber
    data = np.load("/home/jack/FL_1bitJoint/NN/MNIST_IID_diff_batchs3_1bits_sr_flip0.4_sgd_0.01_U100+6_bs128_2024-12-16-16:13:20/TraRecorder.npy")[:L]
    Y5 = data[:, 1]
    axs.plot(data[:,0], data[:,1], color = '#00BFFF', lw = 3, linestyle='--',  label = '1-bit, BER=0.4',)
    axins.plot(data[:,0], data[:,1], color = '#00BFFF', linestyle = '--', linewidth = 2)

    ###########
    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    axs.set_xlabel( "Communication round", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    axs.set_ylabel('Test accuracy', fontproperties=font2, )
    # axs.set_title("CNN, IID", fontproperties=font2)

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 26}
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
    axs.set_ylim(0.2, 1.06)  #拉开坐标轴范围显示投影


    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细


    ##==================== mother and son ==================================
    ### 局部显示并且进行连线,方法3
    zone_and_linked(axs, axins, 600, 700, data[:, 0] , [Y1, Y2, Y3, Y4, Y5], 'bottom', x_ratio = 0.3, y_ratio = 0.2)
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
    out_fig.savefig('./Figures/MNIST_IID_localBatch_acc.eps' )
    out_fig.savefig('./Figures/MNIST_IID_localBatch_acc.pdf' )
    plt.show()

def Large_small_loss():
    # %% 画图
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    L = 400
    # ## erf
    # data = np.load("/home/jack/DigitalFL/NN/CNN_IID_diff_epoch1_sgd_0.01_U100_bs128_2024-10-28-17:59:45/TraRecorder.npy")[:L]
    # axs.plot(data[:,0], savgol_filter(data[:,2], 5, 3, mode= 'nearest'), color = 'k', lw = 3, linestyle= 'none', marker = 'o', ms = 18, mfc = 'white',mew = 2, markevery = 50, label = 'Error-free',)

    # ## 1-bit erf
    # data = np.load("/home/jack/DigitalFL/NN/CNN_IID_diff_epoch1_1bits_nr_sgd_0.01_U100_bs128_2024-10-28-20:46:47/TraRecorder.npy")[:L]
    # axs.plot(data[:,0], savgol_filter(data[:,2], 5, 3, mode= 'nearest'), color = 'b', lw = 5, linestyle='--', label = '1-bit Error-free',)

    # ## 1-bit -65dBm, MIMO
    # data = np.load("/home/jack/DigitalFL/CNN_NormZF/CNN_IID_diff_epoch1_1bits_mimo-65(dBm)_sgd_0.01_U100_bs128_2024-11-02-15:13:09/TraRecorder.npy")[:L]
    # axs.plot(data[:,0], savgol_filter(data[:,2], 5, 3, mode= 'nearest'), color = '#008000' , lw = 3, linestyle='--', label = '1-bit w/o LDPC, -65dBm',)

    # # ## 1-bit -60dBm, LDPC
    # data = np.load("/home/jack/DigitalFL/CNN_NormZF/CNN_IID_diff_epoch1_1bits_mimo-60(dBm)_sgd_0.01_U100_bs128_2024-11-02-15:40:45/TraRecorder.npy")[:L]
    # axs.plot(data[:,0], savgol_filter(data[:,2], 5, 3, mode= 'nearest'), color = 'olive', lw = 3, linestyle='--', label = '1-bit w/o LDPC, -60dBm',)

    # # ## 1-bit -55dBm, LDPC
    # data = np.load("/home/jack/DigitalFL/CNN_NormZF/CNN_IID_diff_epoch1_1bits_mimo-55(dBm)_sgd_0.01_U100_bs128_2024-11-02-15:44:01/TraRecorder.npy")[:L]
    # axs.plot(data[:,0], savgol_filter(data[:,2], 5, 3, mode= 'nearest'), color = 'r', lw = 3, linestyle='-',  label = '1-bit w/o LDPC, -55dBm',)

    # # ## 1-bit -55dBm, LDPC
    # data = np.load("/home/jack/DigitalFL/CNN_NormZF/CNN_IID_diff_epoch1_1bits_ldpc-55(dBm)_sgd_0.01_U100_bs128_2024-11-02-16:05:11/TraRecorder.npy")[:L]
    # axs.plot(data[:,0], savgol_filter(data[:,2], 5, 3, mode= 'nearest'), color = 'r', lw = 3, linestyle='none', marker = 'o', ms = 12, markevery = 50, label = 'Proposed 1-bit w/ LDPC, -55dBm',)


    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    axs.set_xlabel( "Communication round", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    axs.set_ylabel('Cross Entropy', fontproperties=font2, )
    # axs.set_title("CNN, IID", fontproperties=font2)

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 26}
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

    # axs.set_xlim(-10, )  #拉开坐标轴范围显示投影
    # axs.set_ylim(0.1, 1.06)  #拉开坐标轴范围显示投影

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    out_fig = plt.gcf()
    out_fig.savefig('./Figures/MNIST_IID_laregesmall_loss.eps' )
    out_fig.savefig('./Figures/MNIST_IID_laregesmall_loss.pdf' )
    plt.show()

# Only_small()

LocalBatchIID()
# Large_small_loss()


plt.close('all')





































