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



def LocalBatchIID():
    # %% 画图
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    L = 1000
    ## erf
    data = np.load("/home/jack/FL_1bitJoint/NN/CIFAR10_IID_diff_batchs15_sgd_0.01_U100+6_bs32_2024-12-16-17:16:49/TraRecorder.npy")[:L]
    axs.plot(data[:,0], data[:,1], color = 'k', lw = 1.2, linestyle='-', marker = 'o', ms = 18, mfc = 'white', markevery = 100, label = 'Error-free',)

    ## 1-bit erf
    data = np.load("/home/jack/FL_1bitJoint/NN/CIFAR10_IID_diff_batchs15_1bits_sr_erf_sgd_0.01_U100+6_bs32_2024-12-16-17:35:50/TraRecorder.npy")[:L]
    axs.plot(data[:,0], data[:,1], color = '#E918B5', lw = 2, linestyle='-', label = '1-bit Error-free',)

    ## 1-bit, 0.1ber
    data = np.load("/home/jack/FL_1bitJoint/NN/CIFAR10_IID_diff_batchs15_1bits_sr_flip0.1_sgd_0.01_U100+6_bs32_2024-12-16-18:06:33/TraRecorder.npy")[:L]
    axs.plot(data[:,0], data[:,1], color = 'b', lw = 2, linestyle='--', label = '1-bit, BER=0.1',)

    ## 1-bit, 0.2ber
    data = np.load("/home/jack/FL_1bitJoint/NN/CIFAR10_IID_diff_batchs15_1bits_sr_flip0.2_sgd_0.01_U100+6_bs32_2024-12-16-18:09:49/TraRecorder.npy")[:L]
    axs.plot(data[:,0], data[:,1], color = 'g', lw = 2, linestyle='--', label = '1-bit, BER=0.2',)

    # ## 1-bit 0.3ber
    data = np.load("/home/jack/FL_1bitJoint/NN/CIFAR10_IID_diff_batchs15_1bits_sr_flip0.3_sgd_0.01_U100+6_bs32_2024-12-16-19:29:00/TraRecorder.npy")[:L]
    Y4 = data[:, 1]
    axs.plot(data[:,0], data[:,1], color = '#CD853F', lw = 2, linestyle='--',  label = '1-bit, BER=0.3',)
    # axins.plot(data[:,0], data[:,1], color = '#CD853F', linestyle = '--', linewidth = 2)

    # ## 1-bit 0.4ber
    data = np.load("/home/jack/FL_1bitJoint/NN/CIFAR10_IID_diff_batchs15_1bits_sr_flip0.4_sgd_0.01_U100+6_bs32_2024-12-16-19:29:05/TraRecorder.npy")[:L]
    Y5 = data[:, 1]
    axs.plot(data[:,0], data[:,1], color = '#00BFFF', lw = 2, linestyle='--',  label = '1-bit, BER=0.4',)
    # axins.plot(data[:,0], data[:,1], color = '#00BFFF', linestyle = '--', linewidth = 2)

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
    # axs.set_ylim(0.1, 1.06)  #拉开坐标轴范围显示投影

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    out_fig = plt.gcf()
    out_fig.savefig('./Figures/Cifar10_IID_1bit_bitflip_acc.eps' )
    out_fig.savefig('./Figures/Cifar10_IID_1bit_bitflip_acc.pdf' )
    plt.show()
    return

def Large_small_loss():
    # %% 画图
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    L = 500
    ## erf
    data = np.load("/home/jack/FL_1bitJoint/NN/CIFAR10_IID_diff_batchs15_sgd_0.01_U100+6_bs32_2024-12-16-17:16:49/TraRecorder.npy")[:L]
    axs.plot(data[:,0], data[:,2], color = 'k', lw = 1, linestyle='-', marker = 'o', ms = 18, mfc = 'white', markevery = 100, label = 'Error-free',)

    ## 1-bit erf
    data = np.load("/home/jack/FL_1bitJoint/NN/CIFAR10_IID_diff_batchs15_1bits_sr_erf_sgd_0.01_U100+6_bs32_2024-12-16-17:35:50/TraRecorder.npy")[:L]
    axs.plot(data[:,0], data[:,2], color = '#E918B5', lw = 2, linestyle='-', label = '1-bit Error-free',)

    ## 1-bit, 0.1ber
    data = np.load("/home/jack/FL_1bitJoint/NN/CIFAR10_IID_diff_batchs15_1bits_sr_flip0.1_sgd_0.01_U100+6_bs32_2024-12-16-18:06:33/TraRecorder.npy")[:L]
    axs.plot(data[:,0], data[:,2], color = 'b', lw = 2, linestyle='--', label = '1-bit, BER=0.1',)

    ## 1-bit, 0.2ber
    data = np.load("/home/jack/FL_1bitJoint/NN/CIFAR10_IID_diff_batchs15_1bits_sr_flip0.2_sgd_0.01_U100+6_bs32_2024-12-16-18:09:49/TraRecorder.npy")[:L]
    axs.plot(data[:,0], data[:,2], color = 'g', lw = 2, linestyle='--', label = '1-bit, BER=0.2',)

    # ## 1-bit 0.3ber
    data = np.load("/home/jack/FL_1bitJoint/NN/CIFAR10_IID_diff_batchs15_1bits_sr_flip0.3_sgd_0.01_U100+6_bs32_2024-12-16-19:29:00/TraRecorder.npy")[:L]
    Y4 = data[:, 1]
    axs.plot(data[:,0], data[:,2], color = '#CD853F', lw = 2, linestyle='--',  label = '1-bit, BER=0.3',)
    # axins.plot(data[:,0], data[:,1], color = '#CD853F', linestyle = '--', linewidth = 2)

    # ## 1-bit 0.4ber
    data = np.load("/home/jack/FL_1bitJoint/NN/CIFAR10_IID_diff_batchs15_1bits_sr_flip0.4_sgd_0.01_U100+6_bs32_2024-12-16-19:29:05/TraRecorder.npy")[:L]
    Y5 = data[:, 1]
    axs.plot(data[:,0], data[:,2], color = '#00BFFF', lw = 2, linestyle='--',  label = '1-bit, BER=0.4',)
    # axins.plot(data[:,0], data[:,1], color = '#00BFFF', linestyle = '--', linewidth = 2)


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

    # axs.set_xlim(-0.2, 2)  #拉开坐标轴范围显示投影
    # axs.set_ylim(0.1, 1.06)  #拉开坐标轴范围显示投影

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    out_fig = plt.gcf()
    out_fig.savefig('./Figures/Cifar10_IID_1bit_bitflip_loss.eps' )
    out_fig.savefig('./Figures/Cifar10_IID_1bit_bitflip_loss.pdf' )
    plt.show()
    return


LocalBatchIID()
Large_small_loss()


plt.close('all')





































