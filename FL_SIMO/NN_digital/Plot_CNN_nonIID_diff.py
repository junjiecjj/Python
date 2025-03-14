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
# import matplotlib
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
from matplotlib.pyplot import MultipleLocator
# 使用Savitzky-Golay 滤波器后得到平滑图线
from scipy.signal import savgol_filter


filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

# 一、移动平均滤波
def moving_average(signal, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')

# 二、中值滤波
def median_filter(signal, window_size):
    filtered_signal = np.zeros_like(signal)
    half_window = window_size // 2

    for i in range(half_window, len(signal) - half_window):
        window = signal[i - half_window : i + half_window + 1]
        filtered_signal[i] = np.median(window)

    return filtered_signal

def CNN_nonIID_small():
    # %% 画图
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    L = 800
    ## erf
    data = np.load("/home/jack/DigitalFL/NN/CNN_noIID_diff_epoch5_sgd_0.01_U100_bs64_2024-10-27-20:07:28/TraRecorder.npy")[:L]
    axs.plot(data[:,0], savgol_filter(data[:,1], 10, 3, mode= 'interp'), color = 'k', lw = 3, linestyle='--', marker = 'o', ms = 14, mfc = 'white', markevery = 100, label = 'Error-free',)

    ## 1-bit erf
    data = np.load("/home/jack/DigitalFL/NN/CNN_noIID_diff_epoch5_1bits_nr_sgd_0.01_U100_bs64_2024-10-27-16:49:55/TraRecorder.npy")[:L]
    axs.plot(data[:,0], savgol_filter(data[:,1], 10, 3, mode= 'interp'), color = 'r', lw = 3, linestyle='-',  label = '1-bit Error-free',)

    ## 1-bit 6dB, MIMO
    data = np.load("/home/jack/DigitalFL/NN/CNN_noIID_diff_epoch5_1bits_mimo6(dB)_sgd_0.01_U100_bs64_2024-10-28-13:27:17/TraRecorder.npy")[:L]
    axs.plot(data[:,0], savgol_filter(data[:,1], 10, 3, mode= 'interp'), color = 'b', lw = 3, linestyle='--', label = '1-bit w/o LDPC, 6 dB',)


    ## 1-bit 6dB, LDPC
    data = np.load("/home/jack/DigitalFL/NN/CNN_noIID_diff_epoch5_1bits_ldpc10(dB)_sgd_0.01_U100_bs64_2024-10-28-13:07:48/TraRecorder.npy")[:L]
    axs.plot(data[:,0], savgol_filter(data[:,1], 10, 3, mode= 'interp'), color = 'olive', lw = 3, linestyle='--', label = '1-bit w/ LDPC, 10 dB',)

    # axs.semilogy(APlst, sca_wo_res, color = 'r', lw = 3, linestyle='--',marker = 'd',ms = 14, label = 'SCA w/o RIS', )
    # axs.semilogy(APlst, sdr_res, color = 'b', lw = 3, linestyle='--',  marker = 'o',ms = 14, label = 'SDR w/ RIS',  )
    # axs.semilogy(APlst, dc_res, color = 'olive', lw = 3, linestyle='--', marker = 's',ms = 12, label = 'DC w/ RIS', )
    # axs.semilogy(APlst, dc_rand_res, color = 'olive', lw = 3, linestyle='--',  marker = '^', ms = 16, label = 'DC random',  )
    # axs.semilogy(APlst, dc_wo_res, color = 'olive', lw = 3, linestyle='--',  marker = '*', ms = 16, label = 'DC w/o RIS',  )

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    axs.set_xlabel( "Communication round", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    axs.set_ylabel('Test accuracy', fontproperties=font2, )

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 23}
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
    # axs.set_ylim(0.5, 1.06)  #拉开坐标轴范围显示投影

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    out_fig = plt.gcf()
    out_fig.savefig('./Figures/CNN_nonIID_small.eps' )
    out_fig.savefig('./Figures/CNN_nonIID_small.pdf' )
    plt.show()

def CNN_nonIID_Large_small_acc():
    # %% 画图
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    L = 620
    ## erf
    data = np.load("/home/jack/DigitalFL/NN_pathloss/CNN_noIID_diff_epoch2_sgd_0.01_U100_bs64_2024-10-30-12:05:11/TraRecorder.npy")[:L]
    axs.plot(data[:-2,0], moving_average(data[:,1], 5)[:-2], color = 'k', lw = 3, linestyle='--', marker = 'o', ms = 14, mfc = 'white', markevery = 100, label = 'Error-free',)

    ## 1-bit erf
    data = np.load("/home/jack/DigitalFL/NN_pathloss/CNN_noIID_diff_epoch2_1bits_nr_sgd_0.01_U100_bs64_2024-10-30-12:01:30/TraRecorder.npy")[:L]
    axs.plot(data[:-2,0], moving_average(data[:,1], 5)[:-2], color = 'b', lw = 3, linestyle='-',  label = '1-bit Error-free',)

    ## 1-bit -60dBm, MIMO
    data = np.load("/home/jack/DigitalFL/NN_pathloss/CNN_noIID_diff_epoch2_1bits_mimo-60(dBm)_sgd_0.01_U100_bs64_2024-10-30-13:18:03/TraRecorder.npy")[:L]
    axs.plot(data[:-2,0], moving_average(data[:,1], 5)[:-2], color = 'olive', lw = 3, linestyle='-', label = '1-bit w/o LDPC',)

    ## 1-bit 6dB, LDPC
    data = np.load("/home/jack/DigitalFL/NN_pathloss/CNN_noIID_diff_epoch2_1bits_ldpc-60(dBm)_sgd_0.01_U100_bs64_2024-10-30-13:28:10/TraRecorder.npy")[:L]
    axs.plot(data[:-2,0], moving_average(data[:,1], 5)[:-2], color = 'r', lw = 3, linestyle='-',   label = 'Proposed 1-bit w/ LDPC',)

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    axs.set_xlabel( "Communication Round", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    axs.set_ylabel('Test accuracy', fontproperties=font2, )
    # axs.set_title("CNN, non-IID", fontproperties=font2)
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
    # axs.set_ylim(0.3, 1.06)  #拉开坐标轴范围显示投影

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    out_fig = plt.gcf()
    out_fig.savefig('./Figures/CNN_nonIID_laregesmall_acc.eps' )
    out_fig.savefig('./Figures/CNN_nonIID_laregesmall_acc.pdf' )
    plt.show()

def CNN_nonIID_Large_small_loss():
    # %% 画图
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    L = 600
    ## erf
    data = np.load("/home/jack/DigitalFL/NN_pathloss/CNN_noIID_diff_epoch2_sgd_0.01_U100_bs64_2024-10-30-12:05:11/TraRecorder.npy")[:L]
    axs.plot(data[:-2,0], moving_average(data[:,2], 5)[:-2], color = 'k', lw = 3, linestyle='--', marker = 'o', ms = 14, mfc = 'white', markevery = 100, label = 'Error-free',)

    ## 1-bit erf
    data = np.load("/home/jack/DigitalFL/NN_pathloss/CNN_noIID_diff_epoch2_1bits_nr_sgd_0.01_U100_bs64_2024-10-30-12:01:30/TraRecorder.npy")[:L]
    axs.plot(data[:-2,0], moving_average(data[:,2], 5)[:-2], color = 'b', lw = 3, linestyle='-',  label = '1-bit Error-free',)

    ## 1-bit -60dBm, MIMO
    data = np.load("/home/jack/DigitalFL/NN_pathloss/CNN_noIID_diff_epoch2_1bits_mimo-60(dBm)_sgd_0.01_U100_bs64_2024-10-30-13:18:03/TraRecorder.npy")[:250]
    axs.plot(data[:,0], data[:,2], color = 'olive', lw = 3, linestyle='-', label = '1-bit w/o LDPC',)

    ## 1-bit 6dB, LDPC
    data = np.load("/home/jack/DigitalFL/NN_pathloss/CNN_noIID_diff_epoch2_1bits_ldpc-60(dBm)_sgd_0.01_U100_bs64_2024-10-30-13:28:10/TraRecorder.npy")[:L]
    axs.plot(data[:,0], data[:,2], color = 'r', lw = 3, linestyle='-', label = 'Proposed 1-bit w/ LDPC',)

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    axs.set_xlabel( "Communication Round", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
    axs.set_ylabel('Cross Entropy', fontproperties=font2, )
    # axs.set_title("CNN, non-IID", fontproperties=font2)
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
    # axs.set_ylim(0.3, 1.06)  #拉开坐标轴范围显示投影

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    out_fig = plt.gcf()
    out_fig.savefig('./Figures/CNN_nonIID_laregesmall_loss.eps' )
    out_fig.savefig('./Figures/CNN_nonIID_laregesmall_loss.pdf' )
    plt.show()

# CNN_nonIID_small()

CNN_nonIID_Large_small_acc()
CNN_nonIID_Large_small_loss()




































