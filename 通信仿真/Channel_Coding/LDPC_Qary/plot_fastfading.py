#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:11:10 2024

@author: jack
"""



import os
import sys


import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib.patches import ConnectionPatch
import numpy as np
# import torch
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
import socket, getpass
from scipy.signal import savgol_filter

# 获取当前系统主机名
host_name = socket.gethostname()
# 获取当前系统用户名
user_name = getpass.getuser()
# 获取当前系统用户目录
user_home = os.path.expanduser('~')
home = os.path.expanduser('~')

# 本项目自己编写的库
# from option import args
sys.path.append("..")
# checkpoint
# import Utility



fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)

# mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', '$\\clubsuit$', '$\\bowtie$', '$\\boxdot$', '$\\boxtimes$', '$\\boxminus$',  '$\\boxplus$', '$\\bigodot$','$\\bigoplus$', '$\\bigotimes$', '$\\bigstar$', '$\\blacksquare$', '$\circledR$', '$\circledS$',  '$\\bigtriangledown$','$\\bigtriangleup$','$\\blacktriangleleft$', '$\\blacktriangleright$', '$\\diamondsuit',   '$\heartsuit$', '$666$', '$\\varrho$', '$\\Omega$', '$\\nabla$', '$\\Upsilon$', '$\\mho$', '$\\Phi$',  '$\\divideontimes$', '$\\otimes$', '$\\odot$', '$\\ominus$', '$\\oplus$', '$\\oslash$', '$\\otimes$']

color = ['#1E90FF','#FF6347','#00FF00','#0000FF','#4ea142','#FF00FF','#FFA500','#800080','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE','#808000']
lsty = [(0, (3, 10, 1, 10, 1, 10)), (0, (1, 1)), (0, (1, 2)), (0, (5, 1)), (0, (1, 10)), (0, (1, 2)),  (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),  '-', ':', '--', '-.', ]
alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']


# 第一组数据，第一列是Eb/N0或SNR, 第二列是BER，第三列是WER，下同。
Joint_fastfading_4 = np.array([[0.00, 1.00000000, 0.22974609, 0.00000000],
                        [1.00, 1.00000000, 0.20391276, 0.00000000],
                        [2.00, 1.00000000, 0.17126302, 0.00000000],
                        [3.00, 1.00000000, 0.12468750, 0.00000000],
                        [3.20, 0.97402597, 0.10669262, 0.00000000],
                        [3.40, 0.84269663, 0.07720879, 0.00000000],
                        [3.60, 0.39062500, 0.03249359, 0.00000000],
                        [3.80, 0.05584320, 0.00429232, 0.00000000],
                        [4.00, 0.0037257824, 0.0002743762, 0.0000],
                        [4.10, 0.0006524815, 0.0000486007, 0.00000],
                        [4.20, 0.0000623960, 0.0000048747, 0.00000],
                        ])

Joint_fastfading_2 = np.array([[0.00, 1.00000000, 0.14276693, 0.23450520833333333, 0.00000000, 25.0],
                                [0.50, 1.00000000, 0.12561198, 0.20567708333333334, 0.00000000, 25.0],
                                [1.00, 1.00000000, 0.10027344, 0.16825520833333332, 0.00000000, 25.0],
                                [1.50, 0.84269663, 0.06209950, 0.1063575316011236, 0.00000000, 23.76685393258427],
                                [2.00, 0.09804560, 0.00501832, 0.00876806799674267, 0.00000000, 11.50586319218241],
                                [2.50, 0.0001938736, 0.0000068159, 0.000024, 0.00000000, 5.23]
                                ])
Joint_blockfading_2 = np.array([[0.00, 0.61680328, 0.14722080, 0.2450531506147541, 0.00000000, 21.598360655737704],
                                [0.50, 0.58593750, 0.12818146, 0.21350860595703125, 0.00000000, 21.0625],
                                [1.00, 0.56179775, 0.12101694, 0.20212283473782772, 0.00000000, 19.88951310861423],
                                [1.50, 0.49833887, 0.11036778, 0.18298380398671096, 0.00000000, 18.51827242524917],
                                [2.00, 0.44378698, 0.09619141, 0.16417806952662722, 0.00000000, 17.618343195266274],
                                [2.50, 0.41346154, 0.08605555, 0.14639315762362637, 0.00000000, 16.673076923076923],
                                [3.00, 0.38461538, 0.07742638, 0.13671875, 0.00000000, 16.002564102564104],
                                [3.50, 0.35128806, 0.07023703, 0.12093365778688525, 0.00000000, 14.600702576112413],
                                [4.00, 0.32822757, 0.06682508, 0.1126230853391685, 0.00000000, 13.408096280087527],
                                ])





def SISO_4user():
    lw = 2
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##=========================   ===============================
    lb = "4 User, Fastfading"
    axs.semilogy(Joint_fastfading_4[:, 0], Joint_fastfading_4[:, cols], color = 'k', ls = '-',  marker = 'o', mfc = 'none', ms = 18, label = lb,)

    # #=========================  ===============================
    # lb = "MPA, large fading, SISO, w/ LDPC"
    # axs.semilogy(fastfading_4[:, 0], fastfading_4[:, cols], color = 'r', ls='--', lw = 3, marker = '*', ms = 16,  mew = 2, label = lb)
    #=========================   ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    # lb = "EPA, large fading, SIMO, w/ LDPC"
    # axs.semilogy(EPA_LPDC_SIMO_large[:, 0], EPA_LPDC_SIMO_large[:, cols], color = 'b', ls='--', lw = 3, marker = 'd', ms = 16,  mew = 2, label = lb)

    ##===========================================================
    # plt.gca().invert_xaxis()

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    # label
    font = {'family':'Times New Roman','style':'normal','size':35}
    axs.set_xlabel("SNR (dB)", fontproperties=font)
    if cols == 3:
        axs.set_ylabel( "SER",      fontproperties = font )# , fontdict = font1
    elif cols == 2:
        axs.set_ylabel( "BER",      fontproperties = font )# , fontdict = font1

    #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
    font1 = {'family':'Times New Roman','style':'normal','size':18, }
    legend1 = axs.legend(loc = 'upper right', borderaxespad = 0, edgecolor = 'black', prop = font1,)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    bw = 2.5
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 16, width = bw)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(30) for label in labels] #刻度值字号

    fontt = {'family':'Times New Roman','style':'normal','size':35 }
    plt.suptitle("4 User, [1024, 512], regular LDPC", fontproperties = fontt, )
    out_fig = plt.gcf()

    if cols == 1:
        out_fig.savefig("./Figures/fastfading_4user_ser.eps")
        out_fig.savefig("./Figures/fastfading_4user_ser.png")
    elif cols == 2:
        out_fig.savefig( "./Figures/fastfading_4user_ber.eps")
        out_fig.savefig( "./Figures/fastfading_4user_ber.png")

    plt.show()
    plt.close()
    return


def SISO_2user():
    lw = 2
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##=========================   ===============================
    lb = "2 User, Fastfading, Joint"
    axs.semilogy(Joint_fastfading_2[:, 0], Joint_fastfading_2[:, cols], color = 'k', ls = '-',  marker = 'o', mfc = 'none', ms = 18, label = lb,)

    # #=========================  ===============================
    lb = "2 User, Blockfading, Joint"
    axs.semilogy(Joint_blockfading_2[:, 0], Joint_blockfading_2[:, cols], color = 'r', ls='--', lw = 3, marker = '*', ms = 16,  mew = 2, label = lb)
    #=========================   ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    # lb = "EPA, large fading, SIMO, w/ LDPC"
    # axs.semilogy(EPA_LPDC_SIMO_large[:, 0], EPA_LPDC_SIMO_large[:, cols], color = 'b', ls='--', lw = 3, marker = 'd', ms = 16,  mew = 2, label = lb)

    ##===========================================================
    # plt.gca().invert_xaxis()

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    # label
    font = {'family':'Times New Roman','style':'normal','size':35}
    axs.set_xlabel("SNR (dB)", fontproperties=font)
    if cols == 1:
        axs.set_ylabel( "SER",      fontproperties = font )# , fontdict = font1
    elif cols == 2:
        axs.set_ylabel( "BER",      fontproperties = font )# , fontdict = font1

    #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
    font1 = {'family':'Times New Roman','style':'normal','size':18, }
    legend1 = axs.legend(loc = 'upper right', borderaxespad = 0, edgecolor = 'black', prop = font1,)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    bw = 2.5
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 16, width = bw)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(30) for label in labels] #刻度值字号

    fontt = {'family':'Times New Roman','style':'normal','size':35 }
    plt.suptitle("2 User, [1024, 512], regular LDPC", fontproperties = fontt, )
    out_fig = plt.gcf()

    if cols == 1:
        out_fig.savefig("./Figures/2user_ser.eps")
        out_fig.savefig("./Figures/2user_ser.png")
    elif cols == 2:
        out_fig.savefig( "./Figures/2user_ber.eps")
        out_fig.savefig( "./Figures/2user_ber.png")

    plt.show()
    plt.close()
    return


SISO_4user()
SISO_2user()







































