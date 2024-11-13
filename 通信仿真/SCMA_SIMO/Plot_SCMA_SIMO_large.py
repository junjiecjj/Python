#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:12:29 2024

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
MPA_SIMO_large = np.array([[-50.00, 0.83333333, 0.27278001, 0.44084545],
                        [-55.00, 0.80608974, 0.20960725, 0.34783153],
                        [-60.00, 0.67338710, 0.14225785, 0.24487042],
                        [-65.00, 0.58217593, 0.07530156, 0.13572862],
                        [-70.00, 0.42991453, 0.02447082, 0.04641426],
                        [-75.00, 0.21576227, 0.00264454, 0.00521170],
                        [-77.00, 0.11044974, 0.00062391, 0.00124008],
                        [-80.00, 0.0107687210, 0.0000307400, 0.0000608329]
                        ])

EPA_SIMO_large = np.array([[-50.00, 0.83333333, 0.27297339, 0.44087124],
                            [-55.00, 0.80608974, 0.20961351, 0.34784405],
                            [-60.00, 0.67338710, 0.14232348, 0.24492293],
                            [-65.00, 0.58508159, 0.07540701, 0.13599942],
                            [-70.00, 0.43264249, 0.02459453, 0.04664224],
                            [-75.00, 0.21744792, 0.00266181, 0.00524224],
                            [-77.00, 0.11193029, 0.00063184, 0.00125234],
                            [-80.00, 0.01171601, 0.00003298, 0.00006376],
                        ])

EPA_LPDC_SIMO_large = np.array([[-50.00, 0.67607527, 0.26029591, 0.44135637],
                                [-55.00, 0.65364583, 0.20210775, 0.34868876],
                                [-60.00, 0.49313725, 0.13041896, 0.24489124],
                                [-65.00, 0.30873309, 0.05880516, 0.13666590],
                                [-70.00, 0.03955471, 0.00406878, 0.04621733],
                                [-75.00, 0.0000174301, 0.0000013617, 0.0053003861],
                        ])


def SCMAdetector_SISO():
    lw = 2
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##=========================   ===============================
    lb = "MPA, large fading, SIMO, w/o LDPC"
    axs.semilogy(MPA_SIMO_large[:, 0], MPA_SIMO_large[:, cols], color = 'k', ls = '-',  marker = 'o', mfc = 'none', ms = 18, label = lb,)

    # #=========================  ===============================
    lb = "EPA, large fading, SIMO, w/o LDPC"
    axs.semilogy(EPA_SIMO_large[:, 0], EPA_SIMO_large[:, cols], color = 'r', ls='--', lw = 3, marker = '*', ms = 16,  mew = 2, label = lb)
    #=========================   ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "EPA, large fading, SIMO, w/ LDPC"
    axs.semilogy(EPA_LPDC_SIMO_large[:, 0], EPA_LPDC_SIMO_large[:, cols], color = 'b', ls='--', lw = 3, marker = 'd', ms = 16,  mew = 2, label = lb)
    #=========================   ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    # lb = "Large-small"
    # axs.semilogy(largesmall[:, 0], largesmall[:, cols], color = 'r', ls='-', lw = 3, marker = 'o', ms = 25,  mfc = 'none', mew = 2, label = lb)
    # #=========================   ===============================
    # lb = "Fast fading, SIMO, w/o LDPC"
    # axs.semilogy(SIMO_fastfading[:, 0], SIMO_fastfading[:, cols], color = '#1E90FF', ls='--', lw = 3, marker = 'D', ms = 13, mfc = 'none', mew = 2, label = lb)

    # # #=========================  ===============================
    # lb = "Fast fading, EPA, SIMO, w/o LDPC"
    # axs.semilogy(EPA_SIMO_fastfading[:, 0], EPA_SIMO_fastfading[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 'o', ms = 14, mfc = 'none',  mew = 2, label = lb)

    # # #=========================  ===============================
    # lb = "Fast fading, EPA, SIMO, w/ LDPC"
    # axs.semilogy(EPA_LDPC_SIMO_fastfading[:, 0], EPA_LDPC_SIMO_fastfading[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 'd', ms = 14, mfc = 'none',  mew = 2, label = lb)

    # #========================= ===============================
    # lb = " "
    # axs.semilogy(SIC_norm_mmse_Hf[:, 0], SIC_norm_mmse_Hf[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 's', ms = 20, mfc = 'none', mew = 2, label = lb)

    # #=========================  ===============================
    # lb = " "
    # axs.semilogy(SIC_norm_zf_Hf[:, 0], SIC_norm_zf_Hf[:, cols], color = '#FFA500', ls='--', lw = 3, marker = '1', ms = 16, mew = 3, label = lb)
    # ========================= ===============================
    # lb = " "
    # axs.semilogy(ML[:, 0], ML[:, cols], color = '#FF00FF', ls='-', lw = 3, marker = 'H', ms = 18,  mfc = 'none', mew = 2, label = lb)
    ##===========================================================
    plt.gca().invert_xaxis()

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    # label
    font = {'family':'Times New Roman','style':'normal','size':35}
    axs.set_xlabel("Noise power (dBm)", fontproperties=font)
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
    plt.suptitle("SCMA, 512", fontproperties = fontt, )
    out_fig = plt.gcf()

    if cols == 3:
        out_fig.savefig("./Figures/SCMA_SIMO_ser.eps")
        out_fig.savefig("./Figures/SCMA_SIMO_ser.png")
    elif cols == 2:
        out_fig.savefig( "./Figures/SCMA_SIMO_ber.eps")
        out_fig.savefig( "./Figures/SCMA_SIMO_ber.png")

    plt.show()
    plt.close()
    return

SCMAdetector_SISO()












































