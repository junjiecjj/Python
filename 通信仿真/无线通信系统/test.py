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

sic_fastfading_4u_equallo = np.array([[-124.00, 0.50000000, 0.11328125, 0.35023437, 0.00000000, 26.800],
[-126.00, 0.50000000, 0.09238281, 0.28876953, 0.00000000, 26.500],
[-128.00, 0.49019608, 0.07021676, 0.22370941, 0.00000000, 26.096],
[-130.00, 0.27932961, 0.02955602, 0.09718706, 0.00000000, 18.183],
[-132.00, 0.04344049, 0.00391431, 0.01286076, 0.00000000, 7.341],
[-134.00, 0.00258705, 0.00020959, 0.00069396, 0.00000000, 4.191],
[-136.00, 0.00007913, 0.0000061938, 0.00069396, 0.000020, 3.44],
                                        ])

sic_fastfading_4u_powerallo = np.array([[-124.00, 0.50000000, 0.12586914, 0.40748047, 0.00000000, 26.258],
[-126.00, 0.40000000, 0.08693359, 0.31692188, 0.00000000, 23.454],
[-128.00, 0.25125628, 0.05881213, 0.23500314, 0.00000000, 15.505],
[-130.00, 0.25000000, 0.04337891, 0.17351563, 0.00000000, 14.606],
[-132.00, 0.18867925, 0.01680793, 0.06723172, 0.00000000, 12.730],
[-134.00, 0.0000539043, 0.0000032266, 0.000013, 0.000000, 3.46],
                                        ])

Joint_fastfading_4u_w_powdiv = np.array([[-120.00, 1.00000000, 0.27999023, 0.52195313, 0.00000000, 12.500],
                                        [-125.00, 1.00000000, 0.14413086, 0.31226562, 0.00000000, 12.500],
                                        [-126.00, 0.28248588, 0.02390095, 0.05561441, 0.00000000, 7.637],
                                        [-126.20, 0.03148496, 0.00231842, 0.00563053, 0.00000000, 4.477],
                                        [-126.40, 0.00132541, 0.00009629, 0.00024484, 0.00000000, 3.105],

                                        ])


def SISO_4user():
    width = 8
    high  = 6
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##=========================   ===============================
    lb = "Proposed"
    axs.semilogy(Joint_fastfading_4u_w_powdiv[:, 0], Joint_fastfading_4u_w_powdiv[:, cols], color = 'r', ls = '--',  marker = '*', mfc = 'none', ms = 16, mew = 2, label = lb, zorder = 12)

    # #=========================  ===============================
    lb = r"SIC, w/ power allo"
    axs.semilogy(sic_fastfading_4u_powerallo[:, 0], sic_fastfading_4u_powerallo[:, cols], color = 'b',ls = '--', lw = 2,  marker = 'v', mfc = 'none', ms = 12, mew = 2, label = lb)

    lb = r"SIC, w/o power allo"
    axs.semilogy(sic_fastfading_4u_equallo[:, 0], sic_fastfading_4u_equallo[:, cols], color = 'b', ls = '--', lw = 2,  marker = 'o', mfc = 'none', ms = 12, mew = 2, label = lb,)

    ##===========================================================
    plt.gca().invert_xaxis()

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    # label
    font = {'family':'Times New Roman','style':'normal','size':35}
    axs.set_xlabel("$N_0$ (dBm/Hz)", fontproperties=font, labelpad = 0.2 )
    if cols == 1:
        axs.set_ylabel( "FER",      fontproperties = font, labelpad = 0.2  )# , fontdict = font1
    elif cols == 2:
        axs.set_ylabel( "BER",      fontproperties = font, labelpad = 0.2  )# , fontdict = font1
    elif cols == 3:
        axs.set_ylabel( "Aggregation error rate",      fontproperties = font )# , fontdict = font1
    #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
    font1 = {'family':'Times New Roman','style':'normal','size':26, }
    legend1 = axs.legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = font1, )
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    bw = 2
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 16, width = bw)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(30) for label in labels] #刻度值字号

    # fontt = {'family':'Times New Roman','style':'normal','size':35 }
    # plt.suptitle("2 User, Fastfading, BPSK, [1024, 512]", fontproperties = fontt, )
    out_fig = plt.gcf()

    # if cols == 1:
    #     out_fig.savefig("./Figures/4user_fast_fer.eps")
    #     out_fig.savefig("./Figures/4user_fast_fer.pdf")
    # elif cols == 2:
    #     # out_fig.savefig( "./Figures/4user_fast_ber.eps")
    #     out_fig.savefig( "./Figures/4user_fast_ber.pdf")
    # elif cols == 3:
    #     out_fig.savefig( "./Figures/4user_fast_aggerr.eps")
    #     out_fig.savefig( "./Figures/4user_fast_aggerr.pdf")
    plt.show()
    plt.close()
    return

SISO_4user()

















