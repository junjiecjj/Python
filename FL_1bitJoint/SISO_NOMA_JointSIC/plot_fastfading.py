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

sic_fastfading_4u_equallo = np.array([[-120.00, 1.00000000, 0.30444336, 0.59968750, 0.00000000, 50.000],
                                        [-125.00, 1.00000000, 0.24949219, 0.51539062, 0.00000000, 50.000],
                                        [-130.00, 0.89732143, 0.17398507, 0.39208984, 0.00000000, 47.326],
                                        [-135.00, 0.56741573, 0.14506891, 0.34493504, 0.00000000, 26.483],
                                        [-140.00, 0.35243056, 0.10589939, 0.25364855, 0.00000000, 16.420],
                                        [-145.00, 0.27322404, 0.08763181, 0.20746884, 0.00000000, 13.743],
                                        [-150.00, 0.29585799, 0.09345530, 0.22331500, 0.00000000, 14.124],
                                        [-155.00, 0.24052133, 0.07863411, 0.18655584, 0.00000000, 12.956],
                                        [-160.00, 0.27472527, 0.08820452, 0.21041166, 0.00000000, 13.330],
                                        [-165.00, 0.29585799, 0.09420361, 0.22576507, 0.00000000, 13.712],
                                        [-170.00, 0.30303030, 0.09613222, 0.23024384, 0.00000000, 14.065],
                                        ])

sic_fastfading_4u_powerallo = np.array([[-120.00, 1.00000000, 0.29964844, 0.60867187, 0.00000000, 50.000],
                                        [-125.00, 0.78906250, 0.18058777, 0.46371460, 0.00000000, 43.316],
                                        [-126.00, 0.54347826, 0.11640200, 0.36459749, 0.00000000, 34.106],
                                        [-128.00, 0.30487805, 0.06925555, 0.25362043, 0.00000000, 22.027],
                                        [-130.00, 0.25380711, 0.04426505, 0.17540451, 0.00000000, 17.774],
                                        [-132.00, 0.19607843, 0.01788450, 0.07013634, 0.00000000, 15.436],
                                        [-134.00, 0.00070510, 0.00017240, 0.00043813, 0.00000000, 5.739],
                                        [-135.00, 0.00049486, 0.00013551, 0.00033863, 0.00000000, 4.981],
                                        [-136, 0.00034305, 0.00010454, 0,0,0]
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
    lb = r"SIC, w/"
    axs.semilogy(sic_fastfading_4u_powerallo[:, 0], sic_fastfading_4u_powerallo[:, cols], color = 'b',ls = '--', lw = 2,  marker = 'v', mfc = 'none', ms = 12, mew = 2, label = lb)

    lb = r"SIC, w/o"
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
    font1 = {'family':'Times New Roman','style':'normal','size':22, }
    legend1 = axs.legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = font1, labelspacing = 0.1)
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

    # fontt = {'family':'Times New Roman','style':'normal','size':35 }
    # plt.suptitle("2 User, Fastfading, BPSK, [1024, 512]", fontproperties = fontt, )
    out_fig = plt.gcf()

    # if cols == 1:
    #     out_fig.savefig("./Figures/2user_fast_fer.eps")
    #     out_fig.savefig("./Figures/2userfast__fer.pdf")
    # elif cols == 2:
    #     out_fig.savefig( "./Figures/2user_fast_ber.eps")
    #     out_fig.savefig( "./Figures/2user_fast_ber.pdf")
    # elif cols == 3:
    #     out_fig.savefig( "./Figures/2user_fast_aggerr.eps")
    #     out_fig.savefig( "./Figures/2user_fast_aggerr.pdf")
    plt.show()
    plt.close()
    return

SISO_4user()

















