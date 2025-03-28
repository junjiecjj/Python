#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:28:54 2024

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
MPA_SISO_large = np.array([[-50.00, 0.91666667, 0.33983667, 0.53588513],
[-55.00, 0.84000000, 0.29432617, 0.47143229],
[-60.00, 0.82352941, 0.23833550, 0.38973780],
[-65.00, 0.72608696, 0.17188632, 0.28967957],
[-70.00, 0.62781955, 0.10606595, 0.18559093],
[-75.00, 0.51117886, 0.04584882, 0.08346434],
[-77.00, 0.45719490, 0.02691506, 0.04977801],
[-80.00, 0.30474453, 0.00997709, 0.01869012],
[-85.00, 0.07660550, 0.00099179, 0.00190056],
[-90.00, 0.00664491, 0.00007764, 0.00014999],
[-95.00, 0.0005784062, 0.0000082845, 0.0000158995]
                        ])

MPA_LPDC_SIMO_large = np.array([[-50.00, 0.83333333, 0.33662722, 0.00000000],
                                [-55.00, 0.73538012, 0.28315744, 0.00000000],
                                [-60.00, 0.66402116, 0.22692936, 0.00000000],
                                [-65.00, 0.53978495, 0.16073379, 0.00000000],
                                [-70.00, 0.35836910, 0.08521521, 0.00000000],
                                [-75.00, 0.14804965, 0.02388653, 0.00000000],
                                [-77.00, 0.06653386, 0.01060575, 0.00000000],
                                [-80.00, 0.01388196, 0.00203333, 0.00000000],
                                [-85.00, 0.0010892512, 0.0001569922, 0.0000000000]
                        ])




def SCMA_SISO_large():
    lw = 2
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##=========================   ===============================
    lb = "MPA, large fading, SISO, w/o LDPC"
    axs.semilogy(MPA_SISO_large[:, 0], MPA_SISO_large[:, cols], color = 'k', ls = '-',  marker = 'o', mfc = 'none', ms = 18, label = lb,)

    # #=========================  ===============================
    lb = "MPA, large fading, SISO, w/ LDPC"
    axs.semilogy(MPA_LPDC_SIMO_large[:, 0], MPA_LPDC_SIMO_large[:, cols], color = 'r', ls='--', lw = 3, marker = '*', ms = 16,  mew = 2, label = lb)
    #=========================   ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "EPA, large fading, SIMO, w/ LDPC"
    # axs.semilogy(EPA_LPDC_SIMO_large[:, 0], EPA_LPDC_SIMO_large[:, cols], color = 'b', ls='--', lw = 3, marker = 'd', ms = 16,  mew = 2, label = lb)

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
        out_fig.savefig("./Figures/SCMA_SISOlarge_ser.eps")
        out_fig.savefig("./Figures/SCMA_SISOlarge_ser.png")
    elif cols == 2:
        out_fig.savefig( "./Figures/SCMA_SISOlarge_ber.eps")
        out_fig.savefig( "./Figures/SCMA_SISOlarge_ber.png")

    plt.show()
    plt.close()
    return

SCMA_SISO_large()












































