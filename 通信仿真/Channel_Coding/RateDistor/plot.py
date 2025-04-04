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
data = np.array([[0.000, 1.00000000,   0.49999023,   1.00000000],
                 [0.050, 1.00000000, 0.04435352, 50.00000000],
                 [0.100, 1.00000000, 0.01178711, 50.00000000],
                 [0.150, 0.0593220339, 0.0000579317, 50],
                 [0.200, 0.00000000,   0.0000000000, 50],
                 [0.250, 0.00000000,   0.0000000000, 50],
                 [0.300, 0.00000000,   0.0000000000, 50],
                 [0.350, 0.00000000,   0.0000000000, 50],
                 [0.400, 0.00000000,   0.0000000000, 50],
                 [0.450, 0.00000000,   0.0000000000, 50],
                 [0.500, 1.00000000,   0.49999023,   1.00000000],
                 ])

def RateDistortion():
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##=========================   ===============================
    lb = "[1024, 512]"
    axs.plot(data[:, 0], data[:, cols], color = 'r', ls = '-',  marker = 'o', mfc = 'none', ms = 18, label = lb,)

    lb = "2 User, Fastfading, Seperat"
    # axs.semilogy(Sep_fastfading_2[:, 0], Sep_fastfading_2[:, cols], color = 'r', ls = '-',  marker = 'd', mfc = 'none', ms = 18, label = lb,)

    # #=========================  ===============================
    lb = "2 User, Blockfading, Joint"
    # axs.semilogy(Joint_blockfading_2[:, 0], Joint_blockfading_2[:, cols], color = 'b', ls='--', lw = 3, marker = 'o', mfc = 'none', ms = 16,  mew = 2, label = lb)

    lb = "2 User, Blockfading, Seperat"
    # axs.semilogy(Sep_blockfading_2[:, 0], Sep_blockfading_2[:, cols], color = 'b', ls='--', lw = 3, marker = 'd', mfc = 'none', ms = 16,  mew = 2, label = lb)
    #=========================   ===============================

    lb = "2 User, Fastfading, Joint, mess"
    # axs.semilogy(Joint_fastfading_2_mess2[:, 0], Joint_fastfading_2_mess2[:, cols], color = 'k', ls='--', lw = 3, marker = 'd', mfc = 'none', ms = 16,  mew = 2, label = lb)
    #=========================   ===============================

    ##===========================================================
    # plt.gca().invert_xaxis()

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    # label
    font = {'family':'Times New Roman','style':'normal','size':35}
    axs.set_xlabel("P [0, 0.5]", fontproperties=font)
    if cols == 1:
        axs.set_ylabel( "FER",      fontproperties = font )# , fontdict = font1
    elif cols == 2:
        axs.set_ylabel( "BER",      fontproperties = font )# , fontdict = font1
    elif cols == 3:
        axs.set_ylabel( "Iter",      fontproperties = font )# , fontdict = font1
    #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
    font1 = {'family':'Times New Roman','style':'normal','size':22, }
    legend1 = axs.legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = font1,)
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
    # plt.suptitle("2 User, BPSK, [1024, 512], regular LDPC", fontproperties = fontt, )
    out_fig = plt.gcf()

    if cols == 1:
        out_fig.savefig("./Figures/fer.eps")
        out_fig.savefig("./Figures/fer.pdf")
        out_fig.savefig("./Figures/fer.png")
    elif cols == 2:
        out_fig.savefig( "./Figures/ber.eps")
        out_fig.savefig( "./Figures/ber.pdf")
        out_fig.savefig( "./Figures/ber.png")
    elif cols == 3:
        out_fig.savefig( "./Figures/iter.eps")
        out_fig.savefig( "./Figures/iter.pdf")
        out_fig.savefig( "./Figures/iter.png")
    plt.show()
    plt.close()
    return


# SISO_4user()
RateDistortion()







































