#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 16:26:49 2023

@author: jack
"""


import os
import sys


import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch
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

mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', '$\\clubsuit$', '$\\bowtie$', '$\\boxdot$', '$\\boxtimes$', '$\\boxminus$',  '$\\boxplus$', '$\\bigodot$','$\\bigoplus$', '$\\bigotimes$', '$\\bigstar$', '$\\blacksquare$', '$\circledR$', '$\circledS$',  '$\\bigtriangledown$','$\\bigtriangleup$','$\\blacktriangleleft$', '$\\blacktriangleright$', '$\\diamondsuit',   '$\heartsuit$', '$666$', '$\\varrho$', '$\\Omega$', '$\\nabla$', '$\\Upsilon$', '$\\mho$', '$\\Phi$',  '$\\divideontimes$', '$\\otimes$', '$\\odot$', '$\\ominus$', '$\\oplus$', '$\\oslash$', '$\\otimes$']

color = ['#1E90FF','#FF6347','#00FF00','#0000FF','#4ea142','#FF00FF','#FFA500','#800080','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE','#808000']
lsty = [(0, (3, 10, 1, 10, 1, 10)), (0, (1, 1)), (0, (1, 2)), (0, (5, 1)), (0, (1, 10)), (0, (1, 2)),  (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),  '-', ':', '--', '-.', ]
alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']


# 第一组数据，第一列是Eb/N0或SNR, 第二列是BER，第三列是WER，下同。
SC_noCRC = np.array([[0.00, 0.77916019, 0.28073970],
                    [0.25, 0.65834428, 0.22755831],
                    [0.50, 0.58802817, 0.19679798],
                    [0.75, 0.50250752, 0.17347355],
                    [1.00, 0.42065491, 0.13169081],
                    [1.25, 0.34719335, 0.10336538],
                    [1.50, 0.24367704, 0.07285080],
                    [1.75, 0.20685384, 0.06061623],
                    [2.00, 0.13986600, 0.03742235],
                    [2.25, 0.09474281, 0.02628002],
                    [2.50, 0.06339365, 0.01658785],
                    [2.75, 0.03666301, 0.00898509],
                    [3.00, 0.02393465, 0.00560446],
                    [3.25, 0.01460471, 0.00348538],
                    [3.50, 0.00734270, 0.00163644],
                    [3.75, 0.00401256, 0.00086548],
                    [4.00, 0.00195081, 0.00040654],
                    [4.25, 0.00099553, 0.00020284],
                    [4.50, 0.00047240, 0.00009229],
                    [4.75, 0.00023299, 0.00003768],
                    [5.00, 0.00009481, 0.00001483],
                    ])

SCL_noCRC_cplus = np.array([[1.000000,  0.2256317690,  0.0594187162],
                            [1.250000,  0.1702417433,  0.0426615169],
                            [1.500000,  0.1135589371,  0.0267466784],
                            [1.750000,  0.0758150114,  0.0156581691],
                            [2.000000,  0.0545911126,  0.0104251965],
                            [2.250000,  0.0362660477,  0.0063465584],
                            [2.500000,  0.0242565371,  0.0039197048],
                            [2.750000,  0.0142029315,  0.0021117984],
                            [3.000000,  0.0094553707,  0.0013488677],
                            [3.250000,  0.0056244868,  0.0007684455],
                            [3.500000,  0.0029577223,  0.0004078884],
                            [3.750000,  0.0017481051,  0.0002382886],
                            [4.000000,  0.0011028422,  0.0001504690],
                            [4.250000,  0.0005596604,  0.0000743299],
                            [4.500000,  0.0003030360,  0.0000382772],
                            [4.750000,  0.0001489296,  0.0000204592],
                            [5.000000,  0.0000710263,  0.0000089848],
                            [5.250000,  0.0000334672,  0.0000044313],
                            [5.500000,  0.0000154794,  0.0000020292]
                            ])


SCL_noCRC = np.array([[0.00, 0.58871915, 0.18762853],
                        [0.25, 0.46047794, 0.14293716],
                        [0.50, 0.38538462, 0.11980769],
                        [0.75, 0.32135985, 0.09261746],
                        [1.00, 0.22436185, 0.06162534],
                        [1.25, 0.18562431, 0.04695026],
                        [1.50, 0.12603774, 0.02909984],
                        [1.75, 0.08404630, 0.01824096],
                        [2.00, 0.05929696, 0.01159346],
                        [2.25, 0.03674906, 0.00619246],
                        [2.50, 0.02437600, 0.00415618],
                        [2.75, 0.01528045, 0.00244142],
                        [3.00, 0.00917835, 0.00134967],
                        [3.25, 0.00533853, 0.00080085],
                        [3.50, 0.00285443, 0.00040189],
                        [3.75, 0.00204581, 0.00027825],
                        [4.00, 0.00103688, 0.00014235],
                        [4.25, 0.00052541, 0.00007359],
                        [4.50, 0.00029735, 0.00004025],
                    ])


def SNR_berfer( ):  ## E = 10, B = 128
    lw = 2
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 1
    ##=============================== LDPC =========================================

    ##========================= hard ===============================
    lb = "SC, without CRC, python"
    axs.semilogy(SC_noCRC[:, 0], SC_noCRC[:, cols], '--*',  color = 'k', label = lb, ms = 22 )

    #========================= llr ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "SCL, without CRC, python"
    axs.semilogy(SCL_noCRC[:, 0], SCL_noCRC[:, cols], 'o', markerfacecolor='none',  markeredgewidth = 2, color='r', label = lb,  ms=22)

    #========================= llr ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "SCL, without CRC, C++"
    axs.semilogy(SCL_noCRC_cplus[:, 0], SCL_noCRC_cplus[:, cols], '-*', markeredgewidth = 2, color='b', label = lb,  ms=18)
    ##===========================================================
    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    # label
    font = {'family':'Times New Roman','style':'normal','size':35}
    axs.set_xlabel("SNR (dB)", fontproperties=font)
    if cols == 1:
        axs.set_ylabel( "FER",      fontproperties = font )# , fontdict = font1
    elif cols == 2:
        axs.set_ylabel( "BER",      fontproperties = font )# , fontdict = font1

    #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
    font1 = {'family':'Times New Roman','style':'normal','size':25, }
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

    # axs.set_xlim(0.05, 0.94)  #拉开坐标轴范围显示投影
    # axs.set_ylim(0.0, 1.001)  #拉开坐标轴范围显示投影
    # x_major_locator=MultipleLocator(0.1)
    # axs.xaxis.set_major_locator(x_major_locator)
    # y_major_locator=MultipleLocator(0.1)
    # axs.yaxis.set_major_locator(y_major_locator)

    fontt = {'family':'Times New Roman','style':'normal','size':20 }
    plt.suptitle("Polar, [128,64], L = 8", fontproperties = fontt, )
    out_fig = plt.gcf()

    # out_fig.savefig(os.path.join("/home/jack/FedAvg_DataResults/results/", f"SNR_berfer.eps") )
    if cols == 1:
        out_fig.savefig(os.path.join("./SNR_fer.eps") )
        out_fig.savefig(os.path.join("./SNR_fer.png") )
    elif cols == 2:
        out_fig.savefig(os.path.join("./SNR_ber.eps") )
        out_fig.savefig(os.path.join("./SNR_ber.png") )

    plt.show()
    plt.close()
    return




SNR_berfer()












































