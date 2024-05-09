#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 16:26:49 2023

@author: jack
"""


import os
import sys


# import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib.patches import ConnectionPatch
import numpy as np
# import torch
from matplotlib.font_manager import FontProperties
# from matplotlib.pyplot import MultipleLocator
import socket, getpass
# from scipy.signal import savgol_filter

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
BPSK = np.array([[-10.00, 1.00000000, 0.26856785],
                [-9.00, 1.00000000, 0.23440928],
                [-8.00, 1.00000000, 0.20777186],
                [-7.00, 1.00000000, 0.19743027],
                [-6.00, 1.00000000, 0.15951113],
                [-5.00, 1.00000000, 0.12848637],
                [-4.00, 1.00000000, 0.10153557],
                [-3.00, 1.00000000, 0.08022563],
                [-2.00, 1.00000000, 0.06675024],
                [-1.00, 1.00000000, 0.03823253],
                [0.00, 1.00000000, 0.02319022],
                [1.00, 1.00000000, 0.01410216],
                [2.00, 1.00000000, 0.00532748],
                [3.00, 1.00000000, 0.00282043],
                [4.00, 1.00000000, 0.00062676],
                [5.00, 1.00000000, 0.00031338],
                # [6.00, 0.00000000, 0.00000000],
                # [7.00, 0.00000000, 0.00000000],
                # [8.00, 0.00000000, 0.00000000],
                # [9.00, 0.00000000, 0.00000000],
                # [10.00, 0.00000000, 0.00000000],
                # [11.00, 0.00000000, 0.00000000],
                    ])

QPSK = np.array([[-10.00, 1.00000000, 0.33155248],
                [-9.00, 1.00000000, 0.31175361],
                [-8.00, 1.00000000, 0.28724073],
                [-7.00, 1.00000000, 0.26587052],
                [-6.00, 1.00000000, 0.24167190],
                [-5.00, 1.00000000, 0.21275927],
                [-4.00, 1.00000000, 0.18541798],
                [-3.00, 1.00000000, 0.15304840],
                [-2.00, 1.00000000, 0.13387806],
                [-1.00, 1.00000000, 0.11093652],
                [0.00, 1.00000000, 0.07353865],
                [1.00, 1.00000000, 0.05876807],
                [2.00, 1.00000000, 0.04588309],
                [3.00, 1.00000000, 0.02199874],
                [4.00, 1.00000000, 0.01351351],
                [5.00, 1.00000000, 0.00848523],
                [6.00, 1.00000000, 0.00157134],
                # [7.00, 1.00000000, 0.00157134],
                # [8.00, 0.00000000, 0.00000000],
                # [9.00, 1.00000000, 0.00031427],
                # [10.00, 0.00000000, 0.00000000],
                # [11.00, 0.00000000, 0.00000000],
                            ])


QAM16 = np.array([[-10.00, 1.00000000, 0.40644753],
                [-9.00, 1.00000000, 0.38969659],
                [-8.00, 1.00000000, 0.36978508],
                [-7.00, 1.00000000, 0.37231353],
                [-6.00, 1.00000000, 0.35998736],
                [-5.00, 1.00000000, 0.32711757],
                [-4.00, 1.00000000, 0.29677623],
                [-3.00, 1.00000000, 0.28603034],
                [-2.00, 1.00000000, 0.27212389],
                [-1.00, 1.00000000, 0.21965866],
                [0.00, 1.00000000, 0.21934260],
                [1.00, 1.00000000, 0.18520860],
                [2.00, 1.00000000, 0.15771176],
                [3.00, 1.00000000, 0.15233881],
                [4.00, 1.00000000, 0.11662453],
                [5.00, 1.00000000, 0.09766119],
                [6.00, 1.00000000, 0.08027813],
                [7.00, 1.00000000, 0.06984829],
                [8.00, 1.00000000, 0.04487990],
                [9.00, 1.00000000, 0.03350190],
                [10.00, 1.00000000, 0.01675095],
                [11.00, 1.00000000, 0.01327434],
                    ])

QAM64 = np.array([[-10.00, 1.00000000, 0.42349206],
                [-9.00, 1.00000000, 0.42031746],
                [-8.00, 1.00000000, 0.42253968],
                [-7.00, 1.00000000, 0.38349206],
                [-6.00, 1.00000000, 0.38793651],
                [-5.00, 1.00000000, 0.36507937],
                [-4.00, 1.00000000, 0.37460317],
                [-3.00, 1.00000000, 0.34634921],
                [-2.00, 1.00000000, 0.32920635],
                [-1.00, 1.00000000, 0.32857143],
                [0.00, 1.00000000, 0.29968254],
                [1.00, 1.00000000, 0.26666667],
                [2.00, 1.00000000, 0.27333333],
                [3.00, 1.00000000, 0.23396825],
                [4.00, 1.00000000, 0.22253968],
                [5.00, 1.00000000, 0.20349206],
                [6.00, 1.00000000, 0.18698413],
                [7.00, 1.00000000, 0.15238095],
                [8.00, 1.00000000, 0.13365079],
                [9.00, 1.00000000, 0.11841270],
                [10.00, 1.00000000, 0.09269841],
                [11.00, 1.00000000, 0.09174603],
                    ])


def SNR_berfer( ):  ## E = 10, B = 128
    lw = 2
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##========================= BPSK ===============================
    lb = "BPSK"
    axs.semilogy(BPSK[:, 0], BPSK[:, cols], '--*',  color = 'k', label = lb, lw = lw, ms = 12 )

    #========================= QPSK ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "QPSK"
    axs.semilogy(QPSK[:, 0], QPSK[:, cols], '-',  color='r', label = lb, lw = lw )

    #========================= QAM16 ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "QAM16"
    axs.semilogy(QAM16[:, 0], QAM16[:, cols], '-', markeredgewidth = 2, color='b', label = lb, lw = lw )

    #========================= QAM64 ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "QAM64"
    axs.semilogy(QAM64[:, 0], QAM64[:, cols], '-', markeredgewidth = 2, color='g', label = lb, lw = lw)


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
    # plt.suptitle("Polar, [128,64], L = 8", fontproperties = fontt, )
    out_fig = plt.gcf()

    out_fig.savefig(os.path.join("./SNR_ber.eps") )
    out_fig.savefig(os.path.join("./SNR_ber.png") )

    plt.show()
    plt.close()
    return




SNR_berfer()












































