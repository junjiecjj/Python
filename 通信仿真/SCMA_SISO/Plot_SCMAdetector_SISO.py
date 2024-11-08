#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:55:01 2024

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
fastfading = np.array([[0.00, 1.00000000, 0.27089921, 0.46235584],
                        [2.00, 1.00000000, 0.23752945, 0.41048177],
                        [4.00, 1.00000000, 0.20113312, 0.35018291],
                        [6.00, 1.00000000, 0.16187686, 0.28354415],
                        [8.00, 1.00000000, 0.12021019, 0.21171255],
                        [10.00, 1.00000000, 0.07624938, 0.13378131],
                        [12.00, 1.00000000, 0.04196506, 0.07396298],
                        [14.00, 1.00000000, 0.01932199, 0.03435020],
                        [16.00, 0.96743295, 0.00741963, 0.01341744],
                        [18.00, 0.67876344, 0.00249653, 0.00454679],
                        [20.00, 0.38127854, 0.00097805, 0.00179259],
                        [22.00, 0.15493827, 0.00034722, 0.00064863],
                        [24.00, 0.06330553, 0.00013450, 0.00025370],
                        [26.00, 0.02191601, 0.00004580, 0.00008681],
                        ])
SIMO_fastfading = np.array([[0.00, 1.00000000, 0.07347083, 0.13926866],
                        [1.00, 1.00000000, 0.05152142, 0.09881882],
                        [2.00, 1.00000000, 0.03498186, 0.06733631],
                        [3.00, 1.00000000, 0.02151925, 0.04177517],
                        [4.00, 1.00000000, 0.01192026, 0.02334449],
                        [5.00, 0.95454545, 0.00585568, 0.01156339],
                        [6.00, 0.78660436, 0.00291752, 0.00576811],
                        [7.00, 0.47175141, 0.00124875, 0.00246440],
                        ])

LDPC_fastfading = np.array([[0.00, 1.00000000, 0.26733011, 0.00000000],
                    [2.00, 1.00000000, 0.23215448, 0.00000000],
                    [4.00, 1.00000000, 0.18391927, 0.00000000],
                    [6.00, 0.99019608, 0.11997549, 0.00000000],
                    [6.50, 0.91123188, 0.09242315, 0.00000000],
                    [7.00, 0.66933333, 0.05686719, 0.00000000],
                    [7.50, 0.24181118, 0.01803254, 0.00000000],
                    [8.00, 0.05024067, 0.00342345, 0.00000000],
                    ])

bolckfading = np.array([[0.00, 1.00000000, 0.27087209, 0.46112351],
                        [2.00, 1.00000000, 0.23726594, 0.41070654],
                        [4.00, 1.00000000, 0.20355903, 0.35505022],
                        [6.00, 1.00000000, 0.16000899, 0.28041295],
                        [8.00, 0.99404762, 0.12482174, 0.21917628],
                        [10.00, 0.98235294, 0.07706036, 0.13605239],
                        [12.00, 0.92307692, 0.04046117, 0.07184352],
                        [14.00, 0.78037383, 0.01733779, 0.03078149],
                        [16.00, 0.49901381, 0.00667607, 0.01189981],
                        [18.00, 0.24159462, 0.00271674, 0.00490251],
                        [20.00, 0.09036797, 0.00093675, 0.00173541],
                        [22.00, 0.02867445, 0.00028785, 0.00053613],
                        ])

largesmall = np.array([[-50.00, 0.61975309, 0.18900463, 0.31180556],
                        [-55.00, 0.46316759, 0.11736375, 0.19927270],
                        [-60.00, 0.35836910, 0.07494104, 0.13229837],
                        [-65.00, 0.16834677, 0.02152441, 0.03908481],
                        [-70.00, 0.01223555, 0.00108881, 0.00205919],
                      ])


def SCMAdetector_SISO( ):
    lw = 2
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##=========================   ===============================
    lb = "Fast fading, SISO, w/o LDPC"
    axs.semilogy(fastfading[:, 0], fastfading[:, cols], color = 'k', ls = '-',  marker = 'o', mfc = 'none', ms = 18, label = lb,)

    # #=========================  ===============================
    lb = "Fast fading, SISO,  w/ LDPC"
    axs.semilogy(LDPC_fastfading[:, 0], LDPC_fastfading[:, cols], color = 'r', ls='--', lw = 3, marker = '*', ms = 16,  mew = 2, label = lb)

    #=========================   ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "Block fading, SISO,  w/o LDPC"
    axs.semilogy(bolckfading[:, 0], bolckfading[:, cols], color = 'b', ls='--', lw = 3,  mew = 2, label = lb)

    #=========================   ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    # lb = "Large-small"
    # axs.semilogy(largesmall[:, 0], largesmall[:, cols], color = 'r', ls='-', lw = 3, marker = 'o', ms = 25,  mfc = 'none', mew = 2, label = lb)

    # #=========================   ===============================
    lb = "Fast fading, SIMO, w/o LDPC"
    axs.semilogy(SIMO_fastfading[:, 0], SIMO_fastfading[:, cols], color = '#1E90FF', ls='--', lw = 3, marker = 'D', ms = 16, mfc = 'none', mew = 2, label = lb)

    # #=========================  ===============================
    # lb = " "
    # axs.semilogy(SIC_norm_mmse_Hcg[:, 0], SIC_norm_mmse_Hcg[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 'o', ms = 14, mfc = 'none',  mew = 2, label = lb)

    # #=========================  ===============================
    # lb = " "
    # axs.semilogy(SIC_norm_zf_Hcg[:, 0], SIC_norm_zf_Hcg[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 'd', ms = 14, mfc = 'none',  mew = 2, label = lb)

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

    fontt = {'family':'Times New Roman','style':'normal','size':35 }
    plt.suptitle("SCMA, 512", fontproperties = fontt, )
    out_fig = plt.gcf()

    if cols == 3:
        out_fig.savefig("./Figures/SCMAdetector_SISO_ser.eps")
        out_fig.savefig("./Figures/SCMAdetector_SISO_ser.png")
    elif cols == 2:
        out_fig.savefig( "./Figures/SCMAdetector_SISO_ber.eps")
        out_fig.savefig( "./Figures/SCMAdetector_SISO_ber.png")

    plt.show()
    plt.close()
    return

SCMAdetector_SISO()












































