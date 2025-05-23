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

bolckfading = np.array([
                 ])

largesmall = np.array([
                        ])


def SCMAdetector_SISO( ):
    lw = 2
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##=========================   ===============================
    lb = "Fast fading"
    axs.semilogy(fastfading[::-1, 0], fastfading[::-1, cols], color = 'k', ls='-', lw = 3, marker = 'o', ms = 12, label = lb,)

    #=========================   ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    # lb = "Block fading"
    # axs.semilogy(bolckfading[:, 0], bolckfading[:, cols], color = 'b', ls='--', lw = 3, marker = 'o', ms = 18,  mew = 2, label = lb)

    #=========================   ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    # lb = "Large-small"
    # axs.semilogy(largesmall[:, 0], largesmall[:, cols], color = 'r', ls='-', lw = 3, marker = 'o', ms = 25,  mfc = 'none', mew = 2, label = lb)

    # #=========================   ===============================
    # lb = " "
    # axs.semilogy(SIC_snr_mmse[:, 0], SIC_snr_mmse[:, cols], color = '#1E90FF', ls='--', lw = 3, marker = 'D', ms = 18, mfc = 'none', mew = 2, label = lb)

    # #=========================  ===============================
    # lb = " "
    # axs.semilogy(SIC_snr_zf[:, 0], SIC_snr_zf[:, cols], color = '#1E90FF', ls='--', lw = 3, marker = '*', ms = 20,  mew = 2, label = lb)

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
    plt.suptitle("SCMA, 512, Uncoded", fontproperties = fontt, )
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












































