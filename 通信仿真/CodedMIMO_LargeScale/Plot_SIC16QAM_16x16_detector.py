#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:28:00 2024

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
ZF = np.array([[-40.00, 1.00000000, 0.43465413],
                [-60.00, 1.00000000, 0.24866478],
                [-70.00, 1.00000000, 0.11638832],
                [-75.00, 1.00000000, 0.06250000],
                [-80.00, 0.99404762, 0.03116668],
                [-85.00, 0.82131148, 0.01174276],
                [-90.00, 0.39417781, 0.00389011],
                [-95.00, 0.14835653, 0.00147511],
                [-100.00, 0.04941803, 0.00043694],
                [-105.00, 0.01506133, 0.00015087],
                [-115.00, 0.00165704, 0.00001390],
                ])

MMSE = np.array([[-40.00, 1.00000000, 0.32707632],
                [-60.00, 1.00000000, 0.13496640],
                [-70.00, 1.00000000, 0.05714937],
                [-75.00, 1.00000000, 0.03249945],
                [-80.00, 0.99011858, 0.01687369],
                [-85.00, 0.81596091, 0.00669438],
                [-90.00, 0.38597843, 0.00262950],
                [-95.00, 0.14483955, 0.00085375],
                [-100.00, 0.04913692, 0.00028073],
                [-105.00, 0.01472057, 0.00008123],
                [-110.00, 0.00533762, 0.00002923],
                [-115.00, 0.00158894, 0.00000907],
                 ])

SIC_sinr_mmse = np.array([[-40.00, 1.00000000, 0.32415832],
[-60.00, 1.00000000, 0.08409548],
[-70.00, 0.91926606, 0.00306766],
[-75.00, 0.06018741, 0.00006265],
[-80.00, 0.00001400, 0.00000001],
                        ])


SIC_snr_mmse = np.array([[-40.00, 1.00000000, 0.32398679],
[-60.00, 1.00000000, 0.08414616],
[-70.00, 0.91926606, 0.00308020],
[-75.00, 0.06018741, 0.00006265],
[-80.00, 0.00001400, 0.00000001],
                        ])

SIC_snr_zf = np.array([[-40.00, 1.00000000, 0.41914998],
[-60.00, 1.00000000, 0.09927801],
[-70.00, 0.90107914, 0.00290861],
[-75.00, 0.06030332, 0.00016997],
[-80.00, 0.00033000, 0.00003438],
[-85.00, 0.00009000, 0.00001117],
[-90.00, 0.00003000, 0.00000279],
                        ])

SIC_norm_mmse_Hcg = np.array([[-40.00, 1.00000000, 0.32413298],
[-60.00, 1.00000000, 0.08415591],
[-70.00, 0.92095588, 0.00316485],
[-75.00, 0.06021635, 0.00006256],
[-80.00, 0.00001400, 0.00000001],
                            ])

SIC_norm_zf_Hcg = np.array([[-40.00, 1.00000000, 0.41945211],
[-60.00, 1.00000000, 0.10381581],
[-70.00, 0.89946140, 0.00335923],
[-75.00, 0.06120953, 0.00034553],
[-80.00, 0.00069952, 0.00007169],
[-85.00, 0.00021300, 0.00002265],
[-90.00, 0.00006500, 0.00000677],
[-95.00, 0.00002300, 0.00000141],
                            ])


SIC_norm_mmse_Hf = np.array([[-40.00, 1.00000000, 0.32895537],
[-60.00, 1.00000000, 0.13932487],
[-70.00, 1.00000000, 0.05902453],
[-75.00, 1.00000000, 0.03256183],
[-80.00, 0.99207921, 0.01649134],
[-85.00, 0.80546624, 0.00648896],
[-90.00, 0.38508839, 0.00247706],
[-95.00, 0.14104730, 0.00080775],
[-100.00, 0.04897361, 0.00027588],
[-105.00, 0.01440069, 0.00007568],
[-110.00, 0.00502346, 0.00002687],
[-115.00, 0.00154726, 0.00000857],
                            ])

SIC_norm_zf_Hf = np.array([[-40.00, 1.00000000, 0.43465413],
[-60.00, 1.00000000, 0.24866478],
[-70.00, 1.00000000, 0.11638832],
[-75.00, 1.00000000, 0.06250000],
[-80.00, 0.99404762, 0.03116668],
[-85.00, 0.82131148, 0.01174276],
[-90.00, 0.39417781, 0.00389011],
[-95.00, 0.14835653, 0.00147511],
[-100.00, 0.04941803, 0.00043694],
[-105.00, 0.01506133, 0.00015087],
[-110.00, 0.00513725, 0.00004565],
[-115.00, 0.00165704, 0.00001390],
                            ])

ML = np.array([ ])


# ZF = np.flipud(ZF)
# MMSE = np.flipud(MMSE)
# SIC_sinr_mmse = np.flipud(SIC_sinr_mmse)
# SIC_snr_mmse = np.flipud(SIC_snr_mmse)
# SIC_snr_zf = np.flipud(SIC_snr_zf)
# SIC_norm_mmse_Hcg = np.flipud(SIC_norm_mmse_Hcg)
# SIC_norm_zf_Hcg = np.flipud(SIC_norm_zf_Hcg)
# SIC_norm_mmse_Hf = np.flipud(SIC_norm_mmse_Hf)
# SIC_norm_zf_Hf = np.flipud(SIC_norm_zf_Hf)

def SNR_berfer( ):  ## E = 10, B = 128
    lw = 2
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##========================= ZF ===============================
    lb = "ZF"
    axs.semilogy(ZF[::-1, 0], ZF[::-1, cols], color = 'k', ls='-', lw = 3, marker = 'o', ms = 22,  mfc = 'none',  mew = 2, label = lb,)

    #========================= MMSE ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "MMSE"
    axs.semilogy(MMSE[:, 0], MMSE[:, cols], color = 'b', ls='--', lw = 3, marker = 'o', ms = 22,  mew = 2, label = lb)

    #========================= SINR ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "SIC, SINR, mmse"
    axs.semilogy(SIC_sinr_mmse[:, 0], SIC_sinr_mmse[:, cols], color = 'r', ls='-', lw = 3, marker = 'o', ms = 22,  mfc = 'none', mew = 2, label = lb)

    #========================= SNR MMSE ===============================

    lb = "SIC, SNR, mmse"
    axs.semilogy(SIC_snr_mmse[:, 0], SIC_snr_mmse[:, cols], color = '#1E90FF', ls='--', lw = 3, marker = 'D', ms = 18, mfc = 'none', mew = 2, label = lb)

    #========================= SNR ZF ===============================

    lb = "SIC, SNR, zf"
    axs.semilogy(SIC_snr_zf[:, 0], SIC_snr_zf[:, cols], color = '#1E90FF', ls='--', lw = 3, marker = '*', ms = 20,  mew = 2, label = lb)

    #========================= norm wmmse Hch ===============================

    lb = "SIC, norm, mmse, Hc"
    axs.semilogy(SIC_norm_mmse_Hcg[:, 0], SIC_norm_mmse_Hcg[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 'o', ms = 14, mfc = 'none',  mew = 2, label = lb)

    #========================= norm zf Hch ===============================

    lb = "SIC, norm, zf, Hc"
    axs.semilogy(SIC_norm_zf_Hcg[:, 0], SIC_norm_zf_Hcg[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 's', ms = 14, mfc = 'none',  mew = 2, label = lb)

    #========================= norm wmmse Hf ===============================

    lb = "SIC, norm, mmse, Hf"
    axs.semilogy(SIC_norm_mmse_Hf[:, 0], SIC_norm_mmse_Hf[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 'd', ms = 16, mfc = 'none', mew = 2, label = lb)

    #========================= norm zf Hf ===============================

    lb = "SIC, norm, zf, Hf"
    axs.semilogy(SIC_norm_zf_Hf[:, 0], SIC_norm_zf_Hf[:, cols], color = '#FFA500', ls='--', lw = 3, marker = '1', ms = 22, mew = 4, label = lb)

    # ========================= ML ===============================

    # lb = "ML"
    # axs.semilogy(ML[:, 0], ML[:, cols], color = '#FF00FF', ls='-', lw = 3, marker = 'H', ms = 18,  mfc = 'none', mew = 2, label = lb)


    ##===========================================================
    plt.gca().invert_xaxis()

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    # label
    font = {'family':'Times New Roman','style':'normal','size':35}
    axs.set_xlabel("Noise Power (dBm)", fontproperties=font)
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

    fontt = {'family':'Times New Roman','style':'normal','size':35 }
    plt.suptitle("16"+r"$\times$"+"16 MIMO, 16QAM, Uncoded", fontproperties = fontt, )
    out_fig = plt.gcf()

    # out_fig.savefig(os.path.join("/home/jack/FedAvg_DataResults/results/", f"SNR_berfer.eps") )
    if cols == 1:
        out_fig.savefig(os.path.join("./SIC_16QAM_16x16_fer.eps") )
        out_fig.savefig(os.path.join("./SIC_16QAM_16x16_fer.png") )
    elif cols == 2:
        out_fig.savefig(os.path.join("./SIC_16QAM_16x16_ber.eps") )
        out_fig.savefig(os.path.join("./SIC_16QAM_16x16_ber.png") )

    plt.show()
    plt.close()
    return


SNR_berfer()












































