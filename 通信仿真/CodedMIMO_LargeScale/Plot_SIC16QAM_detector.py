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
ZF = np.array([[0.00, 1.00000000, 0.49908982],
                [-10.00, 1.00000000, 0.49492615],
                [-40.00, 1.00000000, 0.33560080],
                [-50.00, 1.00000000, 0.19254890],
                [-55.00, 1.00000000, 0.12553693],
                [-60.00, 1.00000000, 0.06715768],
                [-65.00, 1.00000000, 0.02923553],
                [-70.00, 0.99404762, 0.00805556],
                [-75.00, 0.42385787, 0.00083249],
                [-80.00, 0.00982738, 0.00001146],
                ])

MMSE = np.array([[0.00, 1.00000000, 0.49704391],
                [-10.00, 1.00000000, 0.49086028],
                [-40.00, 1.00000000, 0.29812774],
                [-50.00, 1.00000000, 0.17652096],
                [-55.00, 1.00000000, 0.11753892],
                [-60.00, 1.00000000, 0.06604192],
                [-65.00, 1.00000000, 0.03026347],
                [-70.00, 0.99404762, 0.00833532],
                [-75.00, 0.43985953, 0.00088147],
                [-80.00, 0.01062814, 0.00001247],
                 ])

SIC_sinr_mmse = np.array([[0.00, 1.00000000, 0.49704192],
                        [-10.00, 1.00000000, 0.49093014],
                        [-40.00, 1.00000000, 0.29497206],
                        [-50.00, 1.00000000, 0.15610978],
                        [-55.00, 1.00000000, 0.08462874],
                        [-60.00, 1.00000000, 0.03178044],
                        [-65.00, 0.99800797, 0.00781873],
                        [-70.00, 0.42030201, 0.00061326],
                        [-75.00, 0.00151465, 0.00000154],
                        ])


SIC_snr_mmse = np.array([[0.00, 1.00000000, 0.49703194],
                        [-10.00, 1.00000000, 0.49082435],
                        [-40.00, 1.00000000, 0.29564870],
                        [-50.00, 1.00000000, 0.15600599],
                        [-55.00, 1.00000000, 0.08464870],
                        [-60.00, 1.00000000, 0.03179441],
                        [-65.00, 0.99800797, 0.00781873],
                        [-70.00, 0.42030201, 0.00061326],
                        [-75.00, 0.00151465, 0.00000154],
                        ])

SIC_snr_zf = np.array([[0.00, 1.00000000, 0.49900599],
                        [-10.00, 1.00000000, 0.49368663],
                        [-40.00, 1.00000000, 0.32861277],
                        [-50.00, 1.00000000, 0.16217565],
                        [-55.00, 1.00000000, 0.08428942],
                        [-60.00, 1.00000000, 0.03022156],
                        [-65.00, 0.99800797, 0.00716335],
                        [-70.00, 0.39294118, 0.00054902],
                        [-75.00, 0.00140219, 0.00000143],
                        ])

SIC_norm_mmse_Hcg = np.array([[0.00, 1.00000000, 0.49704192],
                            [-10.00, 1.00000000, 0.49093014],
                            [-40.00, 1.00000000, 0.29497206],
                            [-50.00, 1.00000000, 0.15613573],
                            [-55.00, 1.00000000, 0.08468064],
                            [-60.00, 1.00000000, 0.03177246],
                            [-65.00, 0.99800797, 0.00782271],
                            [-70.00, 0.42030201, 0.00061326],
                            [-75.00, 0.00151465, 0.00000154],
                            ])

SIC_norm_zf_Hcg = np.array([[0.00, 1.00000000, 0.49901796],
                            [-10.00, 1.00000000, 0.49396607],
                            [-40.00, 1.00000000, 0.32869461],
                            [-50.00, 1.00000000, 0.16246307],
                            [-55.00, 1.00000000, 0.08419561],
                            [-60.00, 1.00000000, 0.03021557],
                            [-65.00, 0.99800797, 0.00716135],
                            [-70.00, 0.39294118, 0.00054902],
                            [-75.00, 0.00140219, 0.00000143],
                            ])


SIC_norm_mmse_Hf = np.array([[0.00, 1.00000000, 0.49703593],
                            [-10.00, 1.00000000, 0.49091218],
                            [-40.00, 1.00000000, 0.29921357],
                            [-50.00, 1.00000000, 0.17771657],
                            [-55.00, 1.00000000, 0.11865469],
                            [-60.00, 1.00000000, 0.06626747],
                            [-65.00, 1.00000000, 0.03003792],
                            [-70.00, 0.99404762, 0.00830754],
                            [-75.00, 0.43985953, 0.00088762],
                            [-80.00, 0.01048227, 0.00001222],
                            ])

SIC_norm_zf_Hf = np.array([[0.00, 1.00000000, 0.49908982],
                            [-10.00, 1.00000000, 0.49492615],
                            [-40.00, 1.00000000, 0.33560080],
                            [-50.00, 1.00000000, 0.19254890],
                            [-55.00, 1.00000000, 0.12553693],
                            [-60.00, 1.00000000, 0.06715768],
                            [-65.00, 1.00000000, 0.02923553],
                            [-70.00, 0.99404762, 0.00805556],
                            [-75.00, 0.42385787, 0.00083249],
                            [-80.00, 0.00982738, 0.00001146],
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
    axs.semilogy(ZF[::-1, 0], ZF[::-1, cols], color = 'k', ls='-', lw = 3, marker = 'o', ms = 27,  mfc = 'none',  mew = 2, label = lb,)

    #========================= MMSE ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "MMSE"
    axs.semilogy(MMSE[:, 0], MMSE[:, cols], color = 'b', ls='--', lw = 3, marker = 'o', ms = 20,  mew = 2, label = lb)

    #========================= SINR ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "SIC, SINR, mmse"
    axs.semilogy(SIC_sinr_mmse[:, 0], SIC_sinr_mmse[:, cols], color = 'r', ls='-', lw = 3, marker = 'o', ms = 27,  mfc = 'none', mew = 2, label = lb)

    #========================= SNR MMSE ===============================

    lb = "SIC, SNR, mmse"
    axs.semilogy(SIC_snr_mmse[:, 0], SIC_snr_mmse[:, cols], color = '#1E90FF', ls='--', lw = 3, marker = 'D', ms = 24, mfc = 'none', mew = 2, label = lb)

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
    plt.suptitle("16"+r"$\times$"+"10 MIMO, 16QAM, Uncoded", fontproperties = fontt, )
    out_fig = plt.gcf()

    # out_fig.savefig(os.path.join("/home/jack/FedAvg_DataResults/results/", f"SNR_berfer.eps") )
    if cols == 1:
        out_fig.savefig(os.path.join("./SIC_16QAM_fer.eps") )
        out_fig.savefig(os.path.join("./SIC_16QAM_fer.png") )
    elif cols == 2:
        out_fig.savefig(os.path.join("./SIC_16QAM_ber.eps") )
        out_fig.savefig(os.path.join("./SIC_16QAM_ber.png") )

    plt.show()
    plt.close()
    return


SNR_berfer()












































