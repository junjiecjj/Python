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
ZF = np.array([[0.00, 1.00000000, 0.18608876],
                [2.00, 1.00000000, 0.14610233],
                [4.00, 1.00000000, 0.09836577],
                [6.00, 1.00000000, 0.06651151],
                [8.00, 0.99404762, 0.03845409],
                [10.00, 0.92777778, 0.01634838],
                [12.00, 0.72294372, 0.00781532],
                [14.00, 0.42783945, 0.00326244],
                [16.00, 0.16083467, 0.00090602],
                [18.00, 0.06188241, 0.00034571],
                [20.00, 0.01771695, 0.00008005],
                [22.00, 0.00492582, 0.00002218],
                [24.00, 0.00162709, 0.00000776],
                ])

MMSE = np.array([[0.00, 1.00000000, 0.17352794],
                [2.00, 1.00000000, 0.13692537],
                [4.00, 1.00000000, 0.09438935],
                [6.00, 1.00000000, 0.06311596],
                [8.00, 0.99404762, 0.03625295],
                [10.00, 0.93296089, 0.01553771],
                [12.00, 0.71571429, 0.00738002],
                [14.00, 0.41507871, 0.00300655],
                [16.00, 0.15930048, 0.00088993],
                [18.00, 0.06232118, 0.00033309],
                [20.00, 0.01746131, 0.00007692],
                [22.00, 0.00481370, 0.00002203],
                [24.00, 0.00159768, 0.00000714],])

SIC_sinr_mmse = np.array([[0.00, 1.00000000, 0.17520428],
                            [2.00, 1.00000000, 0.13838729],
                            [4.00, 1.00000000, 0.09058835],
                            [6.00, 1.00000000, 0.05444580],
                            [8.00, 0.97281553, 0.02408601],
                            [10.00, 0.68349250, 0.00740216],
                            [12.00, 0.27243067, 0.00194569],
                            [14.00, 0.05420318, 0.00036007],
                            [16.00, 0.00875874, 0.00005982],
                            [18.00, 0.00117885, 0.00001096],
                            [20.00, 0.00017700, 0.00000193] ])


SIC_snr_mmse = np.array([[0.00, 1.00000000, 0.17583973],
                        [2.00, 1.00000000, 0.13861730],
                        [4.00, 1.00000000, 0.09045971],
                        [6.00, 1.00000000, 0.05451987],
                        [8.00, 0.97281553, 0.02424909],
                        [10.00, 0.68349250, 0.00746877],
                        [12.00, 0.27302452, 0.00195419],
                        [14.00, 0.05422078, 0.00036230],
                        [16.00, 0.00875767, 0.00005992],
                        [18.00, 0.00117935, 0.00001091],
                        [20.00, 0.00017600, 0.00000192]])

SIC_snr_zf = np.array([[0.00, 1.00000000, 0.17843220],
                        [2.00, 1.00000000, 0.13972056],
                        [4.00, 1.00000000, 0.09052988],
                        [6.00, 0.99800797, 0.05487814],
                        [8.00, 0.97470817, 0.02444066],
                        [10.00, 0.67068273, 0.00746475],
                        [12.00, 0.27694859, 0.00215394],
                        [14.00, 0.05340013, 0.00037305],
                        [16.00, 0.00979568, 0.00007076],
                        [18.00, 0.00146819, 0.00001523],
                        [20.00, 0.00024400, 0.00000286]])

SIC_norm_mmse_Hcg = np.array([[0.00, 1.00000000, 0.17584752],
                            [2.00, 1.00000000, 0.13821965],
                            [4.00, 1.00000000, 0.09124719],
                            [6.00, 1.00000000, 0.05594670],
                            [8.00, 0.97470817, 0.02589601],
                            [10.00, 0.70862801, 0.00851419],
                            [12.00, 0.30811808, 0.00260657],
                            [14.00, 0.07552005, 0.00065919],
                            [16.00, 0.01834493, 0.00016971],
                            [18.00, 0.00368131, 0.00003918],
                            [20.00, 0.00097246, 0.00001150]])

SIC_norm_zf_Hcg = np.array([[0.00, 1.00000000, 0.17902476],
                            [2.00, 1.00000000, 0.13999735],
                            [4.00, 1.00000000, 0.09080277],
                            [6.00, 0.99800797, 0.05622821],
                            [8.00, 0.97470817, 0.02637479],
                            [10.00, 0.71266003, 0.00880990],
                            [12.00, 0.31668774, 0.00284820],
                            [14.00, 0.08041734, 0.00069566],
                            [16.00, 0.01965785, 0.00018768],
                            [18.00, 0.00404371, 0.00004438],
                            [20.00, 0.00108315, 0.00001296],
                            [22.00, 0.00025700, 0.00000294], ])
                            # [24.00, 0.00006600, 0.00000063],
                            # [26.00, 0.00001800, 0.00000020]

SIC_norm_mmse_Hf = np.array([[0.00, 1.00000000, 0.17778895],
                            [2.00, 1.00000000, 0.14068348],
                            [4.00, 1.00000000, 0.09629569],
                            [6.00, 1.00000000, 0.06415294],
                            [8.00, 0.99011858, 0.03661530],
                            [10.00, 0.93470149, 0.01543916],
                            [12.00, 0.71063830, 0.00734153],
                            [14.00, 0.40964841, 0.00291771],
                            [16.00, 0.15849415, 0.00086627],
                            [18.00, 0.06007194, 0.00031030],
                            [20.00, 0.01651993, 0.00007310],
                            [22.00, 0.00467909, 0.00002092],
                            [24.00, 0.00159898, 0.00000700] ])

SIC_norm_zf_Hf = np.array([[0.00, 1.00000000, 0.18608876],
                            [2.00, 1.00000000, 0.14610233],
                            [4.00, 1.00000000, 0.09836577],
                            [6.00, 1.00000000, 0.06651151],
                            [8.00, 0.99404762, 0.03845409],
                            [10.00, 0.92777778, 0.01634838],
                            [12.00, 0.72294372, 0.00781532],
                            [14.00, 0.42783945, 0.00326244],
                            [16.00, 0.16083467, 0.00090602],
                            [18.00, 0.06188241, 0.00034571],
                            [20.00, 0.01771695, 0.00008005],
                            [22.00, 0.00492582, 0.00002218],
                            [24.00, 0.00162709, 0.00000776],
                            ])

ML = np.array([[0.00, 1.00000000, 0.16384809],
                [2.00, 1.00000000, 0.12536256],
                [4.00, 1.00000000, 0.07749345],
                [6.00, 0.99800797, 0.04231122],
                [8.00, 0.92095588, 0.01605943],
                [10.00, 0.50606061, 0.00371686],
                [12.00, 0.14862059, 0.00067441],
                [14.00, 0.02466644, 0.00008549],
                [16.00, 0.00312549, 0.00000943],
                [18.00, 0.0002522704, 0.0000008869]])

def SNR_berfer( ):  ## E = 10, B = 128
    lw = 2
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##========================= ZF ===============================
    lb = "ZF"
    axs.semilogy(ZF[:, 0], ZF[:, cols], color = 'k', ls='-', lw = 3, marker = 'o', ms = 20,  mfc = 'none',  mew = 2, label = lb,)

    #========================= MMSE ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "MMSE"
    axs.semilogy(MMSE[:, 0], MMSE[:, cols], color = 'b', ls='--', lw = 3, marker = 'o', ms = 13, mew = 2, label = lb)

    #========================= SINR ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "SIC, SINR, mmse"
    axs.semilogy(SIC_sinr_mmse[:, 0], SIC_sinr_mmse[:, cols], color = 'r', ls='-', lw = 3, marker = 'o', ms = 18,  mfc = 'none',  mew = 2, label = lb)

    #========================= SNR MMSE ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "SIC, SNR, mmse"
    axs.semilogy(SIC_snr_mmse[:, 0], SIC_snr_mmse[:, cols], color = '#1E90FF', ls='--', lw = 3, marker = 'o', ms = 12,  mew = 2, label = lb)

    #========================= SNR ZF ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "SIC, SNR, zf"
    axs.semilogy(SIC_snr_zf[:, 0], SIC_snr_zf[:, cols], color = '#1E90FF', ls='--', lw = 3, marker = '*', ms = 12,  mew = 2, label = lb)

    #========================= norm wmmse Hch ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "SIC, norm, mmse, Hc"
    axs.semilogy(SIC_norm_mmse_Hcg[:, 0], SIC_norm_mmse_Hcg[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 'o', ms = 14, mfc = 'none',  mew = 2, label = lb)

    #========================= norm zf Hch ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "SIC, norm, zf, Hc"
    axs.semilogy(SIC_norm_zf_Hcg[:, 0], SIC_norm_zf_Hcg[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 's', ms = 14, mfc = 'none',  mew = 2, label = lb)

    #========================= norm wmmse Hf ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "SIC, norm, mmse, Hf"
    axs.semilogy(SIC_norm_mmse_Hf[:, 0], SIC_norm_mmse_Hf[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 'd', ms = 11, mfc = 'none', mew = 2, label = lb)

    #========================= norm zf Hf ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "SIC, norm, zf, Hf"
    axs.semilogy(SIC_norm_zf_Hf[:, 0], SIC_norm_zf_Hf[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 'v', ms = 11, mfc = 'none',  mew = 2, label = lb)

    #========================= ML ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "ML"
    axs.semilogy(ML[:, 0], ML[:, cols], color = '#FF00FF', ls='-', lw = 3, marker = 'H', ms = 18,  mfc = 'none', mew = 2, label = lb)


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

    fontt = {'family':'Times New Roman','style':'normal','size':35 }
    plt.suptitle("4"+r"$\times$"+"6 MIMO, 16QAM, Uncoded", fontproperties = fontt, )
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












































