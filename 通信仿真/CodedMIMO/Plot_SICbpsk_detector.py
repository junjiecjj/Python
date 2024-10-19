#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:45:04 2024

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
ZF = np.array([[0.00, 0.97849462, 0.02508133],
                [2.00, 0.84330244, 0.01027406],
                [4.00, 0.56330895, 0.00433875],
                [6.00, 0.25431911, 0.00126065],
                [8.00, 0.10015008, 0.00043909],
                [10.00, 0.03251633, 0.00011568],
                [12.00, 0.00914949, 0.00003271],
                [14.00, 0.00262208, 0.00000825],
                ])

MMSE = np.array([[0.00, 0.96996124, 0.01247199],
                [2.00, 0.78386844, 0.00527741],
                [4.00, 0.47061589, 0.00186130],
                [6.00, 0.19988019, 0.00060957],
                [8.00, 0.07367879, 0.00019099],
                [10.00, 0.02116190, 0.00005409],
                [12.00, 0.00592343, 0.00001381],
                [14.00, 0.00153507, 0.00000338],
                [16.00, 0.0004847431, 0.0000010279]])

SIC_sinr_mmse = np.array([[0.00, 0.82183908, 0.00426545],
                            [2.00, 0.37101557, 0.00100118],
                            [4.00, 0.07562136, 0.00013892],
                            [6.00, 0.00902525, 0.00001521],
                            [8.00, 0.00086200, 0.00000134],
                            [10.00, 0.00008100, 0.00000014] ])


SIC_snr_mmse = np.array([[0.00, 0.81914894, 0.00452000],
                        [2.00, 0.37448560, 0.00115302],
                        [4.00, 0.07624343, 0.00014586],
                        [6.00, 0.00919547, 0.00001632],
                        [8.00, 0.00089336, 0.00000144]])

SIC_snr_zf = np.array([[0.00, 1.00000000, 0.18130099],
                        [2.00, 1.00000000, 0.13519501],
                        [4.00, 1.00000000, 0.09318234],
                        [6.00, 1.00000000, 0.05432432],
                        [8.00, 0.99010880, 0.02424322],
                        [10.00, 0.92003676, 0.00823376],
                        [12.00, 0.53273018, 0.00215041],
                        [14.00, 0.14634503, 0.00041233],
                        [16.00, 0.02597841, 0.00008266],
                        [18.00, 0.00355302, 0.00001211],
                        [20.00, 0.00059600, 0.00000309], ])

SIC_norm_mmse_Hcg = np.array([[0.00, 1.00000000, 0.17770303],
                            [2.00, 1.00000000, 0.13401546],
                            [4.00, 1.00000000, 0.09308140],
                            [6.00, 1.00000000, 0.05491852],
                            [8.00, 0.99502982, 0.02513202],
                            [10.00, 0.92258065, 0.00946237],
                            [12.00, 0.55984340, 0.00276467],
                            [14.00, 0.16977612, 0.00068408],
                            [16.00, 0.04348582, 0.00018495],
                            [18.00, 0.00938602, 0.00004016],
                            [20.00, 0.00213016, 0.00001031], ])

SIC_norm_zf_Hcg = np.array([[0.00, 1.00000000, 0.18119485],
                            [2.00, 1.00000000, 0.13523560],
                            [4.00, 1.00000000, 0.09376353],
                            [6.00, 1.00000000, 0.05544924],
                            [8.00, 0.99502982, 0.02580765],
                            [10.00, 0.92003676, 0.00997434],
                            [12.00, 0.56015669, 0.00290670],
                            [14.00, 0.17713679, 0.00071567],
                            [16.00, 0.04544629, 0.00020589],
                            [18.00, 0.01026962, 0.00004620],
                            [20.00, 0.00235940, 0.00001136], ])

SIC_norm_mmse_Hf = np.array([[0.00, 1.00000000, 0.17954650],
                            [2.00, 1.00000000, 0.13614198],
                            [4.00, 1.00000000, 0.09780220],
                            [6.00, 1.00000000, 0.06318733],
                            [8.00, 0.99900200, 0.03537456],
                            [10.00, 0.99010880, 0.01778303],
                            [12.00, 0.89295272, 0.00784316],
                            [14.00, 0.59512485, 0.00262337],
                            [16.00, 0.28649113, 0.00093151],
                            [18.00, 0.11055887, 0.00032249],
                            [20.00, 0.03622349, 0.00008376] ])

SIC_norm_zf_Hf = np.array([[0.00, 1.00000000, 0.18876124],
                            [2.00, 1.00000000, 0.14127799],
                            [4.00, 1.00000000, 0.10114313],
                            [6.00, 1.00000000, 0.06582688],
                            [8.00, 1.00000000, 0.03710248],
                            [10.00, 0.98620690, 0.01881466],
                            [12.00, 0.89215686, 0.00847956],
                            [14.00, 0.60410380, 0.00285625],
                            [16.00, 0.29765091, 0.00103175],
                            [18.00, 0.11322249, 0.00034569],
                            [20.00, 0.03739260, 0.00009609], ])

ML = np.array([[0.00, 1.00000000, 0.16527483],
                [2.00, 1.00000000, 0.12116373],
                [4.00, 1.00000000, 0.07923379],
                [6.00, 1.00000000, 0.04138882],
                [8.00, 0.98233562, 0.01527641],
                [10.00, 0.81184104, 0.00389801],
                [12.00, 0.34600760, 0.00068718],
                [14.00, 0.07243650, 0.00009091],
                [16.00, 0.0089705882, 0.0000089614]
                    ])

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
    axs.semilogy(MMSE[:, 0], MMSE[:, cols], color = 'b', ls='--', lw = 3, marker = 'o', ms = 16, mew = 2, label = lb)

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

    # #========================= SNR ZF ===============================
    # # markeredgecolor # 圆边缘的颜色
    # # markeredgewidth # 圆的线宽
    # # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    # lb = "SIC, SNR, zf"
    # axs.semilogy(SIC_snr_zf[:, 0], SIC_snr_zf[:, cols], color = '#1E90FF', ls='--', lw = 3, marker = '*', ms = 12,  mew = 2, label = lb)

    # #========================= norm wmmse Hch ===============================
    # # markeredgecolor # 圆边缘的颜色
    # # markeredgewidth # 圆的线宽
    # # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    # lb = "SIC, norm, mmse, Hc"
    # axs.semilogy(SIC_norm_mmse_Hcg[:, 0], SIC_norm_mmse_Hcg[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 'o', ms = 14, mfc = 'none',  mew = 2, label = lb)

    # #========================= norm zf Hch ===============================
    # # markeredgecolor # 圆边缘的颜色
    # # markeredgewidth # 圆的线宽
    # # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    # lb = "SIC, norm, zf, Hc"
    # axs.semilogy(SIC_norm_zf_Hcg[:, 0], SIC_norm_zf_Hcg[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 's', ms = 14, mfc = 'none',  mew = 2, label = lb)

    # #========================= norm wmmse Hf ===============================
    # # markeredgecolor # 圆边缘的颜色
    # # markeredgewidth # 圆的线宽
    # # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    # lb = "SIC, norm, mmse, Hf"
    # axs.semilogy(SIC_norm_mmse_Hf[:, 0], SIC_norm_mmse_Hf[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 'd', ms = 11, mfc = 'none', mew = 2, label = lb)

    # #========================= norm zf Hf ===============================
    # # markeredgecolor # 圆边缘的颜色
    # # markeredgewidth # 圆的线宽
    # # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    # lb = "SIC, norm, zf, Hf"
    # axs.semilogy(SIC_norm_zf_Hf[:, 0], SIC_norm_zf_Hf[:, cols], color = '#FFA500', ls='--', lw = 3, marker = 'v', ms = 11, mfc = 'none',  mew = 2, label = lb)

    #========================= ML ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    # lb = "ML"
    # axs.semilogy(ML[:, 0], ML[:, cols], color = '#FF00FF', ls='-', lw = 3, marker = 'H', ms = 18,  mfc = 'none', mew = 2, label = lb)


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
    plt.suptitle("4"+r"$\times$"+"6 MIMO, BPSK, Uncoded", fontproperties = fontt, )
    out_fig = plt.gcf()

    # out_fig.savefig(os.path.join("/home/jack/FedAvg_DataResults/results/", f"SNR_berfer.eps") )
    if cols == 1:
        out_fig.savefig(os.path.join("./SIC_BPSK_fer.eps") )
        out_fig.savefig(os.path.join("./SIC_BPSK_fer.png") )
    elif cols == 2:
        out_fig.savefig(os.path.join("./SIC_BPSK_ber.eps") )
        out_fig.savefig(os.path.join("./SIC_BPSK_ber.png") )

    plt.show()
    plt.close()
    return


SNR_berfer()












































