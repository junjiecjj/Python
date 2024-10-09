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
# matplotlib.use('Agg')
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

# mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', '$\\clubsuit$', '$\\bowtie$', '$\\boxdot$', '$\\boxtimes$', '$\\boxminus$',  '$\\boxplus$', '$\\bigodot$','$\\bigoplus$', '$\\bigotimes$', '$\\bigstar$', '$\\blacksquare$', '$\circledR$', '$\circledS$',  '$\\bigtriangledown$','$\\bigtriangleup$','$\\blacktriangleleft$', '$\\blacktriangleright$', '$\\diamondsuit',   '$\heartsuit$', '$666$', '$\\varrho$', '$\\Omega$', '$\\nabla$', '$\\Upsilon$', '$\\mho$', '$\\Phi$',  '$\\divideontimes$', '$\\otimes$', '$\\odot$', '$\\ominus$', '$\\oplus$', '$\\oslash$', '$\\otimes$']

color = ['#1E90FF','#FF6347','#00FF00','#0000FF','#4ea142','#FF00FF','#FFA500','#800080','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE','#808000']
lsty = [(0, (3, 10, 1, 10, 1, 10)), (0, (1, 1)), (0, (1, 2)), (0, (5, 1)), (0, (1, 10)), (0, (1, 2)),  (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),  '-', ':', '--', '-.', ]
alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']


# 第一组数据，第一列是Eb/N0或SNR, 第二列是BER，第三列是WER，下同。
ZF = np.array([[0.00, 1.00000000, 0.08763996],
                [1.00, 1.00000000, 0.07032135],
                [2.00, 0.99701195, 0.05451372],
                [3.00, 0.99601990, 0.04297316],
                [4.00, 0.97563353, 0.03322673],
                [5.00, 0.94971537, 0.02500840],
                [6.00, 0.87500000, 0.01646361],
                [7.00, 0.79824561, 0.01283393],
                [8.00, 0.65985498, 0.00858671],
                [9.00, 0.53500802, 0.00581852],
                [10.00, 0.43408500, 0.00427914],
                [11.00, 0.31291028, 0.00244950],
                [12.00, 0.22601039, 0.00184497],
                [13.00, 0.16589327, 0.00108543],
                [14.00, 0.12054432, 0.00076777],
                [15.00, 0.07310838, 0.00045887],
                [16.00, 0.05082766, 0.00033656],
                [17.00, 0.03279602, 0.00020533],
                [18.00, 0.02223160, 0.00012264],
                [19.00, 0.01353379, 0.00007820],
                [20.00, 0.00954961, 0.00005419],
                ])

MMSE = np.array([[0.00, 1.00000000, 0.07552552],
                [1.00, 1.00000000, 0.06059513],
                [2.00, 0.99701195, 0.04716135],
                [3.00, 0.99701195, 0.03742893],
                [4.00, 0.97373541, 0.02895337],
                [5.00, 0.94256121, 0.02141498],
                [6.00, 0.87653240, 0.01438357],
                [7.00, 0.79255740, 0.01116967],
                [8.00, 0.66072607, 0.00757460],
                [9.00, 0.53788286, 0.00494022],
                [10.00, 0.43277129, 0.00370370],
                [11.00, 0.30592910, 0.00213921],
                [12.00, 0.21718377, 0.00155391],
                [13.00, 0.16490939, 0.00103034],
                [14.00, 0.11670747, 0.00061957],
                [15.00, 0.07207661, 0.00040206],
                [16.00, 0.05001499, 0.00028933],
                [17.00, 0.03215341, 0.00017836],
                [18.00, 0.02134419, 0.00009820],
                [19.00, 0.01322779, 0.00006834],
                [20.00, 0.00929865, 0.00004830],
                ])


SIC = np.array([[0.00, 1.00000000, 0.06817870],
                [1.00, 0.99502982, 0.05029976],
                [2.00, 0.97563353, 0.03550043],
                [3.00, 0.94344958, 0.02476585],
                [4.00, 0.82522671, 0.01637727],
                [5.00, 0.64372990, 0.01071242],
                [6.00, 0.49456522, 0.00692935],
                [7.00, 0.33692359, 0.00403764],
                [8.00, 0.22590837, 0.00259512],
                [9.00, 0.13814518, 0.00144749],
                [10.00, 0.08624106, 0.00088924],
                [11.00, 0.04922547, 0.00057403],
                [12.00, 0.02972355, 0.00032850],
                [13.00, 0.01787468, 0.00019966],
                [14.00, 0.01073309, 0.00012339],
                [15.00, 0.00638267, 0.00007649],
                [16.00, 0.00385172, 0.00004889],
                [17.00, 0.00231118, 0.00002734],
                ])


SVD = np.array([[0.00, 0.99601990, 0.01018449],
                [1.00, 0.98523622, 0.00699690],
                [2.00, 0.94701987, 0.00457417],
                [3.00, 0.85336743, 0.00288834],
                [4.00, 0.72536232, 0.00178291],
                [5.00, 0.50504541, 0.00085483],
                [6.00, 0.33932203, 0.00045621],
                [7.00, 0.21024995, 0.00023072],
                [8.00, 0.09994010, 0.00009069],
                [9.00, 0.04539477, 0.00003772],
                [10.00, 0.01666140, 0.00001215],
                [11.00, 0.00638923, 0.00000456],
                # [14.00, 0.00000100, 0.00000000],
                # [15.00, 0.00000000, 0.00000000],
                # [16.00, 0.00000000, 0.00000000],
                    ])

pre_mmse = np.array([[0.00, 1.00000000, 0.15667666],
                    [1.00, 1.00000000, 0.14022800],
                    [2.00, 1.00000000, 0.13096539],
                    [3.00, 1.00000000, 0.11418633],
                    [4.00, 1.00000000, 0.10497055],
                    [5.00, 0.99502982, 0.09114273],
                    [6.00, 0.99207136, 0.07688047],
                    [7.00, 0.98330059, 0.06593300],
                    [8.00, 0.95973154, 0.05836679],
                    [9.00, 0.94078947, 0.04748003],
                    [10.00, 0.87807018, 0.04078810],
                    [11.00, 0.80790960, 0.03303950],
                    [12.00, 0.74093264, 0.02743878],
                    [13.00, 0.64331620, 0.02036338],
                    [14.00, 0.53730542, 0.01758896],
                    [15.00, 0.49603568, 0.01576540],
                    [16.00, 0.40362903, 0.01205792],
                    [17.00, 0.33886256, 0.00906945],
                    [18.00, 0.26686217, 0.00739442],
                    [19.00, 0.23608491, 0.00592730],
                    [20.00, 0.18353502, 0.00506213],
                    ])

pre_zf = np.array([[0.00, 1.00000000, 0.23663992],
                    [1.00, 1.00000000, 0.21167166],
                    [2.00, 1.00000000, 0.20132472],
                    [3.00, 1.00000000, 0.17333396],
                    [4.00, 1.00000000, 0.16428051],
                    [5.00, 0.99502982, 0.14314685],
                    [6.00, 0.99305556, 0.11721850],
                    [7.00, 0.98620690, 0.10156250],
                    [8.00, 0.96157541, 0.08992805],
                    [9.00, 0.93726592, 0.07216468],
                    [10.00, 0.88116197, 0.06257152],
                    [11.00, 0.81514658, 0.05090128],
                    [12.00, 0.74981273, 0.04296348],
                    [13.00, 0.64084507, 0.03096791],
                    [14.00, 0.54819277, 0.02777702],
                    [15.00, 0.50863821, 0.02512333],
                    [16.00, 0.41244335, 0.01930410],
                    [17.00, 0.34902371, 0.01472480],
                    [18.00, 0.26988407, 0.01107905],
                    [19.00, 0.24254907, 0.00935014],
                    [20.00, 0.18664926, 0.00781386],])


def SNR_berfer( ):  ## E = 10, B = 128
    lw = 2
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##========================= hard ===============================
    lb = "ZF, Receiver"
    axs.semilogy(ZF[:, 0], ZF[:, cols], '-*',  color = 'r', label = lb, ms = 18 , lw = lw)

    #========================= llr ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "MMSE, Receiver"
    axs.semilogy(MMSE[:, 0], MMSE[:, cols], '-o', lw = lw,  markerfacecolor='none',  markeredgewidth = 2, color='b', label = lb,  ms = 18)

    #========================= llr ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "SIC, Receiver"
    axs.semilogy(SIC[:, 0], SIC[:, cols], '-^', markeredgewidth = 2, color='g', label = lb,  ms = 18)

    #========================= llr ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "SVD, Trans & Rece"
    axs.semilogy(SVD[:, 0], SVD[:, cols], '-d', markeredgewidth = 2, color = 'k', label = lb,  ms = 18)
    #========================= ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "pre-mmse,"
    axs.semilogy(pre_mmse[:, 0], pre_mmse[:, cols], '-s', markeredgewidth = 2, color = 'cyan', label = lb,  ms = 12)

    #========================= ===============================
    # markeredgecolor # 圆边缘的颜色
    # markeredgewidth # 圆的线宽
    # # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "pre-zf,"
    axs.semilogy(pre_zf[:, 0], pre_zf[:, cols], '-h', markeredgewidth = 2, color = 'purple', label = lb,  ms = 12)

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
    # if cols == 1:
    #     out_fig.savefig(os.path.join("./SNR_fer.eps") )
    #     out_fig.savefig(os.path.join("./SNR_fer.png") )
    # elif cols == 2:
    #     out_fig.savefig(os.path.join("./SNR_ber.eps") )
    #     out_fig.savefig(os.path.join("./SNR_ber.png") )

    plt.show()
    plt.close()
    return




SNR_berfer()












































