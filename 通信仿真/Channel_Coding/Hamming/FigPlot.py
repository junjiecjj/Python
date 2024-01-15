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
hard = np.array([[0.00 , 0.30887793,  0.13856350,  1.000],
                [0.50,  0.28257191,  0.12492950,  1.000],
                [1.00,  0.22638952,  0.10099413,  1.000],
                [1.50,  0.20831601,  0.09625780,  1.000],
                [2.00,  0.14937388,  0.06745677,  1.000],
                [2.50,  0.11841172,  0.05199716,  1.000],
                [3.00,  0.10224490,  0.04551020,  1.000],
                [3.50,  0.07528174,  0.03403456,  1.000],
                [4.00,  0.05717873,  0.02490870,  1.000],
                [4.50,  0.04060626,  0.01795267,  1.000],
                [5.00,  0.02581410,  0.01103926,  1.000],
                [5.50,  0.01620310,  0.00695343,  1.000],
                [6.00,  0.01039721,  0.00465903,  1.000],
                [6.50,  0.00593890,  0.00250418,  1.000],
                [7.00,  0.00317000,  0.00137304,  1.000],
                [7.50,  0.00162884,  0.00069250,  1.000],
                [8.00,  0.00078980,  0.00034800,  1.000],
                [8.50,  0.00032200,  0.00013875,  1.000],
                [9.00,  0.00011300,  0.00004725,  1.000],
                [9.50,  0.00004700,  0.00002075,  1.000],
                [10.00,  0.00001100,  0.00000500,  1.000],
                [10.50,  0.00000500,  0.00000175,  1.000],
                # [11.00,  0.00000000,  0.00000000,  1.000],
                # [11.50,  0.00000100,  0.00000050,  1.000],
                # [12.00,  0.00000000,  0.00000000,  1.000]
                ])



soft = np.array([[0.00,  0.22336157,  0.10276416,  1.000],
                [0.50,  0.18680089,  0.08165548,  1.000],
                [1.00,  0.16224093,  0.07431995,  1.000],
                [1.50,  0.11059603,  0.05082781,  1.000],
                [2.00,  0.08835979,  0.04091711,  1.000],
                [2.50,  0.07063302,  0.03225011,  1.000],
                [3.00,  0.04803452,  0.02114094,  1.000],
                [3.50,  0.03415599,  0.01557813,  1.000],
                [4.00,  0.02160041,  0.00938820,  1.000],
                [4.50,  0.01308333,  0.00589533,  1.000],
                [5.00,  0.00731248,  0.00325486,  1.000],
                [5.50,  0.00379732,  0.00169212,  1.000],
                [6.00,  0.00197695,  0.00085826,  1.000],
                [6.50,  0.00093642,  0.00041307,  1.000],
                [7.00,  0.00039700,  0.00016950,  1.000],
                [7.50,  0.00012500,  0.00005650,  1.000],
                [8.00,  0.00005200,  0.00002450,  1.000],
                [8.50,  0.00001000,  0.00000450,  1.000],
                [9.00,  0.00000400,  0.00000175,  1.000],
                # [9.50,  0.00000000,  0.00000000,  1.000],
                # [10.00,  0.00000000,  0.00000000,  1.000],
                # [10.50,  0.00000000,  0.00000000,  1.000],
                # [11.00,  0.00000000,  0.00000000,  1.000],
                # [11.50,  0.00000000,  0.00000000,  1.000],
                # [12.00,  0.00000000,  0.00000000,  1.000]
                ])


def SNR_berfer( ):  ## E = 10, B = 128
    lw = 2
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##========================= hard ===============================
    lb = "Hamming, hard"
    axs.semilogy(hard[:, 0], hard[:, cols], '--*',  color = 'k', label = lb, ms = 22 )

    ##========================= llr ===============================
    ## markeredgecolor # 圆边缘的颜色
    ## markeredgewidth # 圆的线宽
    ## # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "Hamming, soft"
    axs.semilogy(soft[:, 0], soft[:, cols], '-o', markerfacecolor='none',  markeredgewidth = 2, color='r', label = lb,  ms=22)

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

    # fontt = {'family':'Times New Roman','style':'normal','size':16}
    # if self.title != '':
    #     plt.suptitle(self.title, fontproperties = fontt, )
    out_fig = plt.gcf()

    # out_fig.savefig(os.path.join("/home/jack/FedAvg_DataResults/results/", f"SNR_berfer.eps") )
    if cols == 1:
        out_fig.savefig(os.path.join("./SNR_fer.png") )
    elif cols == 2:
        out_fig.savefig(os.path.join("./SNR_ber.png") )
    plt.show()
    plt.close()
    return




SNR_berfer()












































