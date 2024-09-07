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
import Utility



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


#第0组数据,没有额外比特
LDPC5GNoExtraBer  = np.array([[0.000000, 0.1600611772, 0.9920634921],
                                [0.250000, 0.1306803995, 0.9363295880],
                                [0.500000, 0.0865661982, 0.7396449704],
                                [0.750000, 0.0389888584, 0.3663003663],
                                [1.000000, 0.0115016044, 0.1166861144],
                                [1.250000, 0.0015711885, 0.0169877349],
                                [1.500000, 0.0001291701, 0.0014351897],
                                [1.750000, 0.0000063439, 0.0000855838 ],
                                [2.000000, 0.0000003201, 0.0000092097 ]])

# LDPC5GNoExtraFer = [
# 0.000000 0.9920634921
# 0.250000 0.9363295880
# 0.500000 0.7396449704
# 0.750000 0.3663003663
# 1.000000 0.1166861144
# 1.250000 0.0169877349
# 1.500000 0.0014351897
# 1.750000 0.0000855838
# 2.000000 0.0000092097
# ];


# 第一组数据，第一列是Eb/N0或SNR, 第二列是BER，第三列是WER，下同。
LDPC5GFreeRideExtra1BitExtraBerFer = np.array([[ 0.000000,  0.0277023658,  0.0277023658],
                                                [ 0.250000,  0.0096264921,  0.0096264921],
                                                [ 0.500000,  0.0023673570,  0.0023673570],
                                                [ 0.750000,  0.0003360000,  0.0003360000],
                                                [ 1.000000,  0.0000200000,  0.0000200000],
                                                [ 1.250000,  0.0000009000,  0.0000009000 ]])
                                                # 1.500000  0.0000000000  0.0000000000
                                                # 1.750000  0.0000000000  0.0000000000



# The following results correspond to Setup_of_BPSK_AWGN0.txt
LDPC5GFreeRideExtra1BitPayloadBerFer = np.array([[ 0.000000, 0.1695081791,  0.9956230262],
                                                [ 0.250000, 0.1361787138,  0.9467558722],
                                                [ 0.500000, 0.0902647746, 0.7433074818],
                                                [ 0.750000, 0.0411093734,  0.3859320000],
                                                [ 1.000000, 0.0111053406,  0.1133065000 ],
                                                ## 1.250000 0.0016376776  0.0175820000
                                                ## 1.500000 0.0001254078  0.0014485000
                                                [ 1.250000, 0.0016417010,  0.0176497750],
                                                [ 1.500000, 0.0001266064,  0.0014556500],
                                                [ 1.750000, 0.0000051437,  0.0000755000],
                                                [ 2.000000, 0.0000002755,  0.0000090000 ]])
## 2.500000 0.0000000026  0.0000010000


## 第二组数据
## The results correspond to .\Set_up\Setup_of_BlockCodeCRC_BPSK_AWGN0.txt
LDPC5GFreeRideExtra2BitExtraBerFer = np.array([[ 0.000000,  0.0473476854,  0.0703531729],
                                               [ 0.250000,  0.0179997841,  0.0269861831],
                                               [ 0.500000, 0.0039655110,  0.0060634724],
                                               [ 0.750000,  0.0006277281,  0.0009383081],
                                                [1.000000,  0.0000550000,  0.0000770000],
                                                [1.250000,  0.0000014500, 0.0000020500 ]])
                                                ## 1.250000  0.0000014822  0.0000020000
                                                ## [1.500000,  0.0000000000,  0.0000000000 ]])


LDPC5GFreeRideExtra2BitPayloadBerFer = np.array([[  0.000000, 0.1838549728 , 0.9952159842],
                                                  [  0.250000, 0.1418919170,  0.9472420121],
                                                  [  0.500000, 0.0914873711,  0.7425752780],
                                                  [  0.750000, 0.0412932836,  0.3859083010],
                                                  [  1.000000, 0.0111733911,  0.1136940000],
                                                  [  1.250000, 0.0016431222,  0.0176500000],
                                                    ## 1.250000 0.0016257687  0.0175260000
                                                  [  1.500000, 0.0001304365,  0.0014870000],
                                                  [  1.750000, 0.0000052276,  0.0000810000 ],
                                                  [ 2.000000, 0.0000002755,  0.0000090000 ]])




def SNR_berfer( ):  ## E = 10, B = 128
    lw = 2
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 1
    ##=============================== LDPC =========================================
    lb = "Payload, 5G LDPC"
    axs.semilogy(LDPC5GNoExtraBer[:, 0], LDPC5GNoExtraBer[:, cols], label = lb, color = "#FF00FF", linestyle = '--', linewidth = 3)

    ##========================= FreeRide 1bit, Payload data ===============================
    lb = "Payload, Free-Ride, " + r"$\ell = 1$"
    axs.semilogy(LDPC5GFreeRideExtra1BitPayloadBerFer[:, 0], LDPC5GFreeRideExtra1BitPayloadBerFer[:, cols], '*',  color = 'k', label = lb, ms=18 )

    ##========================= FreeRide 2bit, Payload data ===============================
    ## markeredgecolor # 圆边缘的颜色
    ## markeredgewidth # 圆的线宽
    ## # 注意如果令markerfacecolor='none'，那线就会穿过圆
    lb = "Payload, Free-Ride, " + r"$\ell = 2$"
    axs.semilogy(LDPC5GFreeRideExtra2BitPayloadBerFer[:, 0], LDPC5GFreeRideExtra2BitPayloadBerFer[:, cols], 'o', markerfacecolor='none',  markeredgewidth = 2, color='r', label = lb,  ms=22)

    ##========================= FreeRide 1bit, extra data ===============================
    lb = "Extra data, Free-Ride, " + r"$\ell = 1$"
    axs.semilogy(LDPC5GFreeRideExtra1BitExtraBerFer[:, 0], LDPC5GFreeRideExtra1BitExtraBerFer[:, cols], linestyle = '-', linewidth = 3, marker = 'd', color='b', label = lb, markeredgewidth = 2, markerfacecolor = 'white', ms = 18 )

    ##========================= FreeRide 2bit, extra data ===============================
    lb = "Extra data, Free-Ride, " + r"$\ell = 2$"
    axs.semilogy(LDPC5GFreeRideExtra2BitExtraBerFer[:, 0], LDPC5GFreeRideExtra2BitExtraBerFer[:, cols], linestyle = '-', linewidth = 3, marker = '^', color='b', label = lb, markeredgewidth = 2, markerfacecolor = 'white', ms = 18 )

    ##===========================================================
    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    # label
    font = {'family':'Times New Roman','style':'normal','size':35}
    axs.set_xlabel("SNR (dB)", fontproperties=font)
    if cols == 2:
        axs.set_ylabel( "WER",      fontproperties = font )# , fontdict = font1
    elif cols == 1:
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
    if cols == 2:
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", "SNR_fer.pdf") )
        out_fig.savefig("./figures/SNR_fer.eps")
    elif cols == 1:
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", "SNR_ber.pdf") )
        out_fig.savefig("./figures/SNR_ber.eps")
    plt.show()
    plt.close()
    return




SNR_berfer()












































