#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 21:43:36 2025

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


sic_fastfading_2u_equallo = np.array([[-120.00, 1.00000000, 0.20553385, 0.31882813, 0.00000000, 50.000],
                                    [-122.00, 1.00000000, 0.16787760, 0.26065104, 0.00000000, 50.000],
                                    [-124.00, 1.00000000, 0.12842448, 0.20619792, 0.00000000, 50.000],
                                    [-126.00, 0.71428571, 0.07044271, 0.11543899, 0.00000000, 39.624],
                                    [-128.00, 0.14312977, 0.01274377, 0.02077618, 0.00000000, 15.136],
                                    [-130.00, 0.02302026, 0.00176369, 0.00289971, 0.00000000, 7.698],
                                    [-132.00, 0.00420824, 0.00030291, 0.00050186, 0.00000000, 5.595],
                                    [-134.00, 0.00073454, 0.00005152, 0.00008681, 0.00000000, 4.775],
                                    [-136.00, 0.00022588, 0.00001701, 0.00002772, 0.00000000, 4.426],
                                    # [-138.00, 0.0000755131, 0.0000055442, 0.00002772, 0.00000000, 4.426],
                                    ])

sic_fastfading_2u_powerallo = np.array([[-120.00, 0.50675676, 0.14758631, 0.29481630, 0.00000000, 30.632],
                                        [-122.00, 0.50000000, 0.12272786, 0.24545573, 0.00000000, 27.577],
                                        [-124.00, 0.50000000, 0.09438802, 0.18877604, 0.00000000, 26.733],
                                        [-126.00, 0.46583851, 0.04883419, 0.09766838, 0.00000000, 25.587],
                                        [-127.00, 0.14910537, 0.01066647, 0.02133294, 0.00000000, 14.989],
                                        [-128.00, 0.00173909, 0.00010507, 0.00021014, 0.00000000, 5.761],
                                        [-128.20, 0.00050743, 0.00002822, 0.00005644, 0.00000000, 5.186],
                                        # [-128.40, 0.0000767940, 0.0000046533, 0.00005644, 0.00000000, 5.186],
                                        ])

Joint_fastfading_2u_w_powdiv = np.array([[-120.00, 1.00000000, 0.26014323, 0.00000000, 0.00000000, 50.000, 0.1494667057],
                                        [-121.00, 1.00000000, 0.23660156, 0.00000000, 0.00000000, 50.000, 0.1468049849],
                                        [-122.00, 1.00000000, 0.21294271, 0.00000000, 0.00000000, 50.000, 0.1482435448],
                                        [-123.00, 1.00000000, 0.18476562, 0.00000000, 0.00000000, 50.000, 0.1509814138],
                                        [-124.00, 1.00000000, 0.15389323, 0.00000000, 0.00000000, 50.000, 0.1514214115],
                                        [-125.00, 1.00000000, 0.11061198, 0.00000000, 0.00000000, 50.000, 0.1492989433],
                                        [-126.00, 0.42857143, 0.02579241, 0.00000000, 0.00000000, 37.423, 0.1082661786],
                                        [-126.20, 0.12336601, 0.00662531, 0.00000000, 0.00000000, 24.314, 0.0709446920],
                                        [-126.40, 0.01771053, 0.00088470, 0.00000000, 0.00000000, 16.341, 0.0476566265],
                                        [-126.60, 0.00126178, 0.00005658, 0.00000000, 0.00000000, 12.197, 0.0357003152],
                                        ])

Joint_fastfading_2u_wo_powdiv = np.array([[-120.00, 1.00000000, 0.26062500, 0.00000000, 0.00000000, 50.000, 0.1517253878],
                                        [-121.00, 1.00000000, 0.23647135, 0.00000000, 0.00000000, 50.000, 0.1500303794],
                                        [-122.00, 1.00000000, 0.21098958, 0.00000000, 0.00000000, 50.000, 0.1506492623],
                                        [-123.00, 1.00000000, 0.18289063, 0.00000000, 0.00000000, 50.000, 0.1524523123],
                                        [-124.00, 1.00000000, 0.15188802, 0.00000000, 0.00000000, 50.000, 0.1532646655],
                                        [-125.00, 1.00000000, 0.10933594, 0.00000000, 0.00000000, 50.000, 0.1522373272],
                                        [-126.00, 0.30487805, 0.01780837, 0.00000000, 0.00000000, 35.866, 0.1064184196],
                                        [-126.20, 0.07804370, 0.00402311, 0.00000000, 0.00000000, 24.829, 0.0732979961],
                                        [-126.60, 0.00110924, 0.00004892, 0.00000000, 0.00000000, 13.091, 0.0381327593],
                                        ])

def SISO_2user():
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##=========================   ===============================
    lb = "联合译码, 有功率分配"
    axs.semilogy(Joint_fastfading_2u_w_powdiv[:, 0], Joint_fastfading_2u_w_powdiv[:, cols], color = 'r', ls = '-',  marker = '*', mfc = 'none', ms = 16, mew = 2, label = lb, zorder = 12)

    lb = "联合译码, 无功率分配"
    axs.semilogy(Joint_fastfading_2u_wo_powdiv[:, 0], Joint_fastfading_2u_wo_powdiv[:, cols], color = 'r', ls = '--',  marker = 'd', mfc = 'none', ms = 12, mew = 2, label = lb, zorder = 12)

    # #=========================  ===============================
    lb = r"SIC, 有功率分配"
    axs.semilogy(sic_fastfading_2u_powerallo[:, 0], sic_fastfading_2u_powerallo[:, cols], color = 'b',ls = '-', lw = 2,  marker = 'v', mfc = 'none', ms = 12, mew = 2, label = lb)

    lb = r"SIC, 无功率分配"
    axs.semilogy(sic_fastfading_2u_equallo[:, 0], sic_fastfading_2u_equallo[:, cols], color = 'b', ls = '-', lw = 2,  marker = 'o', mfc = 'none', ms = 12, mew = 2, label = lb,)

    ##===========================================================
    plt.gca().invert_xaxis()

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    # label
    # font = {'family':'Times New Roman','style':'normal','size':35}
    font = FontProperties(fname=fontpath+"simsun.ttf", size=32)
    axs.set_xlabel("噪声功率谱密度(dBm/Hz)", fontproperties=font, labelpad = 0.2 )
    if cols == 1:
        axs.set_ylabel( "FER",      fontproperties = font, labelpad = 0.2  )# , fontdict = font1
    elif cols == 2:
        axs.set_ylabel( "误码率",      fontproperties = font, labelpad = 0.2  )# , fontdict = font1
    elif cols == 3:
        axs.set_ylabel( "Aggregation error rate",      fontproperties = font )# , fontdict = font1

    font1 = FontProperties(fname=fontpath+"simsun.ttf", size=30)
    # font1 = {'family':'Times New Roman','style':'normal','size':26, }
    legend1 = axs.legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = font1, labelspacing = 0.2, borderpad= 0, handletextpad = 0.1)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    bw = 2
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 30, width = bw)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # [label.set_fontsize(26) for label in labels] #刻度值字号

    # fontt = {'family':'Times New Roman','style':'normal','size':35 }
    # plt.suptitle("2 User, Fastfading, BPSK, [1024, 512]", fontproperties = fontt, )
    out_fig = plt.gcf()

    if cols == 1:
        out_fig.savefig("./Figures/Fig5b_fer.pdf")
    elif cols == 2:
        out_fig.savefig( "./Figures/Fig5b_ber.pdf")
    # elif cols == 3:
        # out_fig.savefig( "./Figures/Fig5b_aggerr.eps")
    plt.show()
    plt.close()
    return

SISO_2user()


# 第一组数据，第一列是Eb/N0或SNR, 第二列是BER，第三列是WER，下同。
sic_fastfading_4u_equallo = np.array([[-120.00, 1.00000000, 0.30444336, 0.59968750, 0.00000000, 50.000],
                                        [-125.00, 1.00000000, 0.24949219, 0.51539062, 0.00000000, 50.000],
                                        [-130.00, 0.89732143, 0.17398507, 0.39208984, 0.00000000, 47.326],
                                        [-135.00, 0.56741573, 0.14506891, 0.34493504, 0.00000000, 26.483],
                                        [-140.00, 0.35243056, 0.10589939, 0.25364855, 0.00000000, 16.420],
                                        [-145.00, 0.27322404, 0.08763181, 0.20746884, 0.00000000, 13.743],
                                        # [-150.00, 0.29585799, 0.09345530, 0.22331500, 0.00000000, 14.124],
                                        # [-155.00, 0.24052133, 0.07863411, 0.18655584, 0.00000000, 12.956],
                                        # [-160.00, 0.27472527, 0.08820452, 0.21041166, 0.00000000, 13.330],
                                        # [-165.00, 0.29585799, 0.09420361, 0.22576507, 0.00000000, 13.712],
                                        # [-170.00, 0.30303030, 0.09613222, 0.23024384, 0.00000000, 14.065],
                                        ])

sic_fastfading_4u_powerallo = np.array([[-120.00, 1.00000000, 0.29964844, 0.60867187, 0.00000000, 50.000],
                                        [-125.00, 0.78906250, 0.18058777, 0.46371460, 0.00000000, 43.316],
                                        [-126.00, 0.54347826, 0.11640200, 0.36459749, 0.00000000, 34.106],
                                        [-128.00, 0.30487805, 0.06925555, 0.25362043, 0.00000000, 22.027],
                                        [-130.00, 0.25380711, 0.04426505, 0.17540451, 0.00000000, 17.774],
                                        [-132.00, 0.19607843, 0.01788450, 0.07013634, 0.00000000, 15.436],
                                        [-134.00, 0.00070510, 0.00017240, 0.00043813, 0.00000000, 5.739],
                                        [-135.00, 0.00049486, 0.00013551, 0.00033863, 0.00000000, 4.981],
                                        [-136.00, 0.00034860, 0.00010361, 0.00025660, 0.00000000, 4.558],
                                        ])

Joint_fastfading_4u_w_powdiv = np.array([[-120.00, 1.00000000, 0.28054328, 0.00000000, 0.00000000, 50.000, 0.1822076725],
                                        [-121.00, 1.00000000, 0.25929019, 0.00000000, 0.00000000, 50.000, 0.1918319459],
                                        [-122.00, 1.00000000, 0.23851254, 0.00000000, 0.00000000, 50.000, 0.1888704951],
                                        [-123.00, 1.00000000, 0.21305767, 0.00000000, 0.00000000, 50.000, 0.1876441141],
                                        [-124.00, 1.00000000, 0.18770559, 0.00000000, 0.00000000, 50.000, 0.1865629461],
                                        [-125.00, 1.00000000, 0.14067640, 0.00000000, 0.00000000, 50.000, 0.1879788384],
                                        [-126.00, 0.29230769, 0.02460186, 0.00000000, 0.00000000, 31.600, 0.1182412623],
                                        [-126.20, 0.02898551, 0.00212967, 0.00000000, 0.00000000, 17.955, 0.0680545673],
                                        [-126.40, 0.0013487313, 0.0000922279, 0.00000000, 0.00000000, 12.4, 0.0474539846]
                                        ])

Joint_fastfading_4u_wo_powdiv = np.array([[-120.00, 1.00000000, 0.25502416, 0.00000000, 0.00000000, 50.000, 0.1736763883],
                                        [-121.00, 1.00000000, 0.23328279, 0.00000000, 0.00000000, 50.000, 0.1738122156],
                                        [-122.00, 1.00000000, 0.20442280, 0.00000000, 0.00000000, 50.000, 0.1778355281],
                                        [-123.00, 1.00000000, 0.16971628, 0.00000000, 0.00000000, 50.000, 0.1779966912],
                                        [-124.00, 0.87209302, 0.10937500, 0.00000000, 0.00000000, 50.000, 0.1766429688],
                                        [-125.00, 0.25000000, 0.02879557, 0.00000000, 0.00000000, 50.000, 0.1811917017],
                                        [-126.00, 0.24350649, 0.01754325, 0.00000000, 0.00000000, 49.591, 0.1833519345],
                                        [-127.00, 0.00717429, 0.00033377, 0.00000000, 0.00000000, 15.843, 0.0586103715],
                                        [-127.20, 0.00131234, 0.00005803, 0.00000000, 0.00000000, 12.313, 0.0458159503],
                                        ])

def SISO_4user():
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##=========================   ===============================
    lb = "联合译码, 有功率分配"
    axs.semilogy(Joint_fastfading_4u_w_powdiv[:, 0], Joint_fastfading_4u_w_powdiv[:, cols], color = 'r', ls = '-',  marker = '*', mfc = 'none', ms = 16, mew = 2, label = lb, zorder = 12)

    ##=========================   ===============================
    lb = "联合译码, 无功率分配"
    axs.semilogy(Joint_fastfading_4u_wo_powdiv[:, 0], Joint_fastfading_4u_wo_powdiv[:, cols], color = 'r', ls = '--',  marker = 'd', mfc = 'none', ms = 12, mew = 2, label = lb, zorder = 12)

    # #=========================  ===============================
    lb = r"SIC, 有功率分配"
    axs.semilogy(sic_fastfading_4u_powerallo[:, 0], sic_fastfading_4u_powerallo[:, cols], color = 'b',ls = '-', lw = 2,  marker = 'v', mfc = 'none', ms = 12, mew = 2, label = lb)

    lb = r"SIC, 无功率分配"
    axs.semilogy(sic_fastfading_4u_equallo[:, 0], sic_fastfading_4u_equallo[:, cols], color = 'b', ls = '-', lw = 2,  marker = 'o', mfc = 'none', ms = 12, mew = 2, label = lb,)

    ##===========================================================
    plt.gca().invert_xaxis()

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    # label
    # font = {'family':'Times New Roman','style':'normal','size':35}
    font = FontProperties(fname=fontpath+"simsun.ttf", size=32)
    axs.set_xlabel("噪声功率谱密度(dBm/Hz)", fontproperties=font, labelpad = 0.2 )
    if cols == 1:
        axs.set_ylabel( "FER",      fontproperties = font, labelpad = 0.2  )# , fontdict = font1
    elif cols == 2:
        axs.set_ylabel( "误码率",      fontproperties = font, labelpad = 0.2  )# , fontdict = font1
    elif cols == 3:
        axs.set_ylabel( "Aggregation error rate",      fontproperties = font )# , fontdict = font1

    # font1 = {'family':'Times New Roman','style':'normal','size':26, }
    font1 = FontProperties(fname=fontpath+"simsun.ttf", size=30)
    legend1 = axs.legend(loc = 'lower right', borderaxespad = 0, edgecolor = 'black', prop = font1, labelspacing = 0.2, borderpad= 0, handletextpad = 0.1)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    bw = 2
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 30, width = bw)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # [label.set_fontsize(26) for label in labels] #刻度值字号

    # fontt = {'family':'Times New Roman','style':'normal','size':35 }
    # plt.suptitle("2 User, Fastfading, BPSK, [1024, 512]", fontproperties = fontt, )
    out_fig = plt.gcf()

    if cols == 1:
        out_fig.savefig("./Figures/Fig5c_fer.pdf")
    elif cols == 2:
        out_fig.savefig( "./Figures/Fig5c_ber.pdf")
    elif cols == 3:
        out_fig.savefig( "./Figures/Fig5c_aggerr.pdf")
    plt.show()
    plt.close()
    return

SISO_4user()



sic_fastfading_6u_equallo = np.array([[-120.00, 0.84444444, 0.26503906, 0.64114583, 0.00000000, 44.872],
                                        [-122.00, 0.83333333, 0.24866536, 0.60924479, 0.00000000, 43.400],
                                        [-124.00, 0.83333333, 0.24140625, 0.58502604, 0.00000000, 43.367],
                                        [-126.00, 0.83333333, 0.23247613, 0.56712240, 0.00000000, 43.044],
                                        [-128.00, 0.83333333, 0.22866753, 0.55944010, 0.00000000, 42.961],
                                        [-130.00, 0.83333333, 0.22535807, 0.54231771, 0.00000000, 43.033],
                                        [-132.00, 0.83333333, 0.22315538, 0.53554687, 0.00000000, 42.794],
                                        [-134.00, 0.83333333, 0.21962891, 0.51920573, 0.00000000, 43.011],
                                        [-136.00, 0.83333333, 0.21512587, 0.51458333, 0.00000000, 42.922],
                                        [-138.00, 0.83333333, 0.21168620, 0.49440104, 0.00000000, 42.878],
                                        [-140.00, 0.83333333, 0.20927734, 0.47343750, 0.00000000, 42.894],
                                        [-142.00, 0.83333333, 0.20939670, 0.45507812, 0.00000000, 42.872],
                                        [-144.00, 0.83333333, 0.20638021, 0.44648437, 0.00000000, 42.933],
                                        [-146.00, 0.83333333, 0.20285373, 0.43925781, 0.00000000, 42.950],
                                        [-148.00, 0.83333333, 0.20335286, 0.44173177, 0.00000000, 42.944],
                                        # [-150.00, 0.83333333, 0.20507812, 0.43815104, 0.00000000, 42.944],
                                        # [-152.00, 0.83333333, 0.20678168, 0.44218750, 0.00000000, 42.917],
                                        # [-154.00, 0.83333333, 0.20649957, 0.43385417, 0.00000000, 42.994],
                                        # [-156.00, 0.83333333, 0.20546875, 0.43652344, 0.00000000, 42.950],
                                        # [-158.00, 0.83333333, 0.20342882, 0.43085937, 0.00000000, 42.872],
                                        # [-160.00, 0.83333333, 0.20638021, 0.43085937, 0.00000000, 42.994],
                                        # [-162.00, 0.83333333, 0.20945095, 0.43678385, 0.00000000, 42.894],
                                        # [-164.00, 0.83333333, 0.20634766, 0.43085937, 0.00000000, 42.961],
                                        # [-166.00, 0.83333333, 0.20917969, 0.43717448, 0.00000000, 42.856],
                                        # [-168.00, 0.83333333, 0.21163194, 0.43951823, 0.00000000, 42.878],
                                        ])

sic_fastfading_6u_powerallo = np.array([[-120.00, 0.61788618, 0.19346259, 0.61204268, 0.00000000, 35.305],
                                        [-122.00, 0.50000000, 0.16589844, 0.57156250, 0.00000000, 28.497],
                                        [-124.00, 0.50000000, 0.15136068, 0.54054687, 0.00000000, 27.563],
                                        [-126.00, 0.50000000, 0.13776693, 0.50257812, 0.00000000, 27.240],
                                        [-128.00, 0.50000000, 0.12415365, 0.45617188, 0.00000000, 27.000],
                                        [-130.00, 0.50000000, 0.10866536, 0.40109375, 0.00000000, 26.907],
                                        [-132.00, 0.50000000, 0.09666667, 0.36281250, 0.00000000, 26.837],
                                        [-134.00, 0.50000000, 0.08523438, 0.32375000, 0.00000000, 26.810],
                                        [-136.00, 0.50000000, 0.07906901, 0.29937500, 0.00000000, 26.773],
                                        [-138.00, 0.40322581, 0.06142893, 0.23566658, 0.00000000, 23.030],
                                        [-140.00, 0.41666667, 0.05908203, 0.22910156, 0.00000000, 23.400],
                                        [-142.00, 0.34722222, 0.04786965, 0.18701172, 0.00000000, 20.530],
                                        [-144.00, 0.36231884, 0.04921969, 0.18636775, 0.00000000, 21.280],
                                        [-146.00, 0.29761905, 0.03924464, 0.14274089, 0.00000000, 18.621],
                                        [-148.00, 0.30864198, 0.03807790, 0.13269194, 0.00000000, 18.739],
                                        [-150.00, 0.31645570, 0.03704757, 0.12223101, 0.00000000, 19.253],
                                        [-152.00, 0.28735632, 0.03235004, 0.10616469, 0.00000000, 18.142],
                                        [-154.00, 0.34246575, 0.03634239, 0.11306721, 0.00000000, 20.235],
                                        [-156.00, 0.32467532, 0.03394295, 0.10181615, 0.00000000, 19.444],
                                        [-158.00, 0.27777778, 0.02951027, 0.08823785, 0.00000000, 17.444],
                                        # [-160.00, 0.26881720, 0.02764477, 0.08186324, 0.00000000, 17.095],
                                        # [-162.00, 0.29761905, 0.03083147, 0.08775112, 0.00000000, 18.359],
                                        # [-164.00, 0.26595745, 0.02813677, 0.08011968, 0.00000000, 17.275],
                                        # [-166.00, 0.30864198, 0.03201357, 0.09223090, 0.00000000, 18.862],
                                        # [-168.00, 0.26315789, 0.02790913, 0.07905016, 0.00000000, 17.075],
                                        # [-170.00, 0.30120482, 0.03247757, 0.09160862, 0.00000000, 18.701],
                                        # [-172.00, 0.22935780, 0.02410945, 0.06780390, 0.00000000, 15.560],
                                        # [-174.00, 0.29411765, 0.03271293, 0.09124540, 0.00000000, 18.124],
                                        # [-176.00, 0.26595745, 0.02831685, 0.07991190, 0.00000000, 17.066],
                                        # [-178.00, 0.28735632, 0.02997785, 0.08306394, 0.00000000, 17.906],
                                        # [-180.00, 0.29411765, 0.03281250, 0.09216452, 0.00000000, 18.131],
                                        ])

Joint_fastfading_6u_w_powdiv = np.array([[-120.00, 1.00000000, 0.32671875, 0.00000000, 0.00000000, 50.000, 0.3711303649],
                                        [-121.00, 1.00000000, 0.31713542, 0.00000000, 0.00000000, 50.000, 0.3697582367],
                                        [-122.00, 1.00000000, 0.30430990, 0.00000000, 0.00000000, 50.000, 0.3656459606],
                                        [-123.00, 1.00000000, 0.28903646, 0.00000000, 0.00000000, 50.000, 0.3605489332],
                                        [-124.00, 1.00000000, 0.27545573, 0.00000000, 0.00000000, 50.000, 0.3635264950],
                                        [-125.00, 1.00000000, 0.25367187, 0.00000000, 0.00000000, 50.000, 0.3672769744],
                                        [-126.00, 1.00000000, 0.23125000, 0.00000000, 0.00000000, 50.000, 0.3757396185],
                                        [-127.00, 1.00000000, 0.19867187, 0.00000000, 0.00000000, 50.000, 0.3709950611],
                                        [-128.00, 1.00000000, 0.14460938, 0.00000000, 0.00000000, 50.000, 0.3731828407],
                                        [-128.20, 0.69444444, 0.08921984, 0.00000000, 0.00000000, 44.389, 0.3300782215],
                                        [-128.40, 0.16129032, 0.01773143, 0.00000000, 0.00000000, 23.968, 0.1752358337],
                                        [-128.60, 0.01778094, 0.00206333, 0.00000000, 0.00000000, 14.361, 0.1048898537],
                                        [-128.8, 0.0005140633, 0.0000564169,  0.00000000, 0.00000000, 10.3, 0.0753655042 ]
                                        ])
Joint_fastfading_6u_wo_powdiv = np.array([[-120.00, 0.66666667, 0.19237425, 0.00000000, 0.00000000, 50.000, 0.3559153724],
                                        [-121.00, 0.66666667, 0.18085252, 0.00000000, 0.00000000, 50.000, 0.3597353170],
                                        [-122.00, 0.66666667, 0.16564727, 0.00000000, 0.00000000, 50.000, 0.3572043981],
                                        [-123.00, 0.66666667, 0.15245511, 0.00000000, 0.00000000, 50.000, 0.3547801394],
                                        [-124.00, 0.66666667, 0.13601631, 0.00000000, 0.00000000, 50.000, 0.3577240081],
                                        [-125.00, 0.66666667, 0.11179071, 0.00000000, 0.00000000, 50.000, 0.3627203525],
                                        [-126.00, 0.51700680, 0.05971646, 0.00000000, 0.00000000, 50.000, 0.3683639131],
                                        [-127.00, 0.10162602, 0.00533140, 0.00000000, 0.00000000, 41.898, 0.3117485328],
                                        [-127.20, 0.03250975, 0.00151501, 0.00000000, 0.00000000, 32.592, 0.2372262227],
                                        [-127.40, 0.00727802, 0.00030969, 0.00000000, 0.00000000, 23.823, 0.1759548730],
                                        [-127.60, 0.00107633, 0.00004359, 0.00000000, 0.00000000, 18.208, 0.1316847807],
                                        ])

def SISO_6user():
    width = 10
    high  = 8
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    cols = 2
    ##=============================== LDPC =========================================

    ##=========================   ===============================
    lb = "联合译码, 有功率分配"
    axs.semilogy(Joint_fastfading_6u_w_powdiv[:, 0], Joint_fastfading_6u_w_powdiv[:, cols], color = 'r', ls = '-',  marker = '*', mfc = 'none', ms = 16, mew = 2, label = lb, zorder = 12)

    ##=========================   ===============================
    lb = "联合译码, 无功率分配"
    axs.semilogy(Joint_fastfading_6u_wo_powdiv[:, 0], Joint_fastfading_6u_wo_powdiv[:, cols], color = 'r', ls = '--',  marker = 'd', mfc = 'none', ms = 12, mew = 2, label = lb, zorder = 12)

    # #=========================  ===============================
    lb = r"SIC, 有功率分配"
    axs.semilogy(sic_fastfading_6u_powerallo[:, 0], sic_fastfading_6u_powerallo[:, cols], color = 'b',ls = '-', lw = 2,  marker = 'v', mfc = 'none', ms = 12, mew = 2, label = lb)

    lb = r"SIC, 无功率分配"
    axs.semilogy(sic_fastfading_6u_equallo[:, 0], sic_fastfading_6u_equallo[:, cols], color = 'b', ls = '-', lw = 2,  marker = 'o', mfc = 'none', ms = 12, mew = 2, label = lb,)

    ##===========================================================
    plt.gca().invert_xaxis()

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    # label
    # font = {'family':'Times New Roman','style':'normal','size':35}
    font = FontProperties(fname=fontpath+"simsun.ttf", size=32)
    axs.set_xlabel("噪声功率谱密度(dBm/Hz)", fontproperties=font, labelpad = 0.2 )
    if cols == 1:
        axs.set_ylabel( "FER",      fontproperties = font, labelpad = 0.2  )# , fontdict = font1
    elif cols == 2:
        axs.set_ylabel( "误码率",      fontproperties = font, labelpad = 0.2  )# , fontdict = font1
    elif cols == 3:
        axs.set_ylabel( "Aggregation error rate",      fontproperties = font )# , fontdict = font1

    # font1 = {'family':'Times New Roman','style':'normal','size':26, }
    font1 = FontProperties(fname=fontpath+"simsun.ttf", size=30)
    legend1 = axs.legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = font1, labelspacing = 0.2, borderpad= 0, handletextpad = 0.1)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    bw = 2
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 30, width = bw)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # [label.set_fontsize(30) for label in labels] #刻度值字号

    # fontt = {'family':'Times New Roman','style':'normal','size':35 }
    # plt.suptitle("2 User, Fastfading, BPSK, [1024, 512]", fontproperties = fontt, )
    out_fig = plt.gcf()

    if cols == 1:
        out_fig.savefig("./Figures/Fig5d_fer.pdf")
    elif cols == 2:
        out_fig.savefig( "./Figures/Fig5d_ber.pdf")
    elif cols == 3:
        out_fig.savefig( "./Figures/Fig5d_aggerr.pdf")
    plt.show()
    plt.close()
    return

SISO_6user()









