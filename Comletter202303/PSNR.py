#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 16:38:48 2025

@author: jack
"""


# import os
# import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
# import torch
from matplotlib.font_manager import FontProperties
# from scipy.signal import savgol_filter


# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "SimSun"
plt.rcParams['mathtext.fontset'] = "stix"
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 1000      # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 24

fontpath = "/usr/share/fonts/truetype/windows/"





LDPC5GNoExtraBer  = np.array([
                    [0.00,  0.1611011601,  0.9957145817 , 0.7335102881 ,  12.8167432369],
                    [0.25,  0.1332193808,  0.9479554520,  0.6414109840,   13.6252593044],
                    [0.50,  0.0893510414,  0.7430479150,  0.4524114524,   15.3443613781],
                    [0.75,  0.0409456574,  0.3859088319,  0.2144924839,   18.7258204968],
                    [1.00,  0.0111369205,  0.1137451437,  0.0593508375,   24.3765885268],
                    [1.25,  0.0016388128,  0.0176109816,  0.0087791875,   32.7401940722],
                    [1.50,  0.0001286017,  0.0014731935,  0.0006898617,   44.4931611384],
                    [1.75,  0.0000053473,  0.0000787361,  0.0000285274,  271.4102207797],
                    [2.00,  0.0000003599,  0.0000088060,  0.0000019559,  338.3438282795],
                    [2.25,  0.0000000086,  0.0000020720,  0.0000000691,  345.8511636695],
                    [2.50,  0.0000000000,  0.0000000000,  0.0000000000,  348.1308036087],
                    ])

LDPC5GWithExtraFer = np.array([
                            [0.00,  0.1819643988,  0.9999886069,  0.7481717519,      12.3198641553],
                            [0.25,  0.1414259893,  0.9973122734,  0.6477140044,      13.3807097245],
                            [0.50,  0.0914532879,  0.9339730709,  0.4541789583,      15.2504190341],
                            [0.75,  0.0413094775,  0.6229880891,  0.2149349594,      18.6881758723],
                            [1.00,  0.0111212304,  0.2138011393,  0.0591450919,      24.3843110888],
                            [1.25,  0.0016320915,  0.0347436561,  0.0087411826,      32.7553098848],
                            [1.50,  0.0001223886,  0.0029155878,  0.0006591106,      44.5811851447],
                            [1.75,  0.0000055336,  0.0001626100,  0.0000305741,     269.6388754169],
                            [2.00,  0.0000002643,  0.0000269291,  0.0000017746,     333.3337478270],
                            [2.25, 0.0000000070,  0.0000020715,  0.0000000561,     347.0169745793],
                            [2.50, 0.0000000383,  0.0000020715,  0.0000002504,     346.9677972167],
                            ])



lw = 2.5
width = 10
high  = 8.5
fig, axs = plt.subplots(1, 1, figsize=(10, 8), constrained_layout = True)

lb = "无额外数据"
axs.semilogy(LDPC5GNoExtraBer[:, 0], LDPC5GNoExtraBer[:, 4], color = 'b', ls = '-', marker = '*', ms = 16, label = lb,  )

lb = "有额外数据, ${\ell}$=2"
axs.semilogy(LDPC5GWithExtraFer[:, 0], LDPC5GWithExtraFer[:, 4], color = 'r', ls = '--',  marker = 'o', mfc = 'none', ms = 20, mew = 2, label = lb, )


font1 = {'family':'Times New Roman','style':'normal','size':26, }
font1 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
legend1 = axs.legend(bbox_to_anchor = (0.46,0.8), borderaxespad = 0, edgecolor = 'black', prop = font1, labelspacing = 0.5, borderpad= 0.7, handletextpad = 0.2)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

font = {'family':'Times New Roman','style':'normal','size':32, }
axs.set_xlabel("SNR(dB)",   fontproperties=font)
# font = FontProperties(fname=fontpath+"simsun.ttf", size=32)
axs.set_ylabel( "PSNR(dB)", fontproperties=font )# , fontdict = font1

## xtick
axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 32, width = 2)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]


## lindwidth
bw = 2
axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细


axs.set_ylim(10, 1100)

axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
out_fig = plt.gcf()


out_fig.savefig("fig8.pdf")

plt.show()
plt.close()



















































































































































































































