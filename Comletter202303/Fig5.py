#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 16:37:13 2025

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

LDPC5GNoExtraFer = np.array([
                        [0.000000, 0.9920634921],
                        [0.250000, 0.9363295880],
                        [0.500000, 0.7396449704],
                        [0.750000, 0.3663003663],
                        [1.000000, 0.1166861144],
                        [1.250000, 0.0169877349],
                        [1.500000, 0.0014351897],
                        [1.750000, 0.0000855838],
                        [2.000000, 0.0000092097],
                        ])

LDPC5GFreeRideExtra1BitPayloadBerFer = np.array([
                                [0.000000, 0.1695081791,  0.9956230262],
                                [0.250000, 0.1361787138,  0.9467558722],
                                [0.500000, 0.0902647746,  0.7433074818],
                                [0.750000, 0.0411093734,  0.3859320000],
                                [1.000000, 0.0111053406,  0.1133065000],
                                [1.250000, 0.0016417010,  0.0176497750],
                                [1.500000, 0.0001266064,  0.0014556500],
                                [1.750000, 0.0000051437,  0.0000755000],
                                [2.000000, 0.0000002755,  0.0000090000],
                                ])

LDPC5GFreeRideExtra2BitPayloadBerFer = np.array([
                        [0.000000, 0.1838549728,  0.9952159842],
                        [0.250000, 0.1418919170,  0.9472420121],
                        [0.500000, 0.0914873711,  0.7425752780],
                        [0.750000, 0.0412932836,  0.3859083010],
                        [1.000000, 0.0111733911,  0.1136940000],
                        [1.250000, 0.0016431222 , 0.0176500000],
                        [1.500000, 0.0001304365,  0.0014870000],
                        [1.750000, 0.0000052276,  0.0000810000],
                        ])

## 5G LDPC 单独统计信息位的前2bit
LDPC5GNoExtraBer2bit  = np.array([
                        [0.000000, 0.2190146266, 0.3849114704],
                        [0.250000, 0.1686355582, 0.2953337271],
                        [0.500000, 0.1020220588, 0.1838235294],
                        [0.750000, 0.0439350525, 0.0795924865],
                        [1.000000, 0.0114697480, 0.0213192342],
                        [1.250000, 0.0017325478, 0.0031673635],
                        [1.500000, 0.0001486809, 0.0002688623],
                        [1.750000, 0.0000069787, 0.0000127660],
                        [2.000000, 0.0000003115, 0.0000006230],
                        ])


#第一组数据，第一列是Eb/N0或SNR, 第二列是BER，第三列是WER，下同。
LDPC5GFreeRideExtra1BitExtraBerFer = np.array([
                        [0.000000,  0.0277023658, 0.0277023658],
                        [0.250000,  0.0096264921,  0.0096264921],
                        [0.500000,  0.0023673570,  0.0023673570],
                        [0.750000,  0.0003360000,  0.0003360000],
                        [1.000000,  0.0000200000,  0.0000200000],
                        [1.250000,  0.0000009000,  0.0000009000],
                        # % 1.500000  0.0000000000  0.0000000000
                        # % 1.750000  0.0000000000  0.0000000000
                        ])

LDPC5GFreeRideExtra2BitExtraBerFer = np.array([
                        [0.000000,  0.0473476854,  0.0703531729],
                        [0.250000,  0.0179997841,  0.0269861831],
                        [0.500000,  0.0039655110,  0.0060634724],
                        [0.750000,  0.0006277281,  0.0009383081],
                        [1.000000,  0.0000550000,  0.0000770000],
                        [1.250000,  0.0000014500,  0.0000020500],
                        # [1.500000,  0.0000000000,  0.0000000000],
                        ])

lw = 2.5
width = 10
high  = 8.5
fig, axs = plt.subplots(1, 1, figsize=(10, 8), constrained_layout = True)

lb = "负载数据, LDPC码"
axs.semilogy(LDPC5GNoExtraFer[:, 0], LDPC5GNoExtraFer[:, 1], color = 'm', ls = '--', label = lb,  zorder = 12)

lb = "负载数据, 便车码, ${\ell}$=1"
axs.semilogy(LDPC5GFreeRideExtra1BitPayloadBerFer[:, 0], LDPC5GFreeRideExtra1BitPayloadBerFer[:, 2], color = 'k', ls = 'none',  marker = '*', mfc = 'none', ms = 14, mew = 2, label = lb, zorder = 12)

# #=========================  ===============================
lb = r"负载数据, 便车码, ${\ell}$=2"
axs.semilogy(LDPC5GFreeRideExtra2BitPayloadBerFer[:, 0], LDPC5GFreeRideExtra2BitPayloadBerFer[:, 2], color = 'r', ls = 'none', marker = 'o', mfc = 'none', ms = 16, mew = 2, label = lb)

lb = r"额外数据, 传统, ${\ell}$=2"
axs.semilogy(LDPC5GNoExtraBer2bit[:, 0], LDPC5GNoExtraBer2bit[:, 2], color = '#00841a', ls = '--', lw = lw,  marker = '*', mfc = 'none', ms = 16, mew = 2, label = lb,)

lb = r"额外数据, 便车码, ${\ell}$=1"
axs.semilogy(LDPC5GFreeRideExtra1BitExtraBerFer[:, 0], LDPC5GFreeRideExtra1BitExtraBerFer[:, 2], color = 'b', ls = '-', lw = lw,  marker = 'd', mfc = 'none', ms = 12, mew = 2, label = lb,)

lb = r"额外数据, 便车码, ${\ell}$=2"
axs.semilogy(LDPC5GFreeRideExtra2BitExtraBerFer[:, 0], LDPC5GFreeRideExtra2BitExtraBerFer[:, 2], color = 'b', ls = '-', lw = lw,  marker = '^', mfc = 'none', ms = 12, mew = 2, label = lb,)

font1 = {'family':'Times New Roman','style':'normal','size':26, }
font1 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
legend1 = axs.legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = font1, labelspacing = 0.2, borderpad= 0, handletextpad = 0.1)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

font = {'family':'Times New Roman','style':'normal','size':32, }
axs.set_xlabel("SNR(dB)",   fontproperties=font)
font = FontProperties(fname=fontpath+"simsun.ttf", size=32)
axs.set_ylabel( "误帧率", fontproperties=font )# , fontdict = font1

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


# axs.set_xlim(-1, 2.1)

axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
out_fig = plt.gcf()


out_fig.savefig("fig5.pdf")

plt.show()
plt.close()





























































































































































































































































