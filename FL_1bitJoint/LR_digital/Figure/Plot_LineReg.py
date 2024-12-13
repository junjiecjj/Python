#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 16:39:04 2024

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter


home = os.path.expanduser('~')
now10 = os.path.join(home, "AirFL", "LinearRegression", "2024-09-10-09:46:36")


mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', ]

color = ['#1E90FF','#FF6347','#00FF00','#0000FF','#4ea142','#FF00FF','#FFA500','#800080','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE','#808000']
lsty = [(0, (3, 10, 1, 10, 1, 10)), (0, (1, 1)), (0, (1, 2)), (0, (5, 1)), (0, (1, 10)), (0, (1, 2)),  (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),  '-', ':', '--', '-.', ]
alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']


## 比较无错的时候，传输梯度、模型、差值三者在不同本地训练轮次时的性能；
def erf_grad():
    lw = 2
    fig, axs = plt.subplots( figsize = (8, 6), constrained_layout=True)

    dt = np.load(now10 + "/user10_bs100_gradient_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', lw = 3, c='b',  label = "erf, grad, User=10, 1-bit quant",  zorder = 1)

    dt = np.load( "/home/jack/AirFL/LinearRegression/2024-09-13-11:23:02/user10_bs100_diff_E1_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', lw = 3, c='k',  label = "erf, diff, User=10, 1-bit quant",  zorder = 1)

    dt = np.load( "/home/jack/AirFL/LinearRegression/2024-09-13-11:23:15/user10_bs100_model_E1_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', lw = 3, c='r',  label = "erf, model, User=10, 1-bit quant",  zorder = 1)

    ##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    bw = 2
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,}
    legend1 = axs.legend(loc='best', borderaxespad = 0, edgecolor = 'black', prop = font, facecolor = 'none',labelspacing = 0.1) ## loc = 'lower left',
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    # frame1.set_facecolor('none')  # 设置图例legend背景透明

    font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
    axs.set_xlabel('Communication round', fontdict = font, )
    axs.set_ylabel('Optimality gap', fontdict = font, )
    # axs.set_title(f"Error-free", fontdict = font,  )

    axs.tick_params(which = 'major', axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)
    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

    # 显示图形
    out_fig = plt.gcf()
    savedir =  '/home/jack/AirFL/LineR_figures/'
    os.makedirs(savedir, exist_ok = True)
    out_fig.savefig(savedir + '1_bit_quant.eps', bbox_inches='tight', pad_inches=0,)
    plt.show()
    return



## 比较无错的时候，三种传输方案在用户数量都为10，不同本地训练轮次时的性能；
erf_grad()




















