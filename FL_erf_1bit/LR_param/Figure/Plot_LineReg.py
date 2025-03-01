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
now10 = os.path.join(home, "AirFL", "LinearRegression", "2024-09-08-14:53:38")
now30 = os.path.join(home, "AirFL", "LinearRegression", "2024-09-08-15:23:24")
now60 = os.path.join(home, "AirFL", "LinearRegression", "2024-09-08-16:23:14")
# now10 = os.path.join(home, "AirFL", "LinearRegression", "2024-09-07-16:58:24")
# now30 = os.path.join(home, "AirFL", "LinearRegression", "2024-09-07-17:17:36")
# now60 = os.path.join(home, "AirFL", "LinearRegression", "2024-09-07-17:54:11")

mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', ]

color = ['#1E90FF','#FF6347','#00FF00','#0000FF','#4ea142','#FF00FF','#FFA500','#800080','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE','#808000']
lsty = [(0, (3, 10, 1, 10, 1, 10)), (0, (1, 1)), (0, (1, 2)), (0, (5, 1)), (0, (1, 10)), (0, (1, 2)),  (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),  '-', ':', '--', '-.', ]
alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']


## 比较无错的时候，传输梯度、模型、差值三者在不同本地训练轮次时的性能；
def erf_E135_U10(U = 10, ):
    lw = 2
    if U == 10:
        now = now10
    elif U == 30:
        now = now30
    elif U == 60:
        now = now60
    fig, axs = plt.subplots( figsize = (8, 6), constrained_layout=True)

    ## erf, grad, E = 1, num_of_user = 10
    dt = np.load(now + f"/user{U}_bs100_gradient_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='', marker = 'o', ms = 15, mfc='none', mew = 2, mec = 'k', markevery = 100, label = f"erf, grad, E=1, User={U}",  zorder = 1)

    ## erf, diff, E = 1, num_of_user = 10
    dt = np.load(now + f"/user{U}_bs100_diff_E1_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', lw = lw, marker = '*', ms = 13, c = '#1E90FF',  markevery = 100, label = f"erf, diff, E=1, User={U}",  zorder = 2 )

    ## erf, diff, E = 3, num_of_user = 10
    dt = np.load(now + f"/user{U}_bs100_diff_E3_erf_fixedLr/TraRecorder.npy")
    # axs.semilogy(dt[:,0], dt[:,1], linestyle='--', lw = lw, marker = '>', ms = 13, c = '#1E90FF',  markevery = 100, label = f"erf, diff, E=3, User={U}", zorder = 2 )

    ## erf, diff, E = 5, num_of_user = 10
    dt = np.load(now + f"/user{U}_bs100_diff_E5_erf_fixedLr/TraRecorder.npy")
    # axs.semilogy(dt[:,0], dt[:,1], linestyle='-.', lw = lw, marker = 's', ms = 10, c = '#1E90FF',  markevery = 100, label = f"erf, diff, E=5, User={U}", zorder = 2)

    ## erf, model, E = 1, num_of_user = 10
    dt = np.load(now + f"/user{U}_bs100_model_E1_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle=(0,(5,5)), lw = 2,  c = '#FF0000',  label = f"erf, model, E=1, User={U}", zorder = 3  )

    ## erf, model, E = 3, num_of_user = 10
    dt = np.load(now + f"/user{U}_bs100_model_E3_erf_fixedLr/TraRecorder.npy")
    # axs.semilogy(dt[:,0], dt[:,1], linestyle=(0,(5,10)), lw = 3,  c = '#FF0000', label = f"erf, model, E=3, User={U}", zorder = 3  )

    ## erf, model, E = 5, num_of_user = 10
    dt = np.load(now + f"/user{U}_bs100_model_E5_erf_fixedLr/TraRecorder.npy")
    # axs.semilogy(dt[:,0], dt[:,1], linestyle=(0,(1,1)), lw = 2,  c = '#FF0000',  label = f"erf, model, E=5, User={U}", zorder = 3 )

    ##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    bw = 2
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,}
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
    # out_fig.savefig(savedir + f'erf_user{U}_E135.eps', bbox_inches='tight', pad_inches=0,)
    out_fig.savefig(savedir + f'erf_user{U}_E1.eps', bbox_inches='tight', pad_inches=0,)
    plt.show()
    return


def erf_E1_User103060():
    lw = 2
    fig, axs = plt.subplots( figsize = (8, 6), constrained_layout=True)

    ## erf, grad, E = 1, num_of_user = 10
    dt = np.load(now10 + "/user10_bs100_gradient_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='', marker = 'o', ms = 15, mfc='none', mew = 2, mec = 'k', markevery = 100, label = "erf, grad, E=1, User=10",  zorder = 1)

    ## erf, grad, E = 1, num_of_user = 30
    dt = np.load(now30 + "/user30_bs100_gradient_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='', marker = 'd', ms = 15, mfc='none', mew = 2, mec = 'k', markevery = 100, label = "erf, grad, E=1, User=30",  zorder = 1)

    ## erf, grad, E = 1, num_of_user = 60
    dt = np.load(now60 + "/user60_bs100_gradient_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='', marker = 's', ms = 15, mfc='none', mew = 2, mec = 'k', markevery = 100, label = "erf, grad, E=1, User=60",  zorder = 1)


    ## erf, diff, E = 1, num_of_user = 10
    dt = np.load(now10 + "/user10_bs100_diff_E1_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', lw = lw, marker = '*', ms = 13, c = '#1E90FF',  markevery = 100, label = "erf, diff, E=1, User=10",  zorder = 2 )

    ## erf, diff, E = 1, num_of_user = 30
    dt = np.load(now30 + "/user30_bs100_diff_E1_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', lw = lw, marker = '>', ms = 13, c = '#1E90FF',  markevery = 100, label = "erf, diff, E=1, User=30", zorder = 2 )

    ## erf, diff, E = 1, num_of_user = 60
    dt = np.load(now60 + "/user60_bs100_diff_E1_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', lw = lw, marker = 's', ms = 10, c = '#1E90FF',  markevery = 100, label = "erf, diff, E=1, User=60", zorder = 2)

    ## erf, model, E = 1, num_of_user = 10
    dt = np.load(now10 + "/user10_bs100_model_E1_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle=(0,(5,5)), lw = 2,  c = '#FF0000',  label = "erf, model, E=1, User=10", zorder = 3  )

    ## erf, model, E = 1, num_of_user = 30
    dt = np.load(now30 + "/user30_bs100_model_E1_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle=(0,(5,10)), lw = 3,  c = '#FF0000', label = "erf, model, E=1, User=30", zorder = 3  )

    ## erf, model, E = 1, num_of_user = 60
    dt = np.load(now60 + "/user60_bs100_model_E1_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle=(0,(1,1)), lw = 2,  c = '#FF0000',  label = "erf, model, E=1, User=60", zorder = 3 )

    ##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    bw = 2
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,}
    legend1 = axs.legend(loc='best', borderaxespad = 0, edgecolor = 'black', prop = font, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
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
    out_fig.savefig(savedir + 'erf_user136_E1.eps', bbox_inches='tight', pad_inches=0,)
    plt.show()



## 比较无错的时候，三种传输方案在用户数量都为10，不同本地训练轮次时的性能；
# erf_E135_U10(U = 60)

## 比较无错的时候，三种传输方案在本地训练轮1次，不同用户数量时的性能；
# erf_E1_User103060()























