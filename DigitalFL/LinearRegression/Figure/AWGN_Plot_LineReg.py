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


mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', ]
color = ['#1E90FF','#FF6347','#00FF00','#0000FF','#4ea142','#FF00FF','#FFA500','#800080','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE','#808000']
lsty = [(0, (3, 10, 1, 10, 1, 10)), (0, (1, 1)), (0, (1, 2)), (0, (5, 1)), (0, (1, 10)), (0, (1, 2)),  (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),  '-', ':', '--', '-.', ]
alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']




def AWGN_SNRsame_Ediff(snr = 3):
    lw = 2
    fig, axs = plt.subplots( figsize = (8, 6), constrained_layout=True)

    ## erf, grad, E = 1, num_of_user = 10
    dt = np.load(now10 + "/user10_bs100_gradient_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-',  c = 'k', marker = 'o', ms = 15, mfc='none', mew = 2, mec = 'k', markevery = 100, label = "grad, E=1, User=10, erf",  zorder = 1)

    ## AWGN, grad, E = 1, num_of_user = 10,
    dt = np.load(now10 + f"/user10_bs100_gradient_awgn_SNR{snr}_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', c = 'k', marker = 'd', ms = 15, mfc='none', mew = 2, mec = 'k', markevery = 100, label = f"grad, E=1, User=10, AWGN, SNR={snr}",  zorder = 1)

    ## AWGN, diff, E = 1, num_of_user = 10
    dt = np.load(now10 + f"/user10_bs100_diff_E1_awgn_SNR{snr}_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', lw = lw, marker = '*', ms = 13, c = '#1E90FF',  markevery = 100, label = f"diff, E=1, User=10, AWGN, SNR={snr}",  zorder = 2 )

    ## erf, diff, E = 3, num_of_user = 10
    dt = np.load(now10 + "/user10_bs100_diff_E3_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', c = '#1E90FF', lw = lw, marker = 'o', ms = 13, mew = 2, mfc='none', markevery = 100, label = "diff, E=3, User=10, erf", zorder = 2 )

    ## AWGN, diff, E = 3, num_of_user = 10
    dt = np.load(now10 + f"/user10_bs100_diff_E3_awgn_SNR{snr}_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', c = '#1E90FF', lw = lw, marker = 'd', ms = 13, mew = 2, mfc='none', markevery = 100, label = f"diff, E=3, User=10, AWGN, SNR={snr}", zorder = 2 )

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
    out_fig.savefig(savedir + f'AWGN_E1_SNR{snr}.eps', bbox_inches='tight', pad_inches=0,)
    return


def AWGN_Esame_SNRcompar(E = 1, U = 10, snr_l = 0, snr_h = 16, ):
    lw = 2
    if U == 10:
        now = now10
    elif U == 30:
        now = now30
    elif U == 60:
        now = now60
    fig, axs = plt.subplots( figsize = (8, 6), constrained_layout=True)

    ####### grad #########
    dt = np.load(now + f"/user{U}_bs100_gradient_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-',  c = 'k', marker = 'o', ms = 15, mfc='none', mew = 2, mec = 'k', markevery = 100, label = f"grad, E=1, User={U}, erf",  zorder = 1)

    dt = np.load(now + f"/user{U}_bs100_gradient_awgn_SNR{snr_l}_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-',  c = 'k', marker = 'd', ms = 15, mfc='none', mew = 2, mec = 'k', markevery = 100, label = f"grad, E=1, User={U}, AWGN,SNR={snr_l}(dB)",  zorder = 1)

    dt = np.load(now + f"/user{U}_bs100_gradient_awgn_SNR{snr_h}_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-',  c = 'k', marker = 'v', ms = 15, mfc='none', mew = 2, mec = 'k', markevery = 100, label = f"grad, E=1, User={U}, AWGN,SNR={snr_h}(dB)",  zorder = 1)

    ########## diff #######
    dt = np.load(now + f"/user{U}_bs100_diff_E{E}_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='--', c = '#1E90FF', lw = lw, marker = 'o', ms = 10, mew = 2,  markevery = 100, label = f"diff, E={E}, User={U}, erf", zorder = 2 )

    dt = np.load(now + f"/user{U}_bs100_diff_E{E}_awgn_SNR{snr_l}_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='--', c = '#1E90FF', lw = lw, marker = 'd', ms = 10, mew = 2, markevery = 100, label = f"diff, E={E}, User={U}, AWGN,SNR={snr_l}(dB)", zorder = 2 )

    dt = np.load(now + f"/user{U}_bs100_diff_E{E}_awgn_SNR{snr_h}_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='--', c = '#1E90FF', lw = lw, marker = 'v', ms = 10, mew = 2,  markevery = 100, label = f"diff, E={E}, User={U}, AWGN,SNR={snr_h}(dB)", zorder = 2 )

    ########## model #######
    # dt = np.load(now + f"/user{U}_bs100_model_E{E}_erf_fixedLr/TraRecorder.npy")
    # axs.semilogy(dt[:,0], dt[:,1], linestyle=(0,(5,1)), c = '#FF6347', lw = lw, marker = 'o', ms = 13, mew = 2, mfc='none', markevery = 100, label = f"model, E={E}, User={U}, erf", zorder = 2 )

    # dt = np.load(now + f"/user{U}_bs100_model_E{E}_awgn_SNR{snr_l}_fixedLr/TraRecorder.npy")
    # axs.semilogy(dt[:,0], dt[:,1], linestyle=(0,(5,1)), c = '#FF6347', lw = lw, marker = 'd', ms = 13, mew = 2, mfc='none', markevery = 100, label = f"model, E={E}, User={U}, SNR={snr_l}(dB)", zorder = 2 )

    # dt = np.load(now + f"/user{U}_bs100_model_E{E}_awgn_SNR{snr_h}_fixedLr/TraRecorder.npy")
    # axs.semilogy(dt[:,0], dt[:,1], linestyle=(0,(5,1)), c = '#FF6347', lw = lw, marker = 'v', ms = 13, mew = 2, mfc='none', markevery = 100, label = f"model, E={E}, User={U}, SNR={snr_h}(dB)", zorder = 2 )

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
    out_fig.savefig(savedir + f'AWGN_E{E}_U{U}_erfSNR_lh.eps', bbox_inches='tight', pad_inches=0,)
    return


def AWGN_Model_Esame_SNRcompar(E = 1, U = 10, snr_l = 0, snr_h = 16, ):
    lw = 2
    if U == 10:
        now = now10
    elif U == 30:
        now = now30
    elif U == 60:
        now = now60
    fig, axs = plt.subplots( figsize = (8, 6), constrained_layout=True)


    ######### model #######
    dt = np.load(now + f"/user{U}_bs100_model_E{E}_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle=(0,(5,1)), c = '#FF6347', lw = lw, marker = 'o', ms = 13, mew = 2, mfc='none', markevery = 100, label = f"model, E={E}, User={U}, erf", zorder = 2 )

    dt = np.load(now + f"/user{U}_bs100_model_E{E}_awgn_SNR{snr_l}_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle=(0,(5,1)), c = '#FF6347', lw = lw, marker = 'd', ms = 13, mew = 2, mfc='none', markevery = 100, label = f"model, E={E}, User={U}, AWGN,SNR={snr_l}(dB)", zorder = 2 )

    dt = np.load(now + f"/user{U}_bs100_model_E{E}_awgn_SNR{snr_h}_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle=(0,(5,1)), c = '#FF6347', lw = lw, marker = 'v', ms = 13, mew = 2, mfc='none', markevery = 100, label = f"model, E={E}, User={U}, AWGN,SNR={snr_h}(dB)", zorder = 2 )

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
    out_fig.savefig(savedir + f'AWGN_Model_E{E}_U{U}_erfSNR_lh.eps', bbox_inches='tight', pad_inches=0,)
    return


def AWGN_E_SNRcompar(E = 3, U = 10, snr_l = 0, snr_h = 16, ):
    lw = 2
    if U == 10:
        now = now10
    elif U == 30:
        now = now30
    elif U == 60:
        now = now60
    fig, axs = plt.subplots( figsize = (8, 6), constrained_layout=True)

    ########## diff #######
    dt = np.load(now + f"/user{U}_bs100_diff_E{E}_erf_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', c = 'k', lw = lw, marker = 'o', ms = 14, mew = 2, mfc='none', markevery = 100, label = f"diff, E={E}, User={U}, erf", zorder = 2 )

    dt = np.load(now + f"/user{U}_bs100_diff_E{E}_awgn_SNR{snr_l}_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-.', c = '#FF00FF', lw = lw, marker = 'o', ms = 14, mew = 2, mfc='none', markevery = 100, label = f"diff, E={E}, User={U}, AWGN,SNR={snr_l}(dB)", zorder = 2 )

    dt = np.load(now + f"/user{U}_bs100_diff_E{E}_awgn_SNR{snr_h}_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='--', c = 'b', lw = lw, marker = 'o', ms = 14, mew = 2, mfc='none', markevery = 100, label = f"diff, E={E}, User={U}, AWGN,SNR={snr_h}(dB)", zorder = 2 )

    ########## model #######

    dt = np.load(now + f"/user{U}_bs100_model_E{E}_awgn_SNR{snr_l}_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', c = '#FF6347', lw = lw, marker = 'o', ms = 13, mew = 2, mfc='none', markevery = 100, label = f"diff, E={E}, User={U}, AWGN,SNR={snr_l}(dB)", zorder = 2 )

    dt = np.load(now + f"/user{U}_bs100_model_E{E}_awgn_SNR{snr_h}_fixedLr/TraRecorder.npy")
    axs.semilogy(dt[:,0], dt[:,1], linestyle='-', c = '#FF6347', lw = lw, marker = 'd', ms = 13, mew = 2, mfc='none', markevery = 100, label = f"diff, E={E}, User={U}, AWGN,SNR={snr_h}(dB)", zorder = 2 )

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
    out_fig.savefig(savedir + f'AWGN_E{E}_U{U}_SNR_lh.eps', bbox_inches='tight', pad_inches=0,)
    return


# ## 比较三种传输方案在同一个信噪比下，相同同用户数量、不同本地训练轮次时的性能；
# AWGN_SNRsame_Ediff(6)

# ## 比较在相同E、相同U下，三种传输方案在地高信噪比下的性能
# AWGN_Esame_SNRcompar(E = 1, U = 10, snr_l = 0, snr_h = 10)
# AWGN_Model_Esame_SNRcompar(E = 1, U = 10, snr_l = 10, snr_h = 20)

## 比较在指定E和U下， diff在不同SNR下的性能
AWGN_E_SNRcompar(E = 5, U = 10, snr_l = 0, snr_h = 10, )























