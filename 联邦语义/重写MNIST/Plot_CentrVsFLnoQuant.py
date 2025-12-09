

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 08:49:00 2025

@author: jack
"""

import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import FontProperties
from scipy.signal import savgol_filter


# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "SimSun"
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

# 本项目自己编写的库



fontpath = "/usr/share/fonts/truetype/windows/"
mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_',  ]
color = ['#1E90FF','#FF6347','#00FF00','#0000FF','#4ea142','#FF00FF','#FFA500','#800080','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C', '#7B68EE','#808000']

compressrate = [0.2, 0.5, 0.9]
snrtrain = [2, 10, 20]
snrtest = np.arange(-5, 36, 1)

r02_2db_dir = "2023-12-01-09:19:20_FLSemantic"
r05_10db_dir = "2023-11-30-21:34:56_FLSemantic"
r09_20db_dir = "2023-11-30-19:35:46_FLSemantic"

home = os.path.expanduser('~')
rootdir = f"{home}/FL_Sem2026/"
central_dir = rootdir + "MNIST_centralized"
fl_noQ_noniid = rootdir + "MNIST_noIID_noQuant_2025-12-08-13:43:21"

save_dir = rootdir + 'Figs_plot'

##### Centralized VS Federated Learning without quantization
# train psnr vs. round
def psnrVSround():
    lw = 3
    width = 10
    high  = 8.5
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)   # constrained_layout = True

    ##================ Central, R = 0.9, SNRtrain = 20dB =========================================
    trainR = 0.9
    tra_snr = 20
    lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data     = torch.load(os.path.join(central_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"), weights_only=False)
    Y3 = data[:, 5]
    Y3 = savgol_filter(Y3, 25, 6)
    axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '-', linewidth = 2)

    ##================ FL, no Quantization, non-IID, R = 0.9, SNRtrain = 20dB =================================
    trainR = 0.9
    tra_snr = 20
    lb = "联邦式+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data     = torch.load(os.path.join(fl_noQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"), weights_only=False)
    Y3 = data[:, 4]
    Y3 = savgol_filter(Y3, 25, 3)
    axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '--', linewidth = lw)

    ##================ Central, R = 0.5, SNRtrain = 10dB =========================================
    trainR = 0.5
    tra_snr = 10
    lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data     = torch.load(os.path.join(central_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"), weights_only=False)
    Y2 = data[:, 5]
    Y2 = savgol_filter(Y2, 25, 6)
    axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '-', linewidth = lw)

    ##================ FL, no Quantization, non-IID, R = 0.5, SNRtrain = 10dB ===========================
    trainR = 0.5
    tra_snr = 10
    lb = "联邦式+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data     = torch.load(os.path.join(fl_noQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"), weights_only=False)
    Y2 = data[:, 4]
    Y2 = savgol_filter(Y2, 25, 6)
    axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '--', linewidth = lw)

    ##================ Central, R = 0.2, SNRtrain = 2dB =========================================
    trainR = 0.2
    tra_snr = 2
    lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data     = torch.load(os.path.join(central_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"), weights_only=False)
    Y1 = data[:, 5]
    Y1 = savgol_filter(Y1, 25, 6)
    axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw)

    ##================ FL, no Quantization, non-IID, R = 0.2, SNRtrain = 2dB =========================================
    trainR = 0.2
    tra_snr = 2
    lb = "联邦式+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data     = torch.load(os.path.join(fl_noQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"), weights_only=False)
    Y1 = data[:, 4]
    Y1 = savgol_filter(Y1, 25, 6)
    axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '--', linewidth = lw)

    ##===========================================================
    # axs.set_ylim(0, 30.)  #拉开坐标轴范围显示投影
    ## xlabel
    axs.grid(linestyle = (0, (5, 10)), linewidth = 1 )
    font = FontProperties(fname=fontpath+"simsun.ttf", size=32 )
    axs.set_xlabel("通信轮数", fontproperties = font)  #
    font = {'family':'Times New Roman','style':'normal','size':32 }
    axs.set_ylabel( "PSNR(dB)",   fontproperties = font  )# ,   fontdict = font1

    ## legend
    # font1 = {'family':'Times New Roman','style':'normal','size':20, }
    # font1 = FontProperties(fname=fontpath+"simsun.ttf", size=28)   ##   prop = font1,
    legend1 = axs.legend(loc='best', borderaxespad = 0, edgecolor = 'black', facecolor = 'none', labelspacing = 0.2, fontsize = 24) ## loc = 'lower left',
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)

    ## lindwidth
    bw = 2
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    ## xtick
    axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 32, width = 2)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ##===================== mother =========================================
    # fontt = {'family':'Times New Roman','style':'normal','size':30}
    # plt.suptitle("non-IID MNIST,  AutoEncoder", fontproperties = fontt, )
    out_fig = plt.gcf()
    out_fig.savefig(save_dir + "/fig5.pdf")

    plt.show()
    plt.close()
    return

def accVSround( ):
    lw = 3
    width = 10
    high  = 8.5
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True

    ##================ Central, R = 0.9, SNRtrain = 20dB =========================================
    trainR = 0.9
    tra_snr = 20
    lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data     = torch.load(os.path.join(central_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"), weights_only=False)
    Y3 = data[:, 6]
    Y3 = savgol_filter(Y3, 25, 6)
    axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '-', linewidth = lw)

    ##================ FL, no Quantization, non-IID, R = 0.9, SNRtrain = 20dB =========================================
    trainR = 0.9
    tra_snr = 20
    lb = "联邦式+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data     = torch.load(os.path.join(fl_noQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"), weights_only=False)
    Y3 = data[:, 3]
    Y3 = savgol_filter(Y3, 25, 6)
    axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '--', linewidth = lw)

    ##================ Central, R = 0.5, SNRtrain = 10dB =========================================
    trainR = 0.5
    tra_snr = 10
    lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data     = torch.load(os.path.join(central_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"), weights_only=False)
    Y2 = data[:, 6]
    Y2 = savgol_filter(Y2, 25, 6)
    axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '-', linewidth = 2)
    # axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)

    ##================ FL, no Quantization, non-IID, R = 0.5, SNRtrain = 10dB =========================================
    trainR = 0.5
    tra_snr = 10
    lb = "联邦式+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data     = torch.load(os.path.join(fl_noQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"), weights_only=False)
    Y2 = data[:, 3]
    Y2 = savgol_filter(Y2, 25, 6)
    axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '--', linewidth = 2)

    ##================ Central, R = 0.2, SNRtrain = 2dB =========================================
    trainR = 0.2
    tra_snr = 2
    lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data     = torch.load(os.path.join(central_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"), weights_only=False)
    Y1 = data[:, 6]
    Y1 = savgol_filter(Y1, 25, 6)
    axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw)

    ##================ FL, no Quantization, non-IID, R = 0.2, SNRtrain = 2dB =========================================
    trainR = 0.2
    tra_snr = 2
    lb = "联邦式+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data     = torch.load(os.path.join(fl_noQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"), weights_only=False)
    Y1 = data[:, 3]
    Y1 = savgol_filter(Y1, 25, 6)
    axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '--', linewidth = lw)

    ##===========================================================
    # axs.set_xlim(-5, 300.)  #拉开坐标轴范围显示投影
    axs.set_ylim(0.6, 1.)  #拉开坐标轴范围显示投影

    ## xlabel
    axs.grid(linestyle = (0, (5, 10)), linewidth = 1 )
    # font = {'family':'Times New Roman','style':'normal','size':35 }
    font = FontProperties(fname=fontpath+"simsun.ttf", size=32)
    axs.set_xlabel("通信轮数",   fontproperties=font)
    axs.set_ylabel( "学习精度", fontproperties=font)# , fontdict = font1

    ## legend
    # font1 = {'family':'Times New Roman','style':'normal','size':20, }
    # font1 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
    legend1 = axs.legend(loc='best',  borderaxespad = 0, edgecolor = 'black',  facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    # frame1.set_facecolor('none')  # 设置图例legend背景透明

    ## lindwidth
    bw = 2.5
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    ## xtick
    axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 32, width = 2)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # [label.set_fontsize(35) for label in labels] #刻度值字号

    ##===================== mother =========================================
    # fontt = {'family':'Times New Roman','style':'normal','size':30}
    # plt.suptitle("non-IID MNIST, AutoEncoder+LeNet", fontproperties = fontt, )
    out_fig = plt.gcf()
    out_fig.savefig(save_dir + "/fig6.pdf")
    plt.show()
    plt.close()
    return


def PSNRvsTestSNR( ):
    lw = 3
    width = 10
    high  = 8.5
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)  # constrained_layout=True

    testresultdir    = "test_results"
    central_res      = torch.load(os.path.join(central_dir, testresultdir, "TesRecorder_TeMetricLog.pt"), weights_only=False)
    flsem_nQ_noniid  = torch.load(os.path.join(fl_noQ_noniid, testresultdir, "TesRecorder_TeMetricLog.pt"), weights_only=False)

    ##================ Central, R = 0.9, SNRtrain = 20dB =========================================
    trainR = 0.9
    tra_snr = 20
    lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data = central_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
    Y3 = data[:, 2]
    # Y3 = savgol_filter(Y3, 25, 6)
    axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '-', linewidth = lw,)

    ##================ FL, no Quantization, non-IID, R = 0.9, SNRtrain = 20dB =========================================
    trainR = 0.9
    tra_snr = 20
    lb = "联邦式+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
    Y3 = data[:, 2]
    # Y3 = savgol_filter(Y3, 25, 3)
    axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '--', linewidth = lw, )

    ##================ Central, R = 0.5, SNRtrain = 10dB =========================================
    trainR = 0.5
    tra_snr = 10
    lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data = central_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
    Y2 = data[:, 2]
    # Y2 = savgol_filter(Y2, 25, 6)
    axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '-', linewidth = lw,)

    ##================ FL, no Quantization, non-IID, R = 0.5, SNRtrain = 10dB =========================================
    trainR = 0.5
    tra_snr = 10
    lb = "联邦式+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
    Y2 = data[:, 2]
    # Y2 = savgol_filter(Y2, 25, 6)
    axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '--', linewidth =lw, )

    ##================ Central, R = 0.2, SNRtrain = 2dB =========================================
    trainR = 0.2
    tra_snr = 2
    lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data = central_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
    Y1 = data[:, 2]
    # Y1 = savgol_filter(Y1, 25, 6)
    axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw,)

    ##================ FL, no Quantization, non-IID, R = 0.2, SNRtrain = 2dB =========================================
    trainR = 0.2
    tra_snr = 2
    lb = "联邦式+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
    Y1 = data[:, 2]
    # Y1 = savgol_filter(Y1, 25, 6)
    axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '--', linewidth = lw,)

    ##===========================================================

    # axs.set_ylim(10, 32)  #拉开坐标轴范围显示投影

    ## xlabel
    axs.grid(linestyle = (0, (5, 10)), linewidth = 1 )
    font = {'family':'Times New Roman','style':'normal','size':32 }
    # font = FontProperties(fname=fontpath+"simsun.ttf", size=35)
    axs.set_xlabel(r'$\mathrm{{SNR}}_\mathrm{{test}}\mathrm{{(dB)}}$', fontproperties=font)
    axs.set_ylabel( "PSNR(dB)",      fontproperties = font )# , fontdict = font1

    ## legend
    # font1 = {'family':'Times New Roman','style':'normal','size':20, }
    # font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
    # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
    legend1 = axs.legend(loc='best',  borderaxespad = 0, edgecolor = 'black',  facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    # frame1.set_facecolor('none')  # 设置图例legend背景透明

    ## lindwidth
    bw = 2.5
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    ## xtick
    axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 32, width = 2)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # [label.set_fontsize(35) for label in labels] #刻度值字号

    ##===================== mother =========================================
    # fontt = {'family':'Times New Roman','style':'normal','size':30}
    # plt.suptitle("non-IID MNIST,  AutoEncoder", fontproperties = fontt, )
    out_fig = plt.gcf()
    out_fig.savefig(save_dir + "/fig7.pdf")
    plt.show()
    plt.close()
    return

def accvsTestSNR( ):
    ##====================================== mother ===================================
    lw = 3
    width = 10
    high  = 8.5
    fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
    i = 0
    ##======================================= son =====================================
    # axins = axs.inset_axes((0.5, 0.4, 0.45, 0.4))

    testresultdir = "test_results"
    central_res     = torch.load(os.path.join( central_dir, testresultdir, "TesRecorder_TeMetricLog.pt"), weights_only=False)
    flsem_nQ_noniid     = torch.load(os.path.join(fl_noQ_noniid, testresultdir,  "TesRecorder_TeMetricLog.pt"), weights_only=False)

    ##================ Central, R = 0.9, SNRtrain = 20dB =========================================
    trainR = 0.9
    tra_snr = 20
    lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )

    data = central_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
    Y3 = data[:, 1]
    # Y3 = savgol_filter(Y3, 25, 6)
    axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '-', linewidth = lw, )

    ##================ FL, no Quantization, non-IID, R = 0.9, SNRtrain = 20dB =========================================
    trainR = 0.9
    tra_snr = 20
    lb = "联邦式+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )

    data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
    Y3 = data[:, 1]
    # Y3 = savgol_filter(Y3, 25, 3)
    axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = "--", linewidth = lw, )


    ##================ Central, R = 0.5, SNRtrain = 10dB =========================================
    trainR = 0.5
    tra_snr = 10
    lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    # test_res     = torch.load(os.path.join(self.central_dir, testresultdir, "TesRecorder_TeMetricLog.pt"))
    data = central_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
    Y2 = data[:, 1]
    # Y2 = savgol_filter(Y2, 25, 6)
    axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '-', linewidth = lw, )

    ##================ FL, no Quantization, non-IID, R = 0.5, SNRtrain = 10dB =========================================
    trainR = 0.5
    tra_snr = 10
    lb = "联邦式+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )

    data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
    Y2 = data[:, 1]
    # Y2 = savgol_filter(Y2, 25, 6)
    axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = "--", linewidth = lw,)


    ##================ Central, R = 0.2, SNRtrain = 2dB =========================================
    trainR = 0.2
    tra_snr = 2
    lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
    # test_res     = torch.load(os.path.join(self.central_dir, testresultdir, "TesRecorder_TeMetricLog.pt"))
    data = central_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
    Y1 = data[:, 1]
    # Y1 = savgol_filter(Y1, 25, 6)
    axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw,)

    ##================ FL, no Quantization, non-IID, R = 0.2, SNRtrain = 2dB =========================================
    trainR = 0.2
    tra_snr = 2
    lb = "联邦式+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )

    data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
    Y1 = data[:, 1]
    # Y1 = savgol_filter(Y1, 25, 6)
    axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = "--", linewidth = lw,)

    ##===========================================================
    # axs.set_ylim(0, 30.)  #拉开坐标轴范围显示投影

    ## xlabel
    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    font = {'family':'Times New Roman','style':'normal','size':32 }
    axs.set_xlabel(r'$\mathrm{{SNR}}_\mathrm{{test}}\mathrm{{(dB)}}$', fontproperties=font)  #  fontproperties=font
    font = FontProperties(fname=fontpath+"simsun.ttf", size=32)
    axs.set_ylabel( "学习精度", fontproperties=font)# , fontproperties = font fontdict = font1

    legend1 = axs.legend(loc='best',  borderaxespad = 0, edgecolor = 'black', facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    # frame1.set_facecolor('none')  # 设置图例legend背景透明

    ## lindwidth
    bw = 2.5
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    ## xtick
    axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 32, width = 2)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ##===================== mother =========================================
    # fontt = {'family':'Times New Roman','style':'normal','size':30}
    # plt.suptitle("non-IID MNIST,  AutoEncoder+LeNet", fontproperties = fontt, )
    out_fig = plt.gcf()
    out_fig.savefig(save_dir + "/fig8.pdf")
    plt.show()
    plt.close()
    return





## Fig5
psnrVSround()
## Fig6
accVSround()
## Fig 7
PSNRvsTestSNR()
## Fig 8
accvsTestSNR()

















































































































































