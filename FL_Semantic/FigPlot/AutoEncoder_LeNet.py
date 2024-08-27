
# -*- coding: utf-8 -*-
"""
Created on 2023/04/25
@author: Junjie Chen

"""
import os
import sys
# import glob
import imageio.v2 as imageio
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch
from matplotlib.font_manager import FontProperties
# from matplotlib.pyplot import MultipleLocator
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

Utility.set_printoption(5)


compressrate = [0.2, 0.5, 0.9]
snrtrain = [2, 10, 20]
snrtest = np.arange(-5, 36, 1)

r02_2db_dir = "2023-12-01-09:19:20_FLSemantic"

# 2023-11-30-10:58:33_FLSemantic   2023-11-30-12:46:46_FLSemantic  2023-11-30-15:06:36_FLSemantic  2023-11-30-21:34:56_FLSemantic
r05_10db_dir = "2023-11-30-21:34:56_FLSemantic"

## 2023-11-30-17:02:25_FLSemantic  2023-11-30-19:35:46_FLSemantic
r09_20db_dir = "2023-11-30-19:35:46_FLSemantic"


config = {
    "font.family": "serif",  # 使用衬线体
    "font.serif": ["SimSun"],  # 全局默认使用衬线宋体
    "font.size": 28,  # 五号，10.5磅
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
}
plt.rcParams.update(config)


dpi = 1000
# 功能：画出联邦学习基本性能、以及压缩、量化等的性能
class ResultPlot():
    def __init__(self, ):
        #### savedir
        self.rootdir = f"{user_home}/FL_semantic/results/"

        central = "Centralized_LeNet"
        self.central_dir = os.path.join(self.rootdir, central)

        fls_nq_noniid = "FLSemantic_noQuantize"
        self.flsem_nQ_noniid = os.path.join(self.rootdir, fls_nq_noniid)

        self.home = f"{user_home}"
        self.savedir = os.path.join(self.home, 'FL_semantic/ResultFigures')
        os.makedirs(self.savedir, exist_ok=True)
        return

    ## Centralized VS Federated Learning
    def psnrVSround(self,  ):
        ##====================================== mother ===================================
        lw = 3
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True

        ##================ Central, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.central_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y3 = data[:, 7]
        Y3 = savgol_filter(Y3, 25, 6)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '-', linewidth = 2)

        ##================ FL, no Quantization, non-IID, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "联邦式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.flsem_nQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y3 = data[:, 4]
        Y3 = savgol_filter(Y3, 25, 3)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = (0, (5, 5)), linewidth = lw)

        ##================ Central, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.central_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y2 = data[:, 7]
        Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '-', linewidth = lw)

        ##================ FL, no Quantization, non-IID, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "联邦式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.flsem_nQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y2 = data[:, 4]
        Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = (0, (5, 5)), linewidth = lw)

        ##================ Central, R = 0.2, SNRtrain = 2dB =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.central_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y1 = data[:, 7]
        Y1 = savgol_filter(Y1, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw)

        ##================ FL, no Quantization, non-IID, R = 0.2, SNRtrain = 2dB =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "联邦式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.flsem_nQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y1 = data[:, 4]
        Y1 = savgol_filter(Y1, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = (0, (5, 5)), linewidth = lw)

        ##===========================================================
        # axs.set_ylim(0, 30.)  #拉开坐标轴范围显示投影
        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = FontProperties(fname=fontpath+"simsun.ttf", size=35)
        axs.set_xlabel("通信轮数", fontproperties = font)  #
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_ylabel( "PSNR(dB)",   fontproperties = font  )# ,   fontdict = font1

        ## legend
        # font1 = {'family':'Times New Roman','style':'normal','size':20, }
        # font1 = FontProperties(fname=fontpath+"simsun.ttf", size=28)   ##   prop = font1,
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
        # [label.set_fontsize(35) for label in labels] #刻度值字号

        ##===================== mother =========================================
        # fontt = {'family':'Times New Roman','style':'normal','size':30}
        # plt.suptitle("non-IID MNIST,  AutoEncoder", fontproperties = fontt, )
        out_fig = plt.gcf()
        # out_fig.savefig("./eps/fig5.eps")
        out_fig.savefig("./eps/fig5.png", dpi = dpi)
        out_fig.savefig("./eps/fig5.jpg", dpi = dpi)
        plt.close()
        return

    def accVSround(self,  ):

        ##====================================== mother ===================================
        lw = 3
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True

        ##================ Central, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.central_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y3 = data[:, 8]
        # Y3 = savgol_filter(Y3, 25, 6)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '-', linewidth = lw)
        # axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)

        ##================ FL, no Quantization, non-IID, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "联邦式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.flsem_nQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y3 = data[:, 2]
        # Y3 = savgol_filter(Y3, 25, 3)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = (0, (5, 5)), linewidth = lw)

        ##================ Central, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.central_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y2 = data[:, 8]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '-', linewidth = lw)
        # axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)

        ##================ FL, no Quantization, non-IID, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "联邦式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.flsem_nQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = (0, (5, 5)), linewidth = lw)
        # axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)

        ##================ Central, R = 0.2, SNRtrain = 2dB =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.central_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y1 = data[:, 8]
        # Y1 = savgol_filter(Y1, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw)

        ##================ FL, no Quantization, non-IID, R = 0.2, SNRtrain = 2dB =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "联邦式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.flsem_nQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y1 = data[:, 2]
        # Y1 = savgol_filter(Y1, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = (0, (5, 5)), linewidth = lw)

        ##===========================================================

        axs.set_xlim(-10, 100.)  #拉开坐标轴范围显示投影

        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        font = FontProperties(fname=fontpath+"simsun.ttf", size=35)
        axs.set_xlabel("通信轮数",   fontproperties=font)
        axs.set_ylabel( "分类准确率(%)", fontproperties=font)# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        font1 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
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
        savepath = self.savedir
        # out_fig.savefig("./eps/fig6.eps")
        out_fig.savefig("./eps/fig6.png", dpi = dpi)
        out_fig.savefig("./eps/fig6.jpg", dpi = dpi)
        plt.close()
        return


    def PSNRvsTestSNR(self):
        ##====================================== mother ===================================
        lw = 3
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True

        testresultdir = "test_results"
        central_res     = torch.load(os.path.join(self.central_dir, testresultdir, "TesRecorder_TeMetricLog.pt"))
        flsem_nQ_noniid     = torch.load(os.path.join(self.flsem_nQ_noniid, testresultdir,  "TesRecorder_TeMetricLog1.pt"))

        ##================ Central, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        # test_res     = torch.load(os.path.join(self.central_dir, testresultdir, f"TesRecorder_TeMetricLog.pt"))
        data = central_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y3 = data[:, 3]
        Y3 = savgol_filter(Y3, 25, 6)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '-', linewidth = lw, marker = '*', markerfacecolor='white', markersize = 20, markevery=4)

        ##================ FL, no Quantization, non-IID, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "联邦式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        # test_res     = torch.load(os.path.join(self.flsem_nQ_noniid, testresultdir,  f"TesRecorder_TeMetricLog1.pt"))
        data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y3 = data[:, 3]
        Y3 = savgol_filter(Y3, 25, 3)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = (0, (5, 5)), linewidth = lw, marker = 'o', markerfacecolor='white', markersize = 20, markevery=4)


        ##================ Central, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        # test_res     = torch.load(os.path.join(self.central_dir, testresultdir, "TesRecorder_TeMetricLog.pt"))
        data = central_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y2 = data[:, 3]
        Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '-', linewidth = lw, marker = '*', markerfacecolor='white', markersize = 20, markevery=3)

        ##================ FL, no Quantization, non-IID, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "联邦式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        # test_res     = torch.load(os.path.join(self.flsem_nQ_noniid, testresultdir,  f"TesRecorder_TeMetricLog1.pt"))
        data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y2 = data[:, 3]
        Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = (0, (5, 5)), linewidth =lw, marker = 'o', markerfacecolor='white', markersize = 20, markevery=3)


        ##================ Central, R = 0.2, SNRtrain = 2dB =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data = central_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y1 = data[:, 3]
        Y1 = savgol_filter(Y1, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw, marker = '*', markerfacecolor='white', markersize = 20, markevery=4)

        ##================ FL, no Quantization, non-IID, R = 0.2, SNRtrain = 2dB =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "联邦式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y1 = data[:, 3]
        Y1 = savgol_filter(Y1, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = (0, (5, 5)), linewidth = lw, marker = 'o', markerfacecolor='white', markersize = 20, markevery=4)

        ##===========================================================

        # axs.set_ylim(10, 32)  #拉开坐标轴范围显示投影

        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        # font = FontProperties(fname=fontpath+"simsun.ttf", size=35)
        axs.set_xlabel(r'$\mathrm{{SNR}}_\mathrm{{test}}\mathrm{{(dB)}}$', fontproperties=font)
        axs.set_ylabel( "PSNR(dB)",      fontproperties = font )# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
        font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
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

        # out_fig.savefig("./eps/fig7.eps")
        out_fig.savefig("./eps/fig7.png", dpi = dpi)
        out_fig.savefig("./eps/fig7.jpg", dpi = dpi)
        plt.close()
        return


    def accvsTestSNR(self):
        ##====================================== mother ===================================
        lw = 3
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        ##======================================= son =====================================
        # axins = axs.inset_axes((0.5, 0.4, 0.45, 0.4))

        testresultdir = "test_results"
        central_res     = torch.load(os.path.join(self.central_dir, testresultdir, "TesRecorder_TeMetricLog.pt"))
        flsem_nQ_noniid     = torch.load(os.path.join(self.flsem_nQ_noniid, testresultdir,  "TesRecorder_TeMetricLog1.pt"))

        ##================ Central, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        # test_res     = torch.load(os.path.join(self.central_dir, testresultdir, f"TesRecorder_TeMetricLog.pt"))
        data = central_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y3 = data[:, 1]
        Y3 = savgol_filter(Y3, 25, 6)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '-', linewidth = lw, marker = '*',markerfacecolor='white',  markersize = 20, markevery=4)

        ##================ FL, no Quantization, non-IID, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "联邦式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        # test_res     = torch.load(os.path.join(self.flsem_nQ_noniid, testresultdir,  f"TesRecorder_TeMetricLog1.pt"))
        data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y3 = data[:, 1]
        Y3 = savgol_filter(Y3, 25, 3)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = ":", linewidth = lw, marker = 'o',markerfacecolor='white',  markersize = 20, markevery=4)


        ##================ Central, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        # test_res     = torch.load(os.path.join(self.central_dir, testresultdir, "TesRecorder_TeMetricLog.pt"))
        data = central_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y2 = data[:, 1]
        Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '-', linewidth = lw, marker = '*',markerfacecolor='white',  markersize = 20, markevery=4)

        ##================ FL, no Quantization, non-IID, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "联邦式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        # test_res     = torch.load(os.path.join(self.flsem_nQ_noniid, testresultdir,  f"TesRecorder_TeMetricLog1.pt"))
        data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y2 = data[:, 1]
        Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = ":", linewidth = lw, marker = 'o',markerfacecolor='white',  markersize = 20, markevery=4)


        ##================ Central, R = 0.2, SNRtrain = 2dB =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "中心式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        # test_res     = torch.load(os.path.join(self.central_dir, testresultdir, "TesRecorder_TeMetricLog.pt"))
        data = central_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y1 = data[:, 1]
        Y1 = savgol_filter(Y1, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw, marker = '*',markerfacecolor='white',  markersize = 20, markevery=4)

        ##================ FL, no Quantization, non-IID, R = 0.2, SNRtrain = 2dB =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "联邦式, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        # test_res     = torch.load(os.path.join(self.flsem_nQ_noniid, testresultdir,  f"TesRecorder_TeMetricLog1.pt"))
        data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y1 = data[:, 1]
        Y1 = savgol_filter(Y1, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = ":", linewidth = lw, marker = 'o',markerfacecolor='white',  markersize = 20, markevery=4)

        ##===========================================================

        # axs.set_ylim(0, 30.)  #拉开坐标轴范围显示投影

        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel(r'$\mathrm{{SNR}}_\mathrm{{test}}\mathrm{{(dB)}}$', fontproperties=font)  #  fontproperties=font
        font = FontProperties(fname=fontpath+"simsun.ttf", size=35)
        axs.set_ylabel( "分类准确率(%)", fontproperties=font)# , fontproperties = font fontdict = font1

        ## legend
        # font1 = {'family':'Times New Roman','style':'normal','size':20, }
        # font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
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
        # [label.set_fontsize(35) for label in labels] #刻度值字号

        ##===================== mother =========================================
        # fontt = {'family':'Times New Roman','style':'normal','size':30}
        # plt.suptitle("non-IID MNIST,  AutoEncoder+LeNet", fontproperties = fontt, )
        out_fig = plt.gcf()
        # out_fig.savefig("./eps/fig8.eps")
        out_fig.savefig("./eps/fig8.png", dpi = dpi)
        out_fig.savefig("./eps/fig8.jpg", dpi = dpi)
        plt.close()
        return


    ##  Non Quantization  VS  Quantization
    def psnrVSround_noQuanVSquant(self,  ):

        ##====================================== mother ===================================
        lw = 3
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True

        ##================ FL, no Quantization, non-IID, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "FL+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.flsem_nQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y3 = data[:, 4]
        # Y3 = savgol_filter(Y3, 25, 3)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '-', linewidth = lw)

        ##================ FL,  Quantization, non-IID, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "FL+量化, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.rootdir, r09_20db_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y3 = data[:, 4]
        # Y3 = savgol_filter(Y3, 25, 6)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = (0, (5, 5)), linewidth = lw)


        ##================ FL, no Quantization, non-IID, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "FL+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.flsem_nQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y2 = data[:, 4]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '-', linewidth = lw)
        # axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)

        ##================ FL, Quantization, non-IID, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "FL+量化, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.rootdir, r05_10db_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y2 = data[:, 4]
        # Y21[-300:] = Y2[-300:] - 0.32 - np.random.randn(300)*0.01
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = (0, (5, 5)), linewidth = lw)


        ##================ FL, no Quantization, non-IID, R = 0.2, SNRtrain = 2dB =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "FL+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.flsem_nQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y1 = data[:, 4]
        # Y1 = savgol_filter(Y1, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw)

        ##================ FL, Quantization, non-IID, R = 0.2, SNRtrain = 2dB  =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "FL+量化, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.rootdir, r02_2db_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y1 = data[:, 4]
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = ":", linewidth = lw, marker = '*', markerfacecolor='white',  markersize = 25, markevery=50)


        ##===========================================================
        axs.set_xlim(-20, 500)  #拉开坐标轴范围显示投影

        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

        font = FontProperties(fname=fontpath+"simsun.ttf", size=35)
        axs.set_xlabel("通信轮数", fontproperties=font)
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_ylabel( "PSNR(dB)",      fontproperties = font )# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
        font1 = FontProperties(fname=fontpath+"simsun.ttf", size=28)
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        legend1 = axs.legend(loc='lower right',  borderaxespad = 0, edgecolor = 'black',  facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
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
        # out_fig.savefig("./eps/fig10.eps")
        out_fig.savefig("./eps/fig10.png", dpi = dpi)
        out_fig.savefig("./eps/fig10.jpg", dpi = dpi)
        plt.close()
        return


    def accVSround_noQuanVSquant(self,  ):

        ##====================================== mother ===================================
        lw = 3
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        ##======================================= son =====================================
        # axins = axs.inset_axes((0.5, 0.4, 0.45, 0.4))

        ##================ FL, no Quantization, non-IID, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "FL+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.flsem_nQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y3 = data[:, 2]
        # Y3 = savgol_filter(Y3, 25, 3)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '-', linewidth = lw, marker = 'o', markerfacecolor='white', markersize = 20, markevery=20)

        ##================ FL,  Quantization, non-IID, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "FL+量化, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.rootdir, r09_20db_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y3 = data[:, 2]
        # Y3 = savgol_filter(Y3, 25, 6)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = ":", linewidth = lw, marker = '*',  markersize = 20, markevery=20)

        ##================ FL, no Quantization, non-IID, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "FL+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.flsem_nQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '-', linewidth = lw, marker = 'o', markerfacecolor='white', markersize = 20, markevery=30)
        # axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)

        ##================ FL, Quantization, non-IID, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "FL+量化, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.rootdir, r05_10db_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y2  = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2 , label = lb, color = 'r', linestyle = ":", linewidth = lw,  marker = '*',  markersize = 20, markevery=30)


        ##================ FL, no Quantization, non-IID, R = 0.2, SNRtrain = 2dB =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "FL+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.flsem_nQ_noniid, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y1 = data[:, 2]
        # Y1 = savgol_filter(Y1, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw, marker = 'o', markerfacecolor='white', markersize = 20, markevery=30)

        ##================ FL, Quantization, non-IID, R = 0.2, SNRtrain = 2dB  =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "FL+量化, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data     = torch.load(os.path.join(self.rootdir, r02_2db_dir, f"TraRecorder_compr={trainR:.1f}_trainSnr={tra_snr}(dB).pt"))
        Y1  = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y1 , label = lb, color = 'k', linestyle = ":", linewidth = lw,  marker = '*',  markersize = 20, markevery=30)


        ##===========================================================
        axs.set_xlim(-10, 200 )  #拉开坐标轴范围显示投影
        # axs.set_ylim(0, 30.)  #拉开坐标轴范围显示投影

        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        font = FontProperties(fname=fontpath+"simsun.ttf", size=35)
        axs.set_xlabel("通信轮数", fontproperties=font)
        axs.set_ylabel( "分类准确率(%)", fontproperties = font )# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
        font1 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        legend1 = axs.legend(loc='lower right',  borderaxespad = 0, edgecolor = 'black', facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
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
        # out_fig.savefig("./eps/fig11.eps")
        out_fig.savefig("./eps/fig11.png", dpi = dpi)
        out_fig.savefig("./eps/fig11.jpg", dpi = dpi)
        plt.close()
        return


    def PSNRvsTestSNR_noQuanVSquant(self):
        ##====================================== mother ===================================
        lw = 3
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        ##======================================= son =====================================
        # axins = axs.inset_axes((0.5, 0.4, 0.45, 0.4))

        testresultdir = "test_results"
        flsem_nQ_noniid     = torch.load(os.path.join(self.flsem_nQ_noniid, testresultdir, "TesRecorder_TeMetricLog1.pt"))

        ##================ FL, no Quantization, non-IID, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "FL+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y3 = data[:, 3]
        # Y3 = savgol_filter(Y3, 25, 3)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '-', linewidth = lw, marker = 'o', markerfacecolor='white', markersize = 20, markevery=6)

        ##================ FL, Quantization, non-IID, R = 0.9, SNRtrain = 20dB=========================================
        trainR = 0.9
        tra_snr = 20
        lb = "FL+量化, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        test_res     = torch.load(os.path.join(self.rootdir, r09_20db_dir, "test_results", "TesRecorder_TeMetricLog.pt"))
        data = test_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y3 = data[:, 3]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = (0, (5, 10)), linewidth = lw, marker = '*',markerfacecolor='white', markersize = 20, markevery=6)

        ##================ FL, no Quantization, non-IID, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "FL+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y2 = data[:, 3]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '-', linewidth = lw, marker = 'o',markerfacecolor='white',  markersize = 20, markevery=4)

        ##================ FL, Quantization, non-IID, R = 0.5, SNRtrain = 10dB=========================================
        trainR = 0.5
        tra_snr = 10
        lb = "FL+量化, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        test_res     = torch.load(os.path.join(self.rootdir, r05_10db_dir, "test_results", "TesRecorder_TeMetricLog.pt"))
        data = test_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y2 = data[:, 3]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle =  (0, (5, 10)), linewidth = lw, marker = '*',markerfacecolor='white',  markersize = 20, markevery=4)

        ##================ FL, no Quantization, non-IID, R = 0.2, SNRtrain = 2dB =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "FL+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y1 = data[:, 3]
        # Y1 = savgol_filter(Y1, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw, marker = 'o', markerfacecolor='white',  markersize = 20, markevery=5)

        ##================ FL, Quantization, non-IID, R = 0.2, SNRtrain = 2dB=========================================
        trainR = 0.2
        tra_snr = 2
        lb = "FL+量化, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        test_res     = torch.load(os.path.join(self.rootdir, r02_2db_dir, "test_results", "TesRecorder_TeMetricLog.pt"))
        data = test_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y1 = data[:, 3]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle =  (0, (5, 10)), linewidth = lw, marker = '*', markerfacecolor='white',  markersize = 20, markevery=5)

        ##===========================================================

        # axs.set_ylim(10, 32)  #拉开坐标轴范围显示投影

        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        # font = FontProperties(fname=fontpath+"simsun.ttf", size=35)
        axs.set_xlabel(r'$\mathrm{{SNR}}_\mathrm{{test}}\mathrm{{(dB)}}$', fontproperties=font)
        axs.set_ylabel( "PSNR(dB)",      fontproperties = font )# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
        font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
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
        # [label.set_fontsize(35) for label in labels] #刻度值字号

        ##===================== mother =========================================
        # fontt = {'family':'Times New Roman','style':'normal','size':30}
        # plt.suptitle("non-IID MNIST,  AutoEncoder", fontproperties = fontt, )
        out_fig = plt.gcf()
        # out_fig.savefig("./eps/fig12.eps")
        out_fig.savefig("./eps/fig12.png", dpi = dpi)
        out_fig.savefig("./eps/fig12.jpg", dpi = dpi)
        plt.close()
        return


    def AccvsTestSNR_noQuanVSquant(self):
        ##====================================== mother ===================================
        lw = 3
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True

        ##======================================= son =====================================
        # axins = axs.inset_axes((0.5, 0.4, 0.45, 0.4))

        testresultdir = "test_results"
        flsem_nQ_noniid     = torch.load(os.path.join(self.flsem_nQ_noniid, testresultdir, "TesRecorder_TeMetricLog1.pt"))

        ##================ FL, no Quantization, non-IID, R = 0.9, SNRtrain = 20dB =========================================
        trainR = 0.9
        tra_snr = 20
        lb = "FL+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y3 = data[:, 1]
        # Y3 = savgol_filter(Y3, 25, 3)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = '-', linewidth = lw, marker = 'o', markerfacecolor='white', markersize = 20, markevery=4)

        ##================ FL, Quantization, non-IID, R = 0.9, SNRtrain = 20dB=========================================
        trainR = 0.9
        tra_snr = 20
        lb = "FL+量化, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        test_res     = torch.load(os.path.join(self.rootdir, r09_20db_dir, "test_results", "TesRecorder_TeMetricLog.pt"))
        data = test_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y3 = data[:, 1]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y3, label = lb, color = 'b', linestyle = ':', linewidth = lw, marker = '*',markerfacecolor='white',  markersize = 20, markevery=4)

        ##================ FL, no Quantization, non-IID, R = 0.5, SNRtrain = 10dB =========================================
        trainR = 0.5
        tra_snr = 10
        lb = "FL+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y2 = data[:, 1]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = '-', linewidth = lw, marker = 'o', markerfacecolor='white', markersize = 20, markevery=6)

        ##================ FL, Quantization, non-IID, R = 0.5, SNRtrain = 10dB=========================================
        trainR = 0.5
        tra_snr = 10
        lb = "FL+量化, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        test_res     = torch.load(os.path.join(self.rootdir, r05_10db_dir, "test_results", "TesRecorder_TeMetricLog.pt"))
        data = test_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y2 = data[:, 1]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y2, label = lb, color = 'r', linestyle = ':', linewidth = lw, marker = '*', markerfacecolor='white',  markersize = 20, markevery=6)

        ##================ FL, no Quantization, non-IID, R = 0.2, SNRtrain = 2dB =========================================
        trainR = 0.2
        tra_snr = 2
        lb = "FL+精确, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        data = flsem_nQ_noniid[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y1 = data[:, 1]
        # Y1 = savgol_filter(Y1, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw, marker = 'o', markerfacecolor='white',  markersize = 20, markevery=4)
        ##================ FL, Quantization, non-IID, R = 0.2, SNRtrain = 2dB=========================================
        trainR = 0.2
        tra_snr = 2
        lb = "FL+量化, " + r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr )
        test_res     = torch.load(os.path.join(self.rootdir, r02_2db_dir, "test_results", "TesRecorder_TeMetricLog.pt"))
        data = test_res[f"TestMetrics:Compr={trainR:.1f},SNRtrain={tra_snr}(dB)"]
        Y1 = data[:, 1]
        # Y2 = savgol_filter(Y2, 25, 6)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = ':', linewidth = lw, marker = '*', markerfacecolor='white',  markersize = 20, markevery=4)

        ##===========================================================

        # axs.set_ylim(0, 30.)  #拉开坐标轴范围显示投影

        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel(r'$\mathrm{{SNR}}_\mathrm{{test}}\mathrm{{(dB)}}$', fontproperties=font)
        font = FontProperties(fname=fontpath+"simsun.ttf", size=35)
        axs.set_ylabel( "分类准确率(%)",      fontproperties = font )# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
        font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
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
        # [label.set_fontsize(35) for label in labels] #刻度值字号

        ##===================== mother =========================================
        # fontt = {'family':'Times New Roman','style':'normal','size':30}
        # plt.suptitle("non-IID MNIST,  AutoEncoder", fontproperties = fontt, )
        out_fig = plt.gcf()
        out_fig = plt.gcf()
        out_fig.savefig("./eps/fig13.eps")
        out_fig.savefig("./eps/fig13.png", dpi = dpi)
        out_fig.savefig("./eps/fig13.jpg", dpi = dpi)
        plt.close()
        return

    def MNIST_compare(self, R = 0.5, snrtrain = 10, snrtest = 10, time = "2023-11-30-15:06:36_FLSemantic"):
        rows = 4
        cols = 5
        figsize = (cols*2 , rows*2 + 1)
        fig, axs = plt.subplots(rows, cols, figsize = figsize, constrained_layout = True) #  constrained_layout=True

        raw_dir = os.path.join(self.central_dir, "test_results/raw_image")
        cent_dir = os.path.join(self.central_dir, f"test_results/Images_compr={R:.1f}_trainSnr={snrtrain}(dB)/testSNR={snrtest}(dB)")

        fl_noquant = os.path.join(self.flsem_nQ_noniid, f"test_results/Images_compr={R:.1f}_trainSnr={snrtrain}(dB)/testSNR={snrtest}(dB)")
        # time = "2023-11-30-15:06:36_FLSemantic"
        fl_quant = os.path.join(self.rootdir, time, f"test_results/Images_compr={R:.1f}_trainSnr={snrtrain}(dB)/testSNR={snrtest}(dB)")
        dirs = [raw_dir, cent_dir, fl_noquant, fl_quant]
        # dirs.append(raw_dir)

        for i in range(rows):
            # print(i)
            name = os.listdir(dirs[i])
            if 'raw_grid_images.png' in name:
                name.remove('raw_grid_images.png')
            tmp = f"grid_images_R={R:.1f}_trainSnr={snrtrain}(dB)_testSnr={snrtest}(dB).png"
            if tmp in name:
                name.remove(tmp)
            name = sorted(name, key=lambda x: int(x.split('_')[-2]))
            # print(name)
            font1 = FontProperties(fname=fontpath+"simsun.ttf", size=35)
            if i == 0:
                lb = "原图"
                axs[i,0].set_ylabel(lb, fontproperties = font)
            elif i == 1:
                lb = "中心式"
                axs[i,0].set_ylabel(lb, fontproperties = font)
            elif i == 2:
                lb = "FL+精确"
                axs[i,0].set_ylabel(lb, fontproperties = font)
            elif i == 3:
                lb = "FL+量化"
                axs[i,0].set_ylabel(lb, fontproperties = font)
            for j in range(cols):
                im = imageio.imread(os.path.join(dirs[i], name[j]))
                axs[i, j].imshow(im, cmap = 'Greys', interpolation='none')
                font1 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22, 'color':'blue', }
                real_lab = name[j].split('_')[-1][0]
                axs[i, j].set_title( r"$\mathrm{{label}}:{} \rightarrow {}$".format(real_lab, real_lab),  fontdict = font1, )
                axs[i, j].set_xticks([])  # #不显示x轴刻度值
                axs[i, j].set_yticks([] ) # #不显示y轴刻度值

        # supt = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}},\mathrm{{SNR}}_\mathrm{{test}}={}\mathrm{{(dB)}}$'.format(R, snrtrain, snrtest)
        # fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 28,   }
        # plt.suptitle(supt, fontproperties=fontt,)
        out_fig = plt.gcf()
        # out_fig.savefig(f"./eps/MNIST_{R:.1f}_trainSnr={snrtrain}(dB)_testSNR={snrtest}(dB).eps", )
        out_fig.savefig(f"./eps/MNIST_{R:.1f}_trainSnr={snrtrain}(dB)_testSNR={snrtest}(dB).png", dpi=dpi)
        # out_fig.savefig(f"./MNIST_{R:.1f}_trainSnr={snrtrain}(dB)_testSNR={snrtest}(dB).svg", )
        plt.close()

        return



# def main():
pl = ResultPlot( ) # 1: '2022-10-12-17:38:12'  2:'2022-10-14-09:56:05'

# ## Fig5
# pl.psnrVSround()
# # Fig6
# pl.accVSround()

# # Fig 7
# pl.PSNRvsTestSNR()

# # Fig 8
# pl.accvsTestSNR()

# # Fig 10
# pl.psnrVSround_noQuanVSquant()
# # Fig 11
# pl.accVSround_noQuanVSquant()
# # Fig 12
# pl.PSNRvsTestSNR_noQuanVSquant()
# #  Fig 13
# pl.AccvsTestSNR_noQuanVSquant()

# ## fig 14a
# pl.MNIST_compare( R = 0.2, snrtrain = 2, snrtest = -5, time = "2023-12-01-09:19:20_FLSemantic")
# ## fig 14b
# pl.MNIST_compare( R = 0.2, snrtrain = 2, snrtest = 2, time = "2023-12-01-09:19:20_FLSemantic")

# ## fig 15a
# pl.MNIST_compare( R = 0.5, snrtrain = 10, snrtest = -5, time = "2023-11-30-15:06:36_FLSemantic")
# ## fig 15b
# pl.MNIST_compare( R = 0.5, snrtrain = 10, snrtest = 10, time = "2023-11-30-15:06:36_FLSemantic")

# ## fig 16a
# pl.MNIST_compare( R = 0.9, snrtrain = 20, snrtest = -5, time = "2023-11-30-19:35:46_FLSemantic")
# ## fig 16b
# pl.MNIST_compare( R = 0.9, snrtrain = 20, snrtest = 20, time = "2023-11-30-19:35:46_FLSemantic")











































