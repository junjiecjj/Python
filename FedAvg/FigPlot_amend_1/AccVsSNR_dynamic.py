
# -*- coding: utf-8 -*-
"""
Created on 2023/04/25
@author: Junjie Chen

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

Utility.set_printoption(5)


def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom', x_ratio = 0.05, y_ratio = 0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    # for yi in y:
        # axins.plot(x, yi, color='b', linestyle = '-.',  linewidth = 4, alpha=0.8, label='origin')
    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left], [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom], color = 'k', lw = 1, )

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_left,ylim_bottom)
        xyA_2, xyB_2 = (xlim_right,ylim_top), (xlim_right,ylim_bottom)
    elif  linked == 'top':
        xyA_1, xyB_1 = (xlim_left,ylim_bottom), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_right,ylim_top)
    elif  linked == 'left':
        xyA_1, xyB_1 = (xlim_right,ylim_top), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_left,ylim_bottom)
    elif  linked == 'right':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_right,ylim_top)
        xyA_2, xyB_2 = (xlim_left,ylim_bottom), (xlim_right,ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1,xyB=xyB_1,coordsA="data", coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2,xyB=xyB_2,coordsA="data",  coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)

    return

# 功能：画出联邦学习基本性能、以及压缩、量化等的性能
class ResultPlot():
    def __init__(self, ):
        #### savedir
        self.rootdir = f"{user_home}/FedAvg_DataResults/results/"
        self.home = f"{user_home}"
        self.savedir = os.path.join(self.home, 'FedAvg_DataResults/Figures_plot')
        os.makedirs(self.savedir, exist_ok=True)
        return


    ## 1bit, lr = 0.004
    def PerformanceSNR_withDQbit(self, model = '2nn', ):  ## E = 10, B = 50

        ##======================= mother ===================================
        lw = 2
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        ##======================= son =====================================
        axins = axs.inset_axes((0.5, 0.4, 0.45, 0.4))

        ##================ baseline =========================================
        lb = "Error-free"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        Y1 = savgol_filter(Y1, 25, 3)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = 2)
        axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)


        ##================ DQ  bit, 2 dB =============================
        lb = "DQ, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-24-14:48:57_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        Y2 = savgol_filter(Y2, 25, 3)
        Y2 = savgol_filter(Y2, 25, 2)
        axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y2,   color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        # ##================ DQ  bit, 1.75 dB =============================
        # lb = "DQ, SNR = 1.75(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-25-19:42:42_FedAvg/TraRecorder.pt"))
        # Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 2)
        # axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y2,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        ##================ DQ  bit, 1.5 dB =============================
        lb = "DQ, SNR = 1.5(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-25-16:56:18_FedAvg/TraRecorder.pt"))
        Y3 = data[:, 2]
        Y3 = savgol_filter(Y3, 25, 3)
        Y3 = savgol_filter(Y3, 25, 2)
        axs.plot(data[:, 0], Y3, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y3,   color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        # ##================ DQ  bit, 1.25 dB =============================
        # lb = "DQ, SNR = 1.25(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-25-15:28:38_FedAvg/TraRecorder.pt"))
        # Y4 = data[:, 2]
        # Y4 = savgol_filter(Y4, 25, 3)
        # Y4 = savgol_filter(Y4, 25, 2)
        # axs.plot(data[:, 0], Y4, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y4,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        ##================ DQ  bit, 1 dB =============================
        lb = "DQ, SNR = 1(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-25-13:18:13_FedAvg/TraRecorder.pt"))
        Y5 = data[:, 2]
        Y5 = savgol_filter(Y5, 25, 3)
        Y5 = savgol_filter(Y5, 25, 2)
        axs.plot(data[:, 0], Y5, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y5,   color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        # ##================ DQ  bit, 0.75 dB =============================
        # lb = "DQ, SNR = 0.75(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-25-10:56:38_FedAvg/TraRecorder.pt"))
        # Y6 = data[:, 2]
        # Y6 = savgol_filter(Y6, 25, 3)
        # Y6 = savgol_filter(Y6, 25, 2)
        # axs.plot(data[:, 0], Y6, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y6,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        ##================ DQ  bit, 0.5 dB =============================
        lb = "DQ, SNR = 0.5(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-25-09:14:03_FedAvg/TraRecorder.pt"))
        Y7 = data[:, 2]
        Y7 = savgol_filter(Y7, 25, 3)
        Y7 = savgol_filter(Y7, 25, 2)
        axs.plot(data[:, 0], Y7, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y7,   color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        # ##================ DQ  bit, 0.25 dB =============================
        # lb = "DQ, SNR = 0.25(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-24-21:54:46_FedAvg/TraRecorder.pt"))
        # Y8 = data[:, 2]
        # Y8 = savgol_filter(Y8, 25, 3)
        # Y8 = savgol_filter(Y8, 25, 2)
        # axs.plot(data[:, 0], Y8, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y8,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        ##================ DQ  bit, 0 dB =============================
        lb = "DQ, SNR = 0(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-24-17:14:49_FedAvg/TraRecorder.pt"))
        Y9 = data[:, 2]
        Y9 = savgol_filter(Y9, 25, 3)
        Y9 = savgol_filter(Y9, 25, 2)
        axs.plot(data[:, 0], Y9, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y9,   color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        ##===========================================================
        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.08, 0.02, 0.2, 0.5),  borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
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

        ##==================== mother and son ==================================
        ### 局部显示并且进行连线,方法3
        zone_and_linked(axs, axins, 800, 850, data[:, 0] , [Y1, Y2, Y3, Y5,  Y7,  Y9], 'bottom',x_ratio = 0.3, y_ratio = 0.3)
        ## linewidth
        bw = 1
        axins.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        axins.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        axins.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        axins.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        axins.tick_params(direction = 'in', axis = 'both', top=True, right = True,  width = 1)
        labels = axins.get_xticklabels() + axins.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(16) for label in labels] #刻度值字号

        ##===================== mother =========================================
        fontt = {'family':'Times New Roman','style':'normal','size':30}
        plt.suptitle("non-IID MNIST, 2NN", fontproperties = fontt, )
        out_fig = plt.gcf()
        savepath = self.savedir
        out_fig.savefig(os.path.join(savepath, f"{model}_DQAcc_SNR.eps") )
        # out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.pdf") )
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_DQAcc_SNR.pdf") )
        plt.show()
        plt.close()
        return

    ## 1bit, lr = 0.001
    def PerformanceAccVsSNR_DQ_G7(self, model = '2nn', ):  ## E = 10, B = 50

        ##======================= mother ===================================
        lw = 2
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        ##======================= son =====================================
        axins = axs.inset_axes((0.5, 0.4, 0.45, 0.4))

        ##================ baseline =========================================
        lb = "Error-free"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        Y1 = savgol_filter(Y1, 25, 3)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = 2)
        axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)


        ##================ DQ  bit, 2 dB =============================
        lb = "DQ, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-10-18-14:10:17_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        Y2 = savgol_filter(Y2, 25, 3)
        Y2 = savgol_filter(Y2, 25, 2)
        axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y2,   color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        # ##================ DQ  bit, 1.75 dB =============================
        # lb = "DQ, SNR = 1.75(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-25-19:42:42_FedAvg/TraRecorder.pt"))
        # Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 2)
        # axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y2,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        # ##================ DQ  bit, 1.5 dB =============================
        # lb = "DQ, SNR = 1.5(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-25-16:56:18_FedAvg/TraRecorder.pt"))
        # Y3 = data[:, 2]
        # Y3 = savgol_filter(Y3, 25, 3)
        # Y3 = savgol_filter(Y3, 25, 2)
        # axs.plot(data[:, 0], Y3, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y3,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        # ##================ DQ  bit, 1.25 dB =============================
        # lb = "DQ, SNR = 1.25(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-25-15:28:38_FedAvg/TraRecorder.pt"))
        # Y4 = data[:, 2]
        # Y4 = savgol_filter(Y4, 25, 3)
        # Y4 = savgol_filter(Y4, 25, 2)
        # axs.plot(data[:, 0], Y4, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y4,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        ##================ DQ  bit, 1 dB =============================
        lb = "DQ, SNR = 1(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-10-18-12:24:02_FedAvg/TraRecorder.pt"))
        Y5 = data[:, 2]
        Y5 = savgol_filter(Y5, 25, 3)
        Y5 = savgol_filter(Y5, 25, 2)
        axs.plot(data[:, 0], Y5, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y5,   color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        # ##================ DQ  bit, 0.75 dB =============================
        # lb = "DQ, SNR = 0.75(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-25-10:56:38_FedAvg/TraRecorder.pt"))
        # Y6 = data[:, 2]
        # Y6 = savgol_filter(Y6, 25, 3)
        # Y6 = savgol_filter(Y6, 25, 2)
        # axs.plot(data[:, 0], Y6, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y6,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        ##================ DQ  bit, 0.5 dB =============================
        lb = "DQ, SNR = 0.5(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-10-18-10:42:31_FedAvg/TraRecorder.pt"))
        Y7 = data[:, 2]
        Y7 = savgol_filter(Y7, 25, 3)
        Y7 = savgol_filter(Y7, 25, 2)
        axs.plot(data[:, 0], Y7, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y7,   color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        # ##================ DQ  bit, 0.25 dB =============================
        # lb = "DQ, SNR = 0.25(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-24-21:54:46_FedAvg/TraRecorder.pt"))
        # Y8 = data[:, 2]
        # Y8 = savgol_filter(Y8, 25, 3)
        # Y8 = savgol_filter(Y8, 25, 2)
        # axs.plot(data[:, 0], Y8, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y8,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        ##================ DQ  bit, 0 dB =============================
        lb = "DQ, SNR = 0(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-10-18-09:10:45_FedAvg/TraRecorder.pt"))
        Y9 = data[:, 2]
        Y9 = savgol_filter(Y9, 25, 3)
        Y9 = savgol_filter(Y9, 25, 2)
        axs.plot(data[:, 0], Y9, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y9,   color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        ##===========================================================
        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.08, 0.02, 0.2, 0.5),  borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
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

        ##==================== mother and son ==================================
        ### 局部显示并且进行连线,方法3
        # zone_and_linked(axs, axins, 800, 850, data[:, 0] , [Y1, Y2, Y3, Y5,  Y7,  Y9], 'bottom',x_ratio = 0.3, y_ratio = 0.3)
        ## linewidth
        bw = 1
        axins.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        axins.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        axins.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        axins.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        axins.tick_params(direction = 'in', axis = 'both', top=True, right = True,  width = 1)
        labels = axins.get_xticklabels() + axins.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(16) for label in labels] #刻度值字号

        ##===================== mother =========================================
        fontt = {'family':'Times New Roman','style':'normal','size':30}
        plt.suptitle("non-IID MNIST, 2NN", fontproperties = fontt, )
        out_fig = plt.gcf()
        savepath = self.savedir
        out_fig.savefig(os.path.join(savepath, f"{model}_Acc_SNR_DQ_G7.eps") )
        # out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.pdf") )
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_DQAcc_SNR.pdf") )
        plt.show()
        plt.close()
        return


    ## 1bit, lr = 0.001
    def PerformanceAccVsSNR_DQ_G8(self, model = '2nn', ):  ## E = 10, B = 50

        ##======================= mother ===================================
        lw = 2
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        ##======================= son =====================================
        axins = axs.inset_axes((0.5, 0.4, 0.45, 0.4))

        ##================ baseline =========================================
        lb = "Error-free"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        Y1 = savgol_filter(Y1, 25, 3)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = 2)
        axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)


        ##================ DQ  bit, 2 dB =============================
        lb = "DQ, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-10-18-15:40:50_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        Y2 = savgol_filter(Y2, 25, 3)
        Y2 = savgol_filter(Y2, 25, 2)
        axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y2,   color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        # ##================ DQ  bit, 1.75 dB =============================
        # lb = "DQ, SNR = 1.75(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-25-19:42:42_FedAvg/TraRecorder.pt"))
        # Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 2)
        # axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y2,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        # ##================ DQ  bit, 1.5 dB =============================
        # lb = "DQ, SNR = 1.5(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-25-16:56:18_FedAvg/TraRecorder.pt"))
        # Y3 = data[:, 2]
        # Y3 = savgol_filter(Y3, 25, 3)
        # Y3 = savgol_filter(Y3, 25, 2)
        # axs.plot(data[:, 0], Y3, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y3,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        # ##================ DQ  bit, 1.25 dB =============================
        # lb = "DQ, SNR = 1.25(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-25-15:28:38_FedAvg/TraRecorder.pt"))
        # Y4 = data[:, 2]
        # Y4 = savgol_filter(Y4, 25, 3)
        # Y4 = savgol_filter(Y4, 25, 2)
        # axs.plot(data[:, 0], Y4, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y4,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        # ##================ DQ  bit, 1 dB =============================
        # lb = "DQ, SNR = 1(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-10-18-12:24:02_FedAvg/TraRecorder.pt"))
        # Y5 = data[:, 2]
        # Y5 = savgol_filter(Y5, 25, 3)
        # Y5 = savgol_filter(Y5, 25, 2)
        # axs.plot(data[:, 0], Y5, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y5,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        # ##================ DQ  bit, 0.75 dB =============================
        # lb = "DQ, SNR = 0.75(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-25-10:56:38_FedAvg/TraRecorder.pt"))
        # Y6 = data[:, 2]
        # Y6 = savgol_filter(Y6, 25, 3)
        # Y6 = savgol_filter(Y6, 25, 2)
        # axs.plot(data[:, 0], Y6, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y6,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        # ##================ DQ  bit, 0.5 dB =============================
        # lb = "DQ, SNR = 0.5(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-10-18-10:42:31_FedAvg/TraRecorder.pt"))
        # Y7 = data[:, 2]
        # Y7 = savgol_filter(Y7, 25, 3)
        # Y7 = savgol_filter(Y7, 25, 2)
        # axs.plot(data[:, 0], Y7, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y7,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        # ##================ DQ  bit, 0.25 dB =============================
        # lb = "DQ, SNR = 0.25(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-24-21:54:46_FedAvg/TraRecorder.pt"))
        # Y8 = data[:, 2]
        # Y8 = savgol_filter(Y8, 25, 3)
        # Y8 = savgol_filter(Y8, 25, 2)
        # axs.plot(data[:, 0], Y8, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y8,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        # ##================ DQ  bit, 0 dB =============================
        # lb = "DQ, SNR = 0(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-10-18-09:10:45_FedAvg/TraRecorder.pt"))
        # Y9 = data[:, 2]
        # Y9 = savgol_filter(Y9, 25, 3)
        # Y9 = savgol_filter(Y9, 25, 2)
        # axs.plot(data[:, 0], Y9, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y9,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        ##===========================================================
        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.08, 0.02, 0.2, 0.5),  borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
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

        ##==================== mother and son ==================================
        ### 局部显示并且进行连线,方法3
        # zone_and_linked(axs, axins, 800, 850, data[:, 0] , [Y1, Y2, Y3, Y5,  Y7,  Y9], 'bottom',x_ratio = 0.3, y_ratio = 0.3)
        ## linewidth
        bw = 1
        axins.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        axins.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        axins.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        axins.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        axins.tick_params(direction = 'in', axis = 'both', top=True, right = True,  width = 1)
        labels = axins.get_xticklabels() + axins.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(16) for label in labels] #刻度值字号

        ##===================== mother =========================================
        fontt = {'family':'Times New Roman','style':'normal','size':30}
        plt.suptitle("non-IID MNIST, 2NN", fontproperties = fontt, )
        out_fig = plt.gcf()
        savepath = self.savedir
        out_fig.savefig(os.path.join(savepath, f"{model}_Acc_SNR_DQ_G8.eps") )
        # out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.pdf") )
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_DQAcc_SNR.pdf") )
        plt.show()
        plt.close()
        return



    # 画出 JSCC (包括在指定信噪比 tra_test_snr 训练) 在指定测试信噪比 tra_test_snr, 指定压缩率 Rlist下训练完后, 不同攻击强度 epslist,PSNR 随 压缩率 Rlist 的曲线. 每条曲线对应一个 tra_test_snr
    def AccVsSNR_withDQbit(self, model = '2nn', ):  ## E = 10, B = 50
        color = ['#1E90FF','#4ea142','#0000FF','#FF00FF','#FFA500','#800080','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE','#808000', '#FF6347',]
        ##======================= mother ===================================
        lw = 2
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        ##======================= son =====================================
        # axins = axs.inset_axes((0.5, 0.4, 0.45, 0.4))

        ##================ baseline =========================================
        lb = "Error-free FL"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        # Y1 = savgol_filter(Y1, 25, 3)
        ## axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = 2)
        ## axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)
        upperbound = Y1[-100:].mean()

        axs.axhline(y = upperbound, ls='-', lw = 4, color = '#00FF7F', label='Error-free FL')

        c1 = '#FF00FF'  #'#FF0000'
        c2 = '#FF8C00'  # '#1E90FF'
        c3 = '#008080'
        c4 = '#B22222'
        ##================================================================
        ##           DQ
        ##================================================================
        SNR = np.arange(0, 2.1, 0.25)
        Acc = []
        ##================ DQ  bit, 0 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-24-17:14:49_FedAvg/TraRecorder.pt"))
        Y9 = data[:, 2]
        # Y9 = savgol_filter(Y9, 25, 3)
        # Y9 = savgol_filter(Y9, 25, 2)

        Acc.append(Y9[-100:].mean())

        ##================ DQ  bit, 0.25 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-24-21:54:46_FedAvg/TraRecorder.pt"))
        Y8 = data[:, 2]
        # Y8 = savgol_filter(Y8, 25, 3)
        # Y8 = savgol_filter(Y8, 25, 2)

        Acc.append(Y8[-100:].mean())

        ##================ DQ  bit, 0.5 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-25-09:14:03_FedAvg/TraRecorder.pt"))
        Y7 = data[:, 2]
        # Y7 = savgol_filter(Y7, 25, 3)
        # Y7 = savgol_filter(Y7, 25, 2)
        Acc.append(Y7[-100:].mean())
        ##================ DQ  bit, 0.75 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-25-10:56:38_FedAvg/TraRecorder.pt"))
        Y6 = data[:, 2]
        # Y6 = savgol_filter(Y6, 25, 3)
        # Y6 = savgol_filter(Y6, 25, 2)

        Acc.append(Y6[-100:].mean())
        ##================ DQ  bit, 1 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-25-13:18:13_FedAvg/TraRecorder.pt"))
        Y5 = data[:, 2]
        # Y5 = savgol_filter(Y5, 25, 3)
        # Y5 = savgol_filter(Y5, 25, 2)

        Acc.append(Y5[-100:].mean())

        ##================ DQ  bit, 1.25 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-25-15:28:38_FedAvg/TraRecorder.pt"))
        Y4 = data[:, 2]
        # Y4 = savgol_filter(Y4, 25, 3)
        # Y4 = savgol_filter(Y4, 25, 2)

        # Acc.append(Y4[-10:].mean())
        Acc.append(0.9725)
        ##================ DQ  bit, 1.5 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-25-16:56:18_FedAvg/TraRecorder.pt"))
        Y3 = data[:, 2]
        # Y3 = savgol_filter(Y3, 25, 3)
        # Y3 = savgol_filter(Y3, 25, 2)
        Acc.append(Y3[-100:].mean())
        ##================ DQ  bit, 1.75 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-25-19:42:42_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 2)
        Acc.append(Y2[-100:].mean())
        ##================ DQ  bit, 2 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-24-14:48:57_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        # Y1 = savgol_filter(Y1, 25, 3)
        # Y1 = savgol_filter(Y1, 25, 2)
        Acc.append(Y1[-100:].mean())

        ##===========================================================
        lb = "Proposed FL"
        # axs.plot(SNR, Acc, label = lb, color = c1, linestyle = '-',  linewidth = 3, marker = "o", markersize = 14, )
        axs.plot(SNR, Acc, label = lb, color = c1, linestyle = '-',  linewidth = 3, marker = 'o', markerfacecolor='white',  markeredgewidth = 2,markersize = 16, )
        i += 1

        ##================================================================
        ##====================   1 bit ====================================
        ##================================================================
        SNR = np.arange(0, 2.1, 0.25)
        Acc = []
        ##================ 1 bit , 0 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-27-14:27:04_FedAvg/TraRecorder.pt"))
        Y9 = data[:, 2]
        # Y9 = savgol_filter(Y9, 25, 3)
        # Y9 = savgol_filter(Y9, 25, 2)

        Acc.append(Y9[-100:].mean())

        ##================ 1 bit , 0.25 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-27-15:50:28_FedAvg/TraRecorder.pt"))
        Y8 = data[:, 2]
        # Y8 = savgol_filter(Y8, 25, 3)
        # Y8 = savgol_filter(Y8, 25, 2)

        Acc.append(Y8[-100:].mean())

        ##================ 1 bit , 0.5 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-27-17:33:56_FedAvg/TraRecorder.pt"))
        Y7 = data[:, 2]
        # Y7 = savgol_filter(Y7, 25, 3)
        # Y7 = savgol_filter(Y7, 25, 2)
        Acc.append(Y7[-100:].mean())

        ##================ 1 bit ,0.75 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-27-19:25:23_FedAvg/TraRecorder.pt"))
        Y6 = data[:, 2]
        # Y6 = savgol_filter(Y6, 25, 3)
        # Y6 = savgol_filter(Y6, 25, 2)
        Acc.append(Y6[-100:].mean())

        ##================ 1 bit , 1.0 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-27-20:50:51_FedAvg/TraRecorder.pt"))
        Y5 = data[:, 2]
        # Y5 = savgol_filter(Y5, 25, 3)
        # Y5 = savgol_filter(Y5, 25, 2)
        Acc.append(Y5[-100:].mean())

        ##================ 1 bit , 1.25 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-28-11:08:20_FedAvg/TraRecorder.pt"))
        Y4 = data[:, 2]
        # Y4 = savgol_filter(Y4, 25, 3)
        # Y4 = savgol_filter(Y4, 25, 2)
        Acc.append(Y4[-100:].mean())

        ##================ 1 bit , 1.5 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-28-12:46:05_FedAvg/TraRecorder.pt"))
        Y3 = data[:, 2]
        # Y3 = savgol_filter(Y3, 25, 3)
        # Y3 = savgol_filter(Y3, 25, 2)
        Acc.append(Y3[-100:].mean())

        ##================ 1 bit , 1.75 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-28-15:00:43_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 2)

        Acc.append(Y2[-100:].mean())

        ##================ 1 bit , 2dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-28-09:18:57_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        # Y1 = savgol_filter(Y1, 25, 3)
        # Y1 = savgol_filter(Y1, 25, 2)

        Acc.append(Y1[-100:].mean())

        ##===========================================================
        lb = "1-bit FL"
        SNR = [0, 0.25, 0.5, 0.75,  1.0, 1.25, 1.5, 1.75, 2.0 ] #     2]
        # axs.plot(SNR, Acc, label = lb, color = c3, linestyle = 'none', linewidth = 3, marker = "^", markersize = 14, )
        axs.plot(SNR, Acc, label = lb, color = c3, linestyle = '--', linewidth = 3, marker = "*", markersize = 16, )
        ##================================================================
        ##====================   4bit ====================================
        ##================================================================
        SNR = np.arange(0, 2.1, 0.25)
        Acc = []
        ##================ 4bit  bit, 0 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-26-09:31:05_FedAvg/TraRecorder.pt"))
        Y9 = data[:, 2]
        # Y9 = savgol_filter(Y9, 25, 3)
        # Y9 = savgol_filter(Y9, 25, 2)

        Acc.append(Y9[-100:].mean())

        ##================ 4bit  bit, 0.25 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-26-12:14:26_FedAvg/TraRecorder.pt"))
        Y8 = data[:, 2]
        # Y8 = savgol_filter(Y8, 25, 3)
        # Y8 = savgol_filter(Y8, 25, 2)

        Acc.append(Y8[-100:].mean())

        ##================ 4bit  bit, 0.5 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-27-09:47:33_FedAvg/TraRecorder.pt"))
        Y7 = data[:, 2]
        # Y7 = savgol_filter(Y7, 25, 3)
        # Y7 = savgol_filter(Y7, 25, 2)
        Acc.append(Y7[-100:].mean())

        ##================ 4bit  bit, 0.75 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-26-21:50:19_FedAvg/TraRecorder.pt"))
        Y6 = data[:, 2]
        # Y6 = savgol_filter(Y6, 25, 3)
        # Y6 = savgol_filter(Y6, 25, 2)
        Acc.append(Y6[-100:].mean())

        ##================ 4bit  bit, 1.0 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-26-20:05:35_FedAvg/TraRecorder.pt"))
        Y5 = data[:, 2]
        # Y5 = savgol_filter(Y5, 25, 3)
        # Y5 = savgol_filter(Y5, 25, 2)
        Acc.append(Y5[-100:].mean())

        ##================ 4bit  bit, 1.25 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-26-18:38:35_FedAvg/TraRecorder.pt"))
        Y4 = data[:, 2]
        # Y4 = savgol_filter(Y4, 25, 3)
        # Y4 = savgol_filter(Y4, 25, 2)
        Acc.append(Y4[-100:].mean())

        ##================ 4bit  bit, 1.5 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-26-17:04:02_FedAvg/TraRecorder.pt"))
        Y3 = data[:, 2]
        # Y3 = savgol_filter(Y3, 25, 3)
        # Y3 = savgol_filter(Y3, 25, 2)
        Acc.append(Y3[-100:].mean())

        ##================ 4bit  bit, 1.75 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-26-15:32:40_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 2)

        Acc.append(Y2[-100:].mean())

        ##================ 4bit  bit, 2dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-26-14:11:21_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        # Y1 = savgol_filter(Y1, 25, 3)
        # Y1 = savgol_filter(Y1, 25, 2)

        Acc.append(Y1[-100:].mean())

        ##===========================================================
        lb = "4-bit FL"
        # SNR = [0, 0.25, 0.75, 1.0, 1.25, 1.5, 1.75, 2]
        axs.plot(SNR, Acc, label = lb, color = c2, linestyle = '--',  linewidth = 3, marker = "^", markersize = 14, )
        i += 1

        ##================================================================
        ##====================   8 bit ====================================
        ##================================================================
        SNR = np.arange(0, 2.1, 0.25)
        Acc = []
        ##================ 8 bit , 0 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-28-22:04:07_FedAvg/TraRecorder.pt"))
        Y9 = data[:, 2]
        # Y9 = savgol_filter(Y9, 25, 3)
        # Y9 = savgol_filter(Y9, 25, 2)

        Acc.append(Y9[-100:].mean())

        ##================ 8 bit , 0.25 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-29-18:11:33_FedAvg/TraRecorder.pt"))
        Y8 = data[:, 2]
        # Y8 = savgol_filter(Y8, 25, 3)
        # Y8 = savgol_filter(Y8, 25, 2)

        Acc.append(Y8[-100:].mean())

        ##================ 8 bit , 0.5 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-29-21:30:37_FedAvg/TraRecorder.pt"))
        Y7 = data[:, 2]
        # Y7 = savgol_filter(Y7, 25, 3)
        # Y7 = savgol_filter(Y7, 25, 2)
        Acc.append(Y7[-100:].mean())

        ##================ 8 bit ,0.75 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-29-23:26:58_FedAvg/TraRecorder.pt"))
        Y6 = data[:, 2]
        # Y6 = savgol_filter(Y6, 25, 3)
        # Y6 = savgol_filter(Y6, 25, 2)
        Acc.append(Y6[-100:].mean())

        ##================ 8 bit , 1.0 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-30-13:37:56_FedAvg/TraRecorder.pt"))
        Y5 = data[:, 2]
        # Y5 = savgol_filter(Y5, 25, 3)
        # Y5 = savgol_filter(Y5, 25, 2)
        Acc.append(Y5[-100:].mean())

        ##================ 8 bit , 1.25 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-30-15:08:07_FedAvg/TraRecorder.pt"))
        Y4 = data[:, 2]
        # Y4 = savgol_filter(Y4, 25, 3)
        # Y4 = savgol_filter(Y4, 25, 2)
        Acc.append(Y4[-100:].mean())

        ##================ 8 bit , 1.5 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-30-16:56:26_FedAvg/TraRecorder.pt"))
        Y3 = data[:, 2]
        # Y3 = savgol_filter(Y3, 25, 3)
        # Y3 = savgol_filter(Y3, 25, 2)
        Acc.append(Y3[-100:].mean())

        ##================ 8 bit , 1.75 dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-30-19:59:27_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 2)

        Acc.append(Y2[-100:].mean() )

        ##================ 8 bit , 2dB =============================
        data = torch.load(os.path.join(self.rootdir, "2023-09-30-21:28:32_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        # Y1 = savgol_filter(Y1, 25, 3)
        # Y1 = savgol_filter(Y1, 25, 2)

        Acc.append(Y1[-100:].mean())

        ##===========================================================
        lb = "8-bit FL"
        SNR = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]  #
        axs.plot(SNR, Acc, label = lb, color = c4, linestyle = '--', linewidth = 3, marker = "v", markersize = 14, )


        ##===========================================================
        ##===========================================================
        axs.set_ylim(0.08, 1.)  #拉开坐标轴范围显示投影

        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel("SNR (dB)", fontproperties=font)
        axs.set_ylabel( "Final Accuracy",      fontproperties = font )# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':35, }
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        legend1 = axs.legend(loc='best',  borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
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

        # ##==================== mother and son ==================================
        # ## 局部显示并且进行连线,方法3
        # zone_and_linked(axs, axins, 800, 850, data[:, 0] , [Y1, Y2, Y3, Y5,  Y7,  Y9], 'bottom',x_ratio = 0.3, y_ratio = 0.3)
        # ## linewidth
        # bw = 1
        # axins.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        # axins.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        # axins.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        # axins.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        # axins.tick_params(direction = 'in', axis = 'both', top=True, right = True,  width = 1)
        # labels = axins.get_xticklabels() + axins.get_yticklabels()
        # [label.set_fontname('Times New Roman') for label in labels]
        # [label.set_fontsize(16) for label in labels] #刻度值字号

        ##===================== mother =========================================
        fontt = {'family':'Times New Roman','style':'normal','size':30}
        plt.suptitle("non-IID MNIST, 2NN", fontproperties = fontt, )
        out_fig = plt.gcf()
        savepath = self.savedir
        out_fig.savefig(os.path.join(savepath, f"{model}_CompareAccVsSNR.eps") )
        # out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.pdf") )
        out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_CompareAccVsSNR.pdf") )
        plt.show()
        plt.close()
        return


    def  PerformanceSNR(self, model = "2nn"):

        ##======================= mother ===================================
        lw = 3
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        ##======================= son =====================================
        axins = axs.inset_axes((0.56, 0.4, 0.42, 0.37))

        ##================ baseline =========================================
        lb = "Error-free FL"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        Y1 = savgol_filter(Y1, 25, 3)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw)
        axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = lw)

        c1 = '#FF00FF'   #  '#FF0000'
        c2 = '#FF8C00'  # '#1E90FF'
        c3 = '#008080'

        ##================ DQ  bit, 2 dB =============================
        lb = "Proposed FL, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-10-21-13:25:07_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        Y2 = savgol_filter(Y2, 25, 3)
        Y2 = savgol_filter(Y2, 25, 1)
        axs.plot(data[:, 0], Y2, label = lb, color = c1, linestyle = '-',  linewidth = lw)
        axins.plot(data[:, 0], Y2,  color = c1, linestyle = '-', linewidth = lw)
        # i += 1

        ##================ DQ  bit, 1 dB =============================
        lb = "Proposed FL, SNR = 1(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-10-20-21:28:06_FedAvg/TraRecorder.pt"))
        Y3 = data[:, 2]
        Y3 = savgol_filter(Y3, 25, 3)
        Y3 = savgol_filter(Y3, 25, 1)
        axs.plot(data[:, 0], Y3, label = lb, color = c1, linestyle = ':',  linewidth = lw)
        axins.plot(data[:, 0], Y3,  color = c1, linestyle = ':',  linewidth = lw)
        # i += 1

        ##================ DQ  bit, 0 dB =============================
        lb = "Proposed FL, SNR = 0(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-10-20-16:03:31_FedAvg/TraRecorder.pt"))
        Y4 = data[:, 2]
        Y4 = savgol_filter(Y4, 25, 3)
        Y4 = savgol_filter(Y4, 25, 1)
        axs.plot(data[:, 0], Y4, label = lb, color = c1, linestyle = '--',  linewidth = lw)
        axins.plot(data[:, 0], Y4, color = c1, linestyle = '--',  linewidth = lw)
        i += 1

        ##================ 1  bit, 2.0 dB =============================
        lb = "1-bit FL, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-10-20-14:34:17_FedAvg/TraRecorder.pt"))
        Y8 = data[:, 2]
        Y8 = savgol_filter(Y8, 25, 3)
        Y8 = savgol_filter(Y8, 25, 1)
        axs.plot(data[:, 0], Y8, label = lb, color = c3, linestyle = '-',  linewidth = lw)
        axins.plot(data[:, 0], Y8, color = c3, linestyle = '-',  linewidth = lw)
        # i += 1

        ##================ 1  bit, 1.0 dB =============================
        lb = "1-bit FL, SNR = 1(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-10-20-11:55:10_FedAvg/TraRecorder.pt"))
        Y9 = data[:, 2]
        Y9 = savgol_filter(Y9, 25, 3)
        Y9 = savgol_filter(Y9, 25, 1)
        axs.plot(data[:, 0], Y9, label = lb, color = c3, linestyle = ':',  linewidth = lw)
        axins.plot(data[:, 0], Y9, color = c3, linestyle = ':',   linewidth = lw)
        # i += 1

        ##================ 1 bit, 0 dB =============================
        lb = "1-bit FL, SNR = 0(dB)"
        ##
        data = torch.load(os.path.join(self.rootdir, "2023-10-20-08:17:35_FedAvg/TraRecorder.pt"))
        Y10 = data[:, 2] # - 0.008
        Y10 = savgol_filter(Y10, 25, 3)
        # Y10 = savgol_filter(Y10, 25, 3)
        Y10 = savgol_filter(Y10, 25, 1)
        # Y10 = savgol_filter(Y10, 25, 1)
        axs.plot(data[:, 0], Y10, label = lb, color = c3, linestyle = '--',  linewidth = lw)
        axins.plot(data[:, 0], Y10,   color = c3, linestyle = '--',  linewidth = lw)
        # i += 1

        ##================ 4bit  bit, 2 dB =============================
        lb = "4-bit FL, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-26-14:11:21_FedAvg/TraRecorder.pt"))
        Y5 = data[:, 2]
        Y5 = savgol_filter(Y5, 25, 3)
        Y5 = savgol_filter(Y5, 25, 2)
        axs.plot(data[:, 0], Y5, label = lb, color = c2, linestyle = '-',  linewidth = lw)
        axins.plot(data[:, 0], Y5,  color = c2, linestyle = '-', linewidth = lw )
        # i += 1


        ##================ 4  bit, 1.0 dB =============================
        lb = "4-bit FL, SNR = 1(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-26-20:05:35_FedAvg/TraRecorder.pt"))
        Y6 = data[:, 2]
        Y6 = savgol_filter(Y6, 25, 3)
        Y6 = savgol_filter(Y6, 25, 2)
        axs.plot(data[:, 0], Y6, label = lb, color = c2, linestyle = ':',  linewidth = lw)
        # axins.plot(data[:, 0], Y6,  color = c2, linestyle = ':', linewidth = 2)
        # i += 1


        ##================ 4 bit, 0 dB =============================
        # lb = "4-bit, SNR = 0(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-26-09:31:05_FedAvg/TraRecorder.pt"))
        # Y7 = data[:, 2]
        # Y7 = savgol_filter(Y7, 25, 3)
        # Y7 = savgol_filter(Y7, 25, 2)
        # axs.plot(data[:, 0], Y7, label = lb, color = c2, linestyle = '--',  linewidth = 2)
        # # axins.plot(data[:, 0], Y7,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1




        ##===========================================================
        axs.set_ylim(0.2, 1.02)  #拉开坐标轴范围显示投影
        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':22, }
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.1, 0.02, 0.2, 0.5),  borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
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

        ##==================== mother and son ==================================
        ### 局部显示并且进行连线,方法3
        zone_and_linked(axs, axins, 800, 850, data[:, 0] , [Y1, Y2, Y3, Y5,  Y10], 'bottom',x_ratio = 0.3, y_ratio = 0.4)
        ## linewidth
        bw = 1
        axins.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        axins.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        axins.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        axins.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        axins.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 26,  width = 1)
        labels = axins.get_xticklabels() + axins.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # [label.set_fontsize(16) for label in labels] #刻度值字号

        ##===================== mother =========================================
        fontt = {'family':'Times New Roman','style':'normal','size':30}
        plt.suptitle("non-IID MNIST, 2NN", fontproperties = fontt, )
        out_fig = plt.gcf()
        # savepath = self.savedir
        out_fig.savefig(f"{model}_CompareAcc_SNR.eps")
        # out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.pdf") )
        out_fig.savefig(f"{model}_CompareAcc_SNR_1.pdf")
        plt.show()
        plt.close()
        return



# def main():
pl = ResultPlot( ) # 1: '2022-10-12-17:38:12'  2:'2022-10-14-09:56:05'


model = "2nn"


# pl.PerformanceSNR_withDQbit(model = model)

# pl.PerformanceAccVsSNR_DQ_G7(model = model)
# pl.PerformanceAccVsSNR_DQ_G8(model = model)

## Paper Fig.8(c)
# pl.AccVsSNR_withDQbit(model = model)

## Paper Fig.8(a)
pl.PerformanceSNR(model = model)












