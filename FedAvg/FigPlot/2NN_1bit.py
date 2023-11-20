
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
class ResultPlot1bit():
    def __init__(self, ):
        #### savedir
        self.rootdir = f"{user_home}/FedAvg_DataResults/results/"
        self.home = f"{user_home}"
        self.savedir = os.path.join(self.home, 'FedAvg_DataResults/Figures_plot')
        os.makedirs(self.savedir, exist_ok=True)

        # self.base_noniid_blance     = torch.load(os.path.join(self.rootdir, "2023-08-29-14:23:04_FedAvg/TraRecorder.pt"))
        # self.quant8bit              = torch.load(os.path.join(self.rootdir, "2023-08-28-14:53:23_FedAvg/TraRecorder.pt"))
        # self.quant8bit0err          = torch.load(os.path.join(self.rootdir, "2023-08-29-14:57:58_FedAvg/TraRecorder.pt"))   ## P_error = 0.
        # self.quant8bit1err          = torch.load(os.path.join(self.rootdir, "2023-08-30-08:38:33_FedAvg/TraRecorder.pt"))   ## P_error = 0.01
        # self.quant8bit5err          = torch.load(os.path.join(self.rootdir, "2023-08-29-22:49:16_FedAvg/TraRecorder.pt"))   ## P_error = 0.05
        # self.quant8bit10err          = torch.load(os.path.join(self.rootdir, "2023-08-30-01:04:44_FedAvg/TraRecorder.pt"))  ## P_error = 0.1
        # self.quant8bit15err          = torch.load(os.path.join(self.rootdir, "2023-08-30-15:22:06_FedAvg/TraRecorder.pt"))  ## P_error = 0.15
        # self.quant8bit20err          = torch.load(os.path.join(self.rootdir, "2023-08-29-21:23:56_FedAvg/TraRecorder.pt"))  ## P_error = 0.2
        # self.diff_publ_comp05       = torch.load(os.path.join(self.rootdir, "2023-07-09-17:14:50_FedAvg/TraRecorder.pt"))
        # self.diff_publ_comp09       = torch.load(os.path.join(self.rootdir, "2023-07-09-17:49:28_FedAvg/TraRecorder.pt"))
        # self.diff_lcoalDP           = torch.load(os.path.join(self.rootdir, "2023-07-12-15:18:23_FedAvg/TraRecorder.pt"))
        # self.diff_quant8bit         = torch.load(os.path.join(self.rootdir, "2023-07-15-15:10:00_FedAvg/TraRecorder.pt"))
        # self.diff_quant4bit         = torch.load(os.path.join(self.rootdir, "2023-07-15-15:51:50_FedAvg/TraRecorder.pt"))
        # self.diff_quant2bit         = torch.load(os.path.join(self.rootdir, "2023-07-15-16:26:34_FedAvg/TraRecorder.pt"))
        return

    ## lr = 0.004
    def compare_1bit_SR_8(self, model = '2nn', ):
        lw = 2
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        axins = axs.inset_axes((0.62, 0.52, 0.35, 0.3))
        ##================ baseline =========================================
        lb = "Error-free"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        Y1 = savgol_filter(Y1, 25, 3)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = 2)
        axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)
        i += 1

        ##================ 1 bit,  0 error =============================
        lb = "1-bit, SR, "+ r"$\epsilon = 0$"
        ##  2023-09-17-18:15:00_FedAvg  2023-09-17-20:32:40_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-18-22:15:43_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 10)
        Y2 = savgol_filter(Y2, 25, 3)
        Y2 = savgol_filter(Y2, 25, 3)
        Y2 = savgol_filter(Y2, 25, 1)
        Y2 = savgol_filter(Y2, 25, 1)
        axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y2, color = color[i], linestyle = '-', linewidth = 2)
        i += 1

        #================ 1 bit,  0.1 error =============================
        lb = "1-bit, SR, "+ r"$\epsilon = 0.1$"
        ##  2023-09-18-09:09:33_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-19-13:46:10_FedAvg/TraRecorder.pt"))
        Y3 = data[:, 2]
        # Y5 = savgol_filter(Y5, 25, 10)
        Y3 = savgol_filter(Y3, 25, 3)
        Y3 = savgol_filter(Y3, 25, 3)
        Y3 = savgol_filter(Y3, 25, 1)
        Y3 = savgol_filter(Y3, 25, 1)
        axs.plot(data[:, 0], Y3, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y3, color = color[i], linestyle = '-', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.2 error =============================
        lb = "1-bit, SR, "+ r"$\epsilon = 0.2$"
        ##  2023-09-18-10:26:59_FedAvg    2023-09-18-11:31:42_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-19-12:01:49_FedAvg/TraRecorder.pt"))
        Y4 = data[:, 2]

        ## 进行均值滤波去除高频噪声
        # window_size = 20  # 滤波窗口大小
        # Y5 = np.convolve(Y5, np.ones(window_size) / window_size, mode='same')

        Y4 = savgol_filter(Y4, 25, 10)
        Y4 = savgol_filter(Y4, 25, 3)
        Y4 = savgol_filter(Y4, 25, 3)
        Y4 = savgol_filter(Y4, 25, 1)
        Y4 = savgol_filter(Y4, 25, 1)
        axs.plot(data[:, 0], Y4, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y4, color = color[i], linestyle = '-', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.3 error =============================
        lb = "1-bit, SR, "+ r"$\epsilon = 0.3$"
        ##  2023-09-18-10:26:59_FedAvg    2023-09-18-11:31:42_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-19-09:11:00_FedAvg/TraRecorder.pt"))
        Y5 = data[:, 2]
        # Y5 = savgol_filter(Y5, 25, 10)
        Y5 = savgol_filter(Y5, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y5 = savgol_filter(Y5, 25, 1)
        axs.plot(data[:, 0], Y5, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y5, color = color[i], linestyle = '-', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.4 error =============================
        lb = "1-bit, SR, "+ r"$\epsilon = 0.4$"
        ##  2023-09-19-10:13:06_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-19-10:13:06_FedAvg/TraRecorder.pt"))
        Y6 = data[:, 2]
        # 进行均值滤波去除高频噪声
        # window_size = 40  # 滤波窗口大小
        # Y7 = np.convolve(Y7, np.ones(window_size) / window_size, mode='same')

        # Y5 = savgol_filter(Y5, 25, 10)
        Y6 = savgol_filter(Y6, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y6 = savgol_filter(Y6, 25, 1)
        axs.plot(data[:, 0], Y6, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        ##================ 1 bit,  0.5 error =============================
        lb = "1-bit, SR, "+ r"$\epsilon = 0.5$"
        ##  2023-09-19-15:09:36_FedAvg
        data5 = torch.load(os.path.join(self.rootdir, "2023-09-19-15:09:36_FedAvg/TraRecorder.pt"))
        Y8 = data5[:, 2]
        # Y5 = savgol_filter(Y5, 25, 10)
        Y8 = savgol_filter(Y8, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y8 = savgol_filter(Y8, 25, 1)
        axs.plot(data5[:, 0], Y8, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        ##===========================================================

        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
        font1 = {'family':'Times New Roman','style':'normal','size':25, }
        legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.15, 0.08, 0.2, 0.5),  borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.2)
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        bw = 2.5
        axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 32, width = bw)
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # [label.set_fontsize(30) for label in labels] #刻度值字号

        # axs.set_xlim(0.05, 0.94)  #拉开坐标轴范围显示投影
        # axs.set_ylim(0.0, 1.001)  #拉开坐标轴范围显示投影
        # x_major_locator=MultipleLocator(0.1)
        # axs.xaxis.set_major_locator(x_major_locator)
        # y_major_locator=MultipleLocator(0.1)
        # axs.yaxis.set_major_locator(y_major_locator)

        ##==================== mother and son ==================================
        ### 局部显示并且进行连线,方法3   Y3, Y4, Y5, Y6
        zone_and_linked(axs, axins, 800, 850, data[:, 0] , [ Y2, Y3, Y4, Y5 ], 'bottom',x_ratio = 0.3, y_ratio = 0.3)
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
        out_fig.savefig(os.path.join(savepath, f"{model}_1Bit_NonIID_performance.eps"))
        # out_fig.savefig(os.path.join(savepath, f"{model}_Param_debug.pdf") )
        out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_1Bit_NonIID_performance.pdf") )
        plt.show()
        plt.close()
        return


    ## lr = 0.004
    def compare_1bit_SR_7(self, model = '2nn', ):

        lw = 2
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        axins = axs.inset_axes((0.63, 0.52, 0.35, 0.3))
        ##================ baseline =========================================
        lb = "Error-free"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        Y1 = savgol_filter(Y1, 25, 3)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = 2)
        axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)
        i += 1

        ##================ 1 bit,  0 error =============================
        lb = "1-bit, "+ r"$\epsilon = 0$"
        ##  2023-09-17-18:15:00_FedAvg  2023-09-17-20:32:40_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-19-16:21:30_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 10)
        Y2 = savgol_filter(Y2, 25, 3)
        Y2 = savgol_filter(Y2, 25, 3)
        Y2 = savgol_filter(Y2, 25, 1)
        Y2 = savgol_filter(Y2, 25, 1)
        axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y2, color = color[i], linestyle = '-', linewidth = 2)
        i += 1

        # ##================ 1 bit,  0.01 error =============================
        # lb = "1-bit, "+ r"$\epsilon = 0.01$"
        # ##  2023-09-17-18:15:00_FedAvg  2023-09-17-20:32:40_FedAvg
        # data = torch.load(os.path.join(self.rootdir, "2023-10-09-17:24:57_FedAvg/TraRecorder.pt"))
        # Y2 = data[:, 2]
        # # Y2 = savgol_filter(Y2, 25, 10)
        # Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 1)
        # Y2 = savgol_filter(Y2, 25, 1)
        # axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y2, color = color[i], linestyle = '-', linewidth = 2)
        # i += 1

        #================ 1 bit,  0.1 error =============================
        lb = "1-bit, "+ r"$\epsilon = 0.1$"
        ##  2023-09-18-09:09:33_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-19-17:49:14_FedAvg/TraRecorder.pt"))
        Y3 = data[:, 2]
        # Y5 = savgol_filter(Y5, 25, 10)
        Y3 = savgol_filter(Y3, 25, 3)
        Y3 = savgol_filter(Y3, 25, 3)
        Y3 = savgol_filter(Y3, 25, 1)
        Y3 = savgol_filter(Y3, 25, 1)
        axs.plot(data[:, 0], Y3, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y3, color = color[i], linestyle = '-', linewidth = 2)
        i += 1

        # ##================ 1 bit,  0.2 error =============================
        lb = "1-bit, "+ r"$\epsilon = 0.2$"
        ##  2023-09-18-10:26:59_FedAvg    2023-09-18-11:31:42_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-19-19:39:15_FedAvg/TraRecorder.pt"))
        Y4 = data[:, 2]

        ## 进行均值滤波去除高频噪声
        # window_size = 20  # 滤波窗口大小
        # Y5 = np.convolve(Y5, np.ones(window_size) / window_size, mode='same')

        Y4 = savgol_filter(Y4, 25, 10)
        Y4 = savgol_filter(Y4, 25, 3)
        Y4 = savgol_filter(Y4, 25, 3)
        Y4 = savgol_filter(Y4, 25, 1)
        Y4 = savgol_filter(Y4, 25, 1)
        axs.plot(data[:, 0], Y4, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y4, color = color[i], linestyle = '-', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.3 error =============================
        lb = "1-bit, "+ r"$\epsilon = 0.3$"
        ##  2023-09-18-10:26:59_FedAvg    2023-09-18-11:31:42_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-19-20:58:20_FedAvg/TraRecorder.pt"))
        Y5 = data[:, 2]
        # Y5 = savgol_filter(Y5, 25, 10)
        Y5 = savgol_filter(Y5, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y5 = savgol_filter(Y5, 25, 1)
        axs.plot(data[:, 0], Y5, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y5, color = color[i], linestyle = '-', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.4 error =============================
        lb = "1-bit, "+ r"$\epsilon = 0.4$"
        ##  2023-09-19-10:13:06_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-19-22:09:43_FedAvg/TraRecorder.pt"))
        Y6 = data[:, 2]
        # 进行均值滤波去除高频噪声
        # window_size = 40  # 滤波窗口大小
        # Y7 = np.convolve(Y7, np.ones(window_size) / window_size, mode='same')

        # Y5 = savgol_filter(Y5, 25, 10)
        Y6 = savgol_filter(Y6, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y6 = savgol_filter(Y6, 25, 1)
        axs.plot(data[:, 0], Y6, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y6, color = color[i], linestyle = '-', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.5 error =============================
        lb = "1-bit, "+ r"$\epsilon = 0.5$"
        ##  2023-09-18-10:26:59_FedAvg    2023-09-18-11:31:42_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-20-09:55:52_FedAvg/TraRecorder.pt"))
        Y8 = data[:, 2]
        # Y5 = savgol_filter(Y5, 25, 10)
        Y8 = savgol_filter(Y8, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y8 = savgol_filter(Y8, 25, 1)
        axs.plot(data[:, 0], Y8, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        ##===========================================================

        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
        font1 = {'family':'Times New Roman','style':'normal','size':28, }
        legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.14, 0.11, 0.2, 0.5),  borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.2)
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        bw = 2.5
        axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 32, width = bw)
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # [label.set_fontsize(30) for label in labels] #刻度值字号

        # axs.set_xlim(0.05, 0.94)  #拉开坐标轴范围显示投影
        # axs.set_ylim(0.0, 1.001)  #拉开坐标轴范围显示投影
        # x_major_locator=MultipleLocator(0.1)
        # axs.xaxis.set_major_locator(x_major_locator)
        # y_major_locator=MultipleLocator(0.1)
        # axs.yaxis.set_major_locator(y_major_locator)

        ##==================== mother and son ==================================
        ### 局部显示并且进行连线,方法3   Y3, Y4, Y5, Y6
        zone_and_linked(axs, axins, 800, 850, data[:, 0] , [ Y2, Y3, Y4, ], 'bottom',x_ratio = 0.3, y_ratio = 0.3)
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
        out_fig.savefig(os.path.join(savepath, f"{model}_1Bit_NonIID_performance_SR7.eps"))
        # out_fig.savefig(os.path.join(savepath, f"{model}_Param_debug.pdf") )
        out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_1Bit_NonIID_performance_SR7.pdf") )
        plt.show()
        plt.close()
        return


    ## lr = 0.001
    def Onebit_G7(self, model = '2nn', ):
        lw = 2
        width = 10
        high  = 8
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0

        ##================ baseline =========================================
        lb = "Error-free"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        Y1 = savgol_filter(Y1, 25, 3)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = 3)
        ## axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)
        i += 1
        ##================ 8bit, 0 error =============================
        lb = "8-bit, lr=0.001, "+ r"$\epsilon = 0$"
        # ##  2023-09-06-09:28:32_FedAvg   2023-09-18-20:16:16_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-06-09:28:32_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 1)
        axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y2,   color = color[1], linestyle = '-',  linewidth = 2)
        i += 1

        # ##================ 8bit, 0 error =============================
        # lb = "8-bit, lr=0.004, "+ r"$\epsilon = 0$"
        # # ##  2023-09-06-09:28:32_FedAvg   2023-09-18-20:16:16_FedAvg
        # data = torch.load(os.path.join(self.rootdir, "2023-10-14-22:39:17_FedAvg/TraRecorder.pt"))
        # Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 3)
        # # Y2 = savgol_filter(Y2, 25, 1)
        # axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # # axins.plot(data[:, 0], Y2,   color = color[1], linestyle = '-',  linewidth = 2)
        # i += 1

        ##================ 1 bit,  0 error =============================
        lb = "1-bit, lr=0.001, G=7, "+ r"$\epsilon = 0$"
        data = torch.load(os.path.join(self.rootdir, "2023-10-14-17:39:26_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 10)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.1 error =============================
        lb = "1-bit, lr=0.001, G=7, "+ r"$\epsilon = 0.1$"
        data = torch.load(os.path.join(self.rootdir, "2023-10-16-19:28:11_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 10)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.2 error =============================
        lb = "1-bit, lr=0.001, G=7, "+ r"$\epsilon = 0.2$"
        data = torch.load(os.path.join(self.rootdir, "2023-10-16-16:54:14_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 10)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1
        ##================ 1 bit,  0.3 error =============================
        lb = "1-bit, lr=0.001, G=7, "+ r"$\epsilon = 0.3$"
        data = torch.load(os.path.join(self.rootdir, "2023-10-16-09:51:08_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 10)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.4 error =============================
        lb = "1-bit, lr=0.001, G=7, "+ r"$\epsilon = 0.4$"
        data = torch.load(os.path.join(self.rootdir, "2023-10-16-10:58:37_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 10)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1
        ##================ 1 bit,  0.5 error =============================
        lb = "1-bit, lr=0.001, G=7, "+ r"$\epsilon = 0.5$"
        data = torch.load(os.path.join(self.rootdir, "2023-10-16-14:17:59_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 10)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1
        # ##================ 1 bit,  0 error =============================
        # lb = "1-bit, lr=0.001, G=8, "+ r"$\epsilon = 0$"
        # data = torch.load(os.path.join(self.rootdir, "2023-10-14-20:32:14_FedAvg/TraRecorder.pt"))
        # Y = data[:, 2]
        # # Y = savgol_filter(Y, 25, 10)
        # Y = savgol_filter(Y, 25, 3)
        # # Y = savgol_filter(Y, 25, 3)
        # # Y = savgol_filter(Y, 25, 1)
        # # Y = savgol_filter(Y, 25, 1)
        # axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        # ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        # i += 1

        # ##================ 1 bit,  0 error =============================
        # lb = "1-bit, lr=0.004, G=7, "+ r"$\epsilon = 0$"
        # data = torch.load(os.path.join(self.rootdir, "2023-09-19-16:21:30_FedAvg/TraRecorder.pt"))
        # Y = data[:, 2]
        # # Y = savgol_filter(Y, 25, 10)
        # Y = savgol_filter(Y, 25, 3)
        # # Y = savgol_filter(Y, 25, 3)
        # # Y = savgol_filter(Y, 25, 1)
        # # Y = savgol_filter(Y, 25, 1)
        # axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        # ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        # i += 1

        # ##================ 1 bit,  0 error =============================
        # lb = "1-bit, lr=0.004, G=8, "+ r"$\epsilon = 0$"
        # data = torch.load(os.path.join(self.rootdir, "2023-09-18-22:15:43_FedAvg/TraRecorder.pt"))
        # Y = data[:, 2]
        # # Y = savgol_filter(Y, 25, 10)
        # Y = savgol_filter(Y, 25, 3)
        # # Y = savgol_filter(Y, 25, 3)
        # # Y = savgol_filter(Y, 25, 1)
        # # Y = savgol_filter(Y, 25, 1)
        # axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        # ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        # i += 1

        ##===========================================================
        # axs.grid()
        # label
        font = {'family':'Times New Roman','style':'normal','size':30 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
        font1 = {'family':'Times New Roman','style':'normal','size':30, }
        legend1 = axs.legend(loc = 'center right', borderaxespad = 0, edgecolor = 'black', prop = font1,)
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
        savepath = self.savedir
        out_fig.savefig(os.path.join(savepath, f"{model}_1bit_G7.eps") )
        # out_fig.savefig(os.path.join(savepath, f"{model}_Param_debug.pdf") )
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_8bitNonIID_performance.pdf") )
        plt.show()
        plt.close()
        return


    ## lr = 0.001
    def Onebit_G8(self, model = '2nn', ):
        lw = 2
        width = 10
        high  = 8
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0

        ##================ baseline =========================================
        lb = "Error-free"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        Y1 = savgol_filter(Y1, 25, 3)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = 3)
        ## axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)
        i += 1
        ##================ 8bit, 0 error =============================
        lb = "8-bit, lr=0.001, "+ r"$\epsilon = 0$"
        # ##  2023-09-06-09:28:32_FedAvg   2023-09-18-20:16:16_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-06-09:28:32_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 1)
        axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y2,   color = color[1], linestyle = '-',  linewidth = 2)
        i += 1

        ##================ 8bit, 0 error =============================
        lb = "8-bit, lr=0.001, "+ r"$\epsilon = 0.01$"
        # ##  2023-09-06-09:28:32_FedAvg   2023-09-18-20:16:16_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-10-17-22:13:39_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 1)
        axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y2,   color = color[1], linestyle = '-',  linewidth = 2)
        i += 1

        ##================ 1 bit,  0 error =============================
        lb = "1-bit, lr=0.001, G=8, "+ r"$\epsilon = 0$"
        data = torch.load(os.path.join(self.rootdir, "2023-10-14-20:32:14_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 10)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.1 error =============================
        lb = "1-bit, lr=0.001, G=8, "+ r"$\epsilon = 0.1$"
        data = torch.load(os.path.join(self.rootdir, "2023-10-16-22:13:27_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 10)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.2 error =============================
        lb = "1-bit, lr=0.001, G=8, "+ r"$\epsilon = 0.2$"
        data = torch.load(os.path.join(self.rootdir, "2023-10-17-09:52:21_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 10)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1
        ##================ 1 bit,  0.3 error =============================
        lb = "1-bit, lr=0.001, G=8, "+ r"$\epsilon = 0.3$"
        data = torch.load(os.path.join(self.rootdir, "2023-10-17-15:15:17_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 10)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.4 error =============================
        lb = "1-bit, lr=0.001, G=8, "+ r"$\epsilon = 0.4$"
        data = torch.load(os.path.join(self.rootdir, "2023-10-17-19:38:15_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 10)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1
        ##================ 1 bit,  0.5 error =============================
        lb = "1-bit, lr=0.001, G=8, "+ r"$\epsilon = 0.5$"
        data = torch.load(os.path.join(self.rootdir, "2023-10-17-21:00:29_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 10)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1


        # ##================ 1 bit,  0 error =============================
        # lb = "1-bit, lr=0.004, G=7, "+ r"$\epsilon = 0$"
        # data = torch.load(os.path.join(self.rootdir, "2023-09-19-16:21:30_FedAvg/TraRecorder.pt"))
        # Y = data[:, 2]
        # # Y = savgol_filter(Y, 25, 10)
        # Y = savgol_filter(Y, 25, 3)
        # # Y = savgol_filter(Y, 25, 3)
        # # Y = savgol_filter(Y, 25, 1)
        # # Y = savgol_filter(Y, 25, 1)
        # axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        # ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        # i += 1

        # ##================ 1 bit,  0 error =============================
        # lb = "1-bit, lr=0.004, G=8, "+ r"$\epsilon = 0$"
        # data = torch.load(os.path.join(self.rootdir, "2023-09-18-22:15:43_FedAvg/TraRecorder.pt"))
        # Y = data[:, 2]
        # # Y = savgol_filter(Y, 25, 10)
        # Y = savgol_filter(Y, 25, 3)
        # # Y = savgol_filter(Y, 25, 3)
        # # Y = savgol_filter(Y, 25, 1)
        # # Y = savgol_filter(Y, 25, 1)
        # axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        # ##axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        # i += 1

        ##===========================================================
        # axs.grid()
        # label
        font = {'family':'Times New Roman','style':'normal','size':30 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
        font1 = {'family':'Times New Roman','style':'normal','size':30, }
        legend1 = axs.legend(loc = 'center right', borderaxespad = 0, edgecolor = 'black', prop = font1,)
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
        savepath = self.savedir
        out_fig.savefig(os.path.join(savepath, f"{model}_1bit_G8.eps") )
        # out_fig.savefig(os.path.join(savepath, f"{model}_Param_debug.pdf") )
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_8bitNonIID_performance.pdf") )
        plt.show()
        plt.close()
        return


# def main():
pl = ResultPlot1bit() # 1: '2022-10-12-17:38:12'  2:'2022-10-14-09:56:05'

model = "2nn"

# pl.compare_1bit_SR_8(model = model)

# pl.compare_1bit_SR_7(model = model)

pl.Onebit_G7(model = model)

pl.Onebit_G8(model = model)
















