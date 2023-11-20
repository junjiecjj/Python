
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

    # 画出 JSCC (包括在指定信噪比 tra_test_snr 训练) 在指定测试信噪比 tra_test_snr, 指定压缩率 Rlist下训练完后, 不同攻击强度 epslist,PSNR 随 压缩率 Rlist 的曲线. 每条曲线对应一个 tra_test_snr
    def compare_8bit(self, model = '2nn', ):  ## E = 10, B = 128
        lw = 2
        width = 10
        high  = 8
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True

        ##================ baseline =========================================
        lb = "Baseline, without quantize"
        base_noniid_blance     = torch.load(os.path.join(self.rootdir, "2023-08-29-14:23:04_FedAvg/TraRecorder.pt"))
        Y = base_noniid_blance[:, 2]
        Y = savgol_filter(Y, 25, 3)
        axs.plot(base_noniid_blance[:, 0], Y, label = lb, color = color[0], linestyle = '-', linewidth = 5)

        # lb = "mask:0.7"
        # axs.plot(self.diff_publ_mask07[:, 0], self.diff_publ_mask07[:, 2], label = lb, color = 'b', linestyle = '-', )
        # lb = "mask:0.8"
        # axs.plot(self.diff_publ_mask08[:, 0], self.diff_publ_mask08[:, 2], label = lb, color = 'b', linestyle = '--', )

        # lb = "compress:0.5"
        # axs.plot(self.diff_publ_comp05[:, 0], self.diff_publ_comp05[:, 2], label = lb, color = '#FFA500', linestyle = '-', )
        # lb = "compress:0.9"
        # axs.plot(self.diff_publ_comp09[:, 0], self.diff_publ_comp09[:, 2], label = lb, color = '#FFA500', linestyle = '--', )

        # lb = "local DP"
        # axs.plot(self.diff_lcoalDP[:, 0], self.diff_lcoalDP[:, 2], label = lb, color = 'r', linestyle = '-', )

        # lb = "8 bit"
        # Y = self.quant8bit[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        # axs.plot(self.quant8bit[:, 0], Y, label = lb, color = color[1], linestyle = '-',  linewidth = 2)
        ##================ 8bit, 0 error =============================
        lb = "8 bit, "+ r"$P_\mathrm{bit\_flip} = 0$"
        quant8bit0err = torch.load(os.path.join(self.rootdir, "2023-08-29-14:57:58_FedAvg/TraRecorder.pt"))   ## P_error = 0.  2023-08-29-14:57:58_FedAvg
        Y = quant8bit0err[:, 2]
        Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 2)
        axs.plot(quant8bit0err[:, 0], Y, label = lb, color = color[1], linestyle = '-',  linewidth = 2)

        ##================ 8bit, 0.0001 error =============================
        quant8bit001err = torch.load(os.path.join(self.rootdir, "2023-08-30-13:17:42_FedAvg/TraRecorder.pt"))   ## P_error = 0.005
        lb = "8 bit, "+ r"$P_\mathrm{bit\_flip} = 0.0001$"
        Y = quant8bit001err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y = savgol_filter(Y, 25, 1)
        axs.plot(quant8bit001err[:, 0], Y, label = lb, color = color[2], linestyle = '-',  linewidth = 1)

        ##================ 8bit, 0.001 error =============================
        quant8bit01err = torch.load(os.path.join(self.rootdir, "2023-08-30-11:22:33_FedAvg/TraRecorder.pt"))   ## P_error = 0.005
        lb = "8 bit, "+ r"$P_\mathrm{bit\_flip} = 0.001$"
        Y = quant8bit01err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant8bit01err[:, 0], Y, label = lb, color = color[3], linestyle = '-',  linewidth = lw)


        ##================ 8bit, 0.005 error =============================
        quant8bit05err = torch.load(os.path.join(self.rootdir, "2023-08-30-10:03:03_FedAvg/TraRecorder.pt"))   ## P_error = 0.005
        lb = "8 bit, "+ r"$P_\mathrm{bit\_flip} = 0.005$"
        Y = quant8bit05err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y = savgol_filter(Y, 25, 1)
        axs.plot(quant8bit05err[:, 0], Y, label = lb, color = color[4], linestyle = '-',  linewidth = lw)

        ##================ 8bit, 0.01 error =============================
        lb = "8 bit, "+ r"$P_\mathrm{bit\_flip} = 0.01$"
        quant8bit1err   = torch.load(os.path.join(self.rootdir, "2023-08-30-08:38:33_FedAvg/TraRecorder.pt"))   ## P_error = 0.01
        Y = quant8bit1err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant8bit1err[:, 0], Y, label = lb, color = color[5], linestyle = '-',  linewidth = lw)

        ##================ 8bit, 0.05 error =============================
        lb = "8 bit, "+ r"$P_\mathrm{bit\_flip} = 0.05$"
        ## 2023-08-29-13:06:56_FedAvg
        quant8bit5err  = torch.load(os.path.join(self.rootdir, "2023-08-29-22:49:16_FedAvg/TraRecorder.pt"))   ## P_error = 0.05
        Y = quant8bit5err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant8bit5err[:, 0], Y, label = lb, color = color[6], linestyle = '-',  linewidth = lw)

        # quant8bit5err   = torch.load(os.path.join(self.rootdir, "2023-08-29-13:06:56_FedAvg/TraRecorder.pt"))   ## P_error = 0.05
        # lb = "8 bit, "+ r"$P_\mathrm{bit\_flip} = 0.05$"
        # Y = quant8bit5err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        # # Y = savgol_filter(Y, 25, 1)
        # axs.plot(quant8bit5err[:, 0], Y, label = lb, color = 'yellow', linestyle = '-',  linewidth = 2)

        ##================ 8bit, 0.1 error =============================
        lb = "8 bit, "+ r"$P_\mathrm{bit\_flip} = 0.1$"
        quant8bit10err  = torch.load(os.path.join(self.rootdir, "2023-08-30-01:04:44_FedAvg/TraRecorder.pt"))  ## P_error = 0.1
        Y = quant8bit10err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant8bit10err[:, 0], Y, label = lb, color = color[7], linestyle = '-',  linewidth = lw)

        ##================ 8bit, 0.15 error =============================
        lb = "8 bit, "+ r"$P_\mathrm{bit\_flip} = 0.15$"
        quant8bit15err    = torch.load(os.path.join(self.rootdir, "2023-08-30-15:22:06_FedAvg/TraRecorder.pt"))  ## P_error = 0.15
        Y = quant8bit15err[:, 2]
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant8bit15err[:, 0], Y, label = lb, color = color[8], linestyle = '-',  linewidth = lw)

        ##================ 8bit, 0.2 error =============================
        lb = "8 bit, "+ r"$P_\mathrm{bit\_flip} = 0.2$"
        quant8bit20err  = torch.load(os.path.join(self.rootdir, "2023-08-29-21:23:56_FedAvg/TraRecorder.pt"))  ## P_error = 0.2
        Y = quant8bit20err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 1)
        axs.plot(quant8bit20err[:, 0], Y, label = lb, color = 'k', linestyle = '-',  linewidth = lw)
        # lb = "Quant: 4 bit"
        # axs.plot(self.diff_quant4bit[:, 0], self.diff_quant4bit[:, 2], label = lb, color = 'g', linestyle = '--', )

        # lb = "Quant: 2 bit"
        # axs.plot(self.diff_quant2bit[:, 0], self.diff_quant2bit[:, 2], label = lb, color = 'g', linestyle = '-.', )
        ##===========================================================
        # axs.grid()
        # label
        font = {'family':'Times New Roman','style':'normal','size':30 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
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
        out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.eps") )
        out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.pdf") )
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_8bitNonIID_performance.pdf") )
        plt.show()
        plt.close()
        return


    # 画出 JSCC (包括在指定信噪比 tra_test_snr 训练) 在指定测试信噪比 tra_test_snr, 指定压缩率 Rlist下训练完后, 不同攻击强度 epslist,PSNR 随 压缩率 Rlist 的曲线. 每条曲线对应一个 tra_test_snr
    def compare_8bit1(self, model = '2nn', ):  ## E = 10, B = 50

        ##======================= mother ===================================
        lw = 2
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True

        ##======================= son =====================================
        axins = axs.inset_axes((0.7, 0.62, 0.25, 0.2))

        ##================ baseline =========================================
        lb = "Error-free"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        Y1 = savgol_filter(Y1, 25, 3)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = 4)
        axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)

        ##================ 8bit, 0 error =============================
        lb = "8 -bit, "+ r"$\epsilon = 0$"
        data = torch.load(os.path.join(self.rootdir, "2023-09-06-09:28:32_FedAvg/TraRecorder.pt"))   ## P_error = 0.  2023-08-29-14:57:58_FedAvg
        Y2 = data[:, 2]
        Y2 = savgol_filter(Y2, 25, 3)
        Y2 = savgol_filter(Y2, 25, 2)
        axs.plot(data[:, 0], Y2, label = lb, color = color[1], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y2,   color = color[1], linestyle = '-',  linewidth = 2)

        ##================ 8bit, 0.0001 error =============================
        ## 2023-09-06-22:10:48_FedAvg  2023-09-07-21:27:54_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-07-21:27:54_FedAvg/TraRecorder.pt"))   ## P_error = 0.005
        lb = "8-bit, "+ r"$\epsilon = 1e^{-4}$"
        Y3 = data[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y3 = savgol_filter(Y3, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y3 = savgol_filter(Y3, 25, 1)
        axs.plot(data[:, 0], Y3, label = lb, color = color[2], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y3,  color = color[2], linestyle = '-',  linewidth = 2)

        ##================ 8bit, 0.001 error =============================
        ##  2023-09-06-20:45:05_FedAvg   2023-09-07-10:06:47_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-07-10:06:47_FedAvg/TraRecorder.pt"))   ## P_error = 0.005
        lb = "8-bit, "+ r"$\epsilon = 0.001$"
        Y4 = data[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y4 = savgol_filter(Y4, 25, 3)
        Y4 = savgol_filter(Y4, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y4, label = lb, color = color[3], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y4,   color = color[3], linestyle = '-',  linewidth = 2)

        ##================ 8bit, 0.005 error =============================
        ##  2023-09-06-19:24:52_FedAvg   2023-09-07-11:33:29_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-07-11:33:29_FedAvg/TraRecorder.pt"))   ## P_error = 0.005
        lb = "8-bit, "+ r"$\epsilon = 0.005$"
        Y5 = data[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y5 = savgol_filter(Y5, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y5 = savgol_filter(Y5, 25, 1)
        axs.plot(data[:, 0], Y5, label = lb, color = color[4], linestyle = '-',  linewidth = lw)
        # axins.plot(data[:, 0], Y5, label = lb, color = color[4], linestyle = '-',  linewidth = lw)

        ##================ 8bit, 0.01 error =============================
        lb = "8-bit, "+ r"$\epsilon = 0.01$"
        ##  2023-09-06-17:51:22_FedAvg      2023-09-07-13:03:40_FedAvg
        data   = torch.load(os.path.join(self.rootdir, "2023-09-07-13:03:40_FedAvg/TraRecorder.pt"))   ## P_error = 0.01
        Y6 = data[:, 2]
        Y6 = savgol_filter(Y6, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        Y6 = savgol_filter(Y6, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y6, label = lb, color = color[5], linestyle = '-',  linewidth = lw)
        # axins.plot(data[:, 0], Y6, label = lb, color = color[5], linestyle = '-',  linewidth = lw)

        ##================ 8bit, 0.05 error =============================
        lb = "8-bit, "+ r"$\epsilon = 0.05$"
        ##  2023-09-06-10:49:16_FedAvg   2023-09-06-12:08:18_FedAvg
        data  = torch.load(os.path.join(self.rootdir, "2023-09-06-10:49:16_FedAvg/TraRecorder.pt"))   ## P_error = 0.05
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        # axs.plot(quant8bit5err[:, 0], Y, label = lb, color = color[6], linestyle = '-',  linewidth = lw)

        data  = torch.load(os.path.join(self.rootdir, "2023-09-06-12:08:18_FedAvg/TraRecorder.pt"))   ## P_error = 0.05
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[6], linestyle = '-',  linewidth = lw)
        # axins.plot(data[:, 0], Y, label = lb, color = color[6], linestyle = '-',  linewidth = lw)

        ##================ 8bit, 0.1 error =============================
        lb = "8-bit, "+ r"$\epsilon = 0.1$"
        ##  2023-09-06-16:16:35_FedAvg   2023-09-07-19:36:10_FedAvg
        data  = torch.load(os.path.join(self.rootdir, "2023-09-07-19:36:10_FedAvg/TraRecorder.pt"))  ## P_error = 0.1
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[7], linestyle = '-',  linewidth = lw)
        # axins.plot(data[:, 0], Y, label = lb, color = color[7], linestyle = '-',  linewidth = lw)

        ##================ 8bit, 0.15 error =============================
        lb = "8-bit, "+ r"$\epsilon = 0.15$"
        ##  2023-09-06-14:54:27_FedAvg  2023-09-07-17:44:15_FedAvg
        data    = torch.load(os.path.join(self.rootdir, "2023-09-07-17:44:15_FedAvg/TraRecorder.pt"))  ## P_error = 0.15
        Y = data[:, 2]
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[8], linestyle = '-',  linewidth = lw)
        # axins.plot(data[:, 0], Y, label = lb, color = color[7], linestyle = '-',  linewidth = lw)

        ##================ 8bit, 0.2 error =============================
        lb = "8-bit, "+ r"$\epsilon = 0.2$"
        ##  2023-09-06-13:26:03_FedAvg   2023-09-07-14:38:05_FedAvg
        data  = torch.load(os.path.join(self.rootdir, "2023-09-07-14:38:05_FedAvg/TraRecorder.pt"))  ## P_error = 0.2
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[0], linestyle = '-',  linewidth = lw)
        # axins.plot(data[:, 0], Y, label = lb, color = 'k', linestyle = '-',  linewidth = lw)

        ##===========================================================
        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        legend1 = axs.legend(loc='lower right', borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.4) ## loc = 'lower left',
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
        zone_and_linked(axs, axins, 800, 850, data[:, 0] , [Y1, Y2, Y3, Y4,], 'bottom',x_ratio = 0.3, y_ratio = 0.7)
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
        out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.eps") )
        # out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.pdf") )
        out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_8bitNonIID_performance.pdf") )
        plt.show()
        plt.close()
        return


    # 画出 JSCC (包括在指定信噪比 tra_test_snr 训练) 在指定测试信噪比 tra_test_snr, 指定压缩率 Rlist下训练完后, 不同攻击强度 epslist,PSNR 随 压缩率 Rlist 的曲线. 每条曲线对应一个 tra_test_snr
    def compare_4bit(self, model = '2nn', ):
        lw = 2
        width = 10
        high  = 8
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True

        ##================ baseline =========================================
        lb = "baseline, without quantiz"
        base_noniid_blance     = torch.load(os.path.join(self.rootdir, "2023-08-29-14:23:04_FedAvg/TraRecorder.pt"))
        Y = base_noniid_blance[:, 2]
        Y = savgol_filter(Y, 25, 3)
        axs.plot(base_noniid_blance[:, 0], Y, label = lb, color = color[0], linestyle = '-', linewidth = 8)


        # ##================ 4bit, 0 error =============================
        # lb = "4 bit, "+ r"$P_\mathrm{bit\_flip} = 0$"
        # quant8bit0err = torch.load(os.path.join(self.rootdir, "2023-08-30-20:45:16_FedAvg/TraRecorder.pt"))   ## P_error = 0.
        # Y = quant8bit0err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 2)
        # axs.plot(quant8bit0err[:, 0], Y, label = lb, color = color[0], linestyle = '-',  linewidth = 4)
        ##================ 4bit, 0 error =============================
        lb = "4 bit, "+ r"$P_\mathrm{bit\_flip} = 0$"
        quant4bit0err = torch.load(os.path.join(self.rootdir, "2023-08-30-21:35:09_FedAvg/TraRecorder.pt"))   ## P_error = 0.
        Y = quant4bit0err[:, 2]
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant4bit0err[:, 0], Y, label = lb, color = color[1], linestyle = '-',  linewidth = 4)



        ##================ 4bit, 0.0001 error =============================
        quant4bit001err = torch.load(os.path.join(self.rootdir, "2023-08-31-09:41:36_FedAvg/TraRecorder.pt"))   ## P_error = 0.005
        lb = "4 bit, "+ r"$P_\mathrm{bit\_flip} = 0.0001$"
        Y = quant4bit001err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y = savgol_filter(Y, 25, 1)
        axs.plot(quant4bit001err[:, 0], Y, label = lb, color = color[2], linestyle = '-',  linewidth = lw)

        ##================ 4bit, 0.001 error =============================
        quant4bit01err = torch.load(os.path.join(self.rootdir, "2023-08-31-00:17:47_FedAvg/TraRecorder.pt"))   ## P_error = 0.005
        lb = "4 bit, "+ r"$P_\mathrm{bit\_flip} = 0.001$"
        Y = quant4bit01err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant4bit01err[:, 0], Y, label = lb, color = color[3], linestyle = '-',  linewidth = lw)


        ##================ 4bit, 0.005 error =============================
        quant4bit05err = torch.load(os.path.join(self.rootdir, "2023-08-31-01:20:35_FedAvg/TraRecorder.pt"))   ## P_error = 0.005
        lb = "4 bit, "+ r"$P_\mathrm{bit\_flip} = 0.005$"
        Y = quant4bit05err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y = savgol_filter(Y, 25, 1)
        axs.plot(quant4bit05err[:, 0], Y, label = lb, color = color[4], linestyle = '-',  linewidth = lw)

        ##================ 8bit, 0.01 error =============================
        lb = "4 bit, "+ r"$P_\mathrm{bit\_flip} = 0.01$"
        quant4bit1err   = torch.load(os.path.join(self.rootdir, "2023-08-31-02:45:38_FedAvg/TraRecorder.pt"))   ## P_error = 0.01
        Y = quant4bit1err[:, 2]
        # Y = savgol_filter(Y, 25, 10)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 1)
        axs.plot(quant4bit1err[:, 0], Y, label = lb, color = color[5], linestyle = '-',  linewidth = lw)

        ##================ 8bit, 0.05 error =============================
        lb = "4 bit, "+ r"$P_\mathrm{bit\_flip} = 0.05$"
        quant4bit5err  = torch.load(os.path.join(self.rootdir, "2023-08-31-08:35:34_FedAvg/TraRecorder.pt"))   ## P_error = 0.05
        Y = quant4bit5err[:, 2]
        Y = savgol_filter(Y, 28, 3)
        # Y = savgol_filter(Y, 28, 3)
        # Y = savgol_filter(Y, 28, 1)
        # Y = savgol_filter(Y, 28, 1)
        Y = savgol_filter(Y, 28, 1)
        axs.plot(quant4bit5err[:, 0], Y, label = lb, color = color[6], linestyle = '-',  linewidth = lw)

        ##================ 8bit, 0.1 error =============================
        lb = "4 bit, "+ r"$P_\mathrm{bit\_flip} = 0.1$"
        quant4bit10err  = torch.load(os.path.join(self.rootdir, "2023-08-30-23:10:43_FedAvg/TraRecorder.pt"))  ## P_error = 0.1
        Y = quant4bit10err[:, 2]
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant4bit10err[:, 0], Y, label = lb, color = color[7], linestyle = '-',  linewidth = lw)

        # ##================ 8bit, 0.15 error =============================
        # lb = "4 bit, "+ r"$P_\mathrm{bit\_flip} = 0.15$"
        # quant8bit15err    = torch.load(os.path.join(self.rootdir, "2023-08-30-15:22:06_FedAvg/TraRecorder.pt"))  ## P_error = 0.15
        # Y = quant8bit15err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        # # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # # Y = savgol_filter(Y, 25, 1)
        # # Y = savgol_filter(Y, 25, 1)
        # axs.plot(quant8bit15err[:, 0], Y, label = lb, color = color[8], linestyle = '-',  linewidth = lw)

        # ##================ 8bit, 0.2 error =============================
        # lb = "4 bit, "+ r"$P_\mathrm{bit\_flip} = 0.2$"
        # quant8bit20err  = torch.load(os.path.join(self.rootdir, "2023-08-29-21:23:56_FedAvg/TraRecorder.pt"))  ## P_error = 0.2
        # Y = quant8bit20err[:, 2]
        # # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # axs.plot(quant8bit20err[:, 0], Y, label = lb, color = 'k', linestyle = '-',  linewidth = lw)

        ##===========================================================
        # axs.grid()
        # label
        font = {'family':'Times New Roman','style':'normal','size':30 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
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
        out_fig.savefig(os.path.join(savepath, f"{model}_4bitNonIID_performance.eps") )
        out_fig.savefig(os.path.join(savepath, f"{model}_4bitNonIID_performance.pdf") )
        out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_4bitNonIID_performance.pdf") )
        plt.show()
        plt.close()
        return

    # 画出 JSCC (包括在指定信噪比 tra_test_snr 训练) 在指定测试信噪比 tra_test_snr, 指定压缩率 Rlist下训练完后, 不同攻击强度 epslist,PSNR 随 压缩率 Rlist 的曲线. 每条曲线对应一个 tra_test_snr
    def compare_6bit(self, model = '2nn', ):
        lw = 2
        width = 10
        high  = 8
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=Trues

        #================ baseline =========================================
        lb = "baseline, without quantiz"
        base_noniid_blance     = torch.load(os.path.join(self.rootdir, "2023-08-29-14:23:04_FedAvg/TraRecorder.pt"))
        Y = base_noniid_blance[:, 2]
        Y = savgol_filter(Y, 25, 3)
        axs.plot(base_noniid_blance[:, 0], Y, label = lb, color = color[0], linestyle = '-', linewidth = 6)


        ##================ 6 bit, 0 error =============================
        lb = "6 bit, "+ r"$P_\mathrm{bit\_flip} = 0$"
        quant6bit0err = torch.load(os.path.join(self.rootdir, "2023-08-31-22:39:07_FedAvg/TraRecorder.pt"))   ## P_error = 0.
        Y = quant6bit0err[:, 2]
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant6bit0err[:, 0], Y, label = lb, color = color[1], linestyle = '-',  linewidth = lw)
        #================ 6 bit, 0.0001 error =============================
        quant6bit001err = torch.load(os.path.join(self.rootdir, "2023-09-01-13:28:25_FedAvg/TraRecorder.pt"))   ## P_error = 0.005
        lb = "6 bit, "+ r"$P_\mathrm{bit\_flip} = 0.0001$"
        Y = quant6bit001err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y = savgol_filter(Y, 25, 1)
        axs.plot(quant6bit001err[:, 0], Y, label = lb, color = color[2], linestyle = '-',  linewidth = lw)

        #================ 8bit, 0.001 error =============================
        quant6bit01err = torch.load(os.path.join(self.rootdir, "2023-09-01-12:10:37_FedAvg/TraRecorder.pt"))   ## P_error = 0.005
        lb = "6 bit, "+ r"$P_\mathrm{bit\_flip} = 0.001$"
        Y = quant6bit01err[:, 2]
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant6bit01err[:, 0], Y, label = lb, color = color[3], linestyle = '-',  linewidth = lw)


        #================ 8bit, 0.005 error =============================
        quant4bit05err = torch.load(os.path.join(self.rootdir, "2023-09-01-10:59:57_FedAvg/TraRecorder.pt"))   ## P_error = 0.005
        lb = "6 bit, "+ r"$P_\mathrm{bit\_flip} = 0.005$"
        Y = quant4bit05err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant4bit05err[:, 0], Y, label = lb, color = color[4], linestyle = '-',  linewidth = lw)

        ##================ 4 bit, 0.01 error =============================
        lb = "6 bit, "+ r"$P_\mathrm{bit\_flip} = 0.01$"
        quant4bit1err   = torch.load(os.path.join(self.rootdir, "2023-09-01-09:47:14_FedAvg/TraRecorder.pt"))   ## P_error = 0.01
        Y = quant4bit1err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant4bit1err[:, 0], Y, label = lb, color = color[5], linestyle = '-',  linewidth = lw)

        ##================ 6 bit, 0.05 error =============================
        lb = "6 bit, "+ r"$P_\mathrm{bit\_flip} = 0.05$"
        quant6bit5err  = torch.load(os.path.join(self.rootdir, "2023-09-01-01:29:16_FedAvg/TraRecorder.pt"))   ## P_error = 0.05
        Y = quant6bit5err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant6bit5err[:, 0], Y, label = lb, color = color[6], linestyle = '-',  linewidth = lw)

        ##================ 6 bit, 0.1 error =============================
        lb = "6 bit, "+ r"$P_\mathrm{bit\_flip} = 0.1$"
        quant6bit10err  = torch.load(os.path.join(self.rootdir, "2023-08-31-23:53:48_FedAvg/TraRecorder.pt"))  ## P_error = 0.1
        Y = quant6bit10err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        Y = savgol_filter(Y, 25, 1)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(quant6bit10err[:, 0], Y, label = lb, color = color[7], linestyle = '-',  linewidth = lw)

        # ##================ 8bit, 0.15 error =============================
        # lb = "4 bit, "+ r"$P_\mathrm{bit\_flip} = 0.15$"
        # quant8bit15err    = torch.load(os.path.join(self.rootdir, "2023-08-30-15:22:06_FedAvg/TraRecorder.pt"))  ## P_error = 0.15
        # Y = quant8bit15err[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        # # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # # Y = savgol_filter(Y, 25, 1)
        # # Y = savgol_filter(Y, 25, 1)
        # axs.plot(quant8bit15err[:, 0], Y, label = lb, color = color[8], linestyle = '-',  linewidth = lw)

        # ##================ 8bit, 0.2 error =============================
        # lb = "4 bit, "+ r"$P_\mathrm{bit\_flip} = 0.2$"
        # quant8bit20err  = torch.load(os.path.join(self.rootdir, "2023-08-29-21:23:56_FedAvg/TraRecorder.pt"))  ## P_error = 0.2
        # Y = quant8bit20err[:, 2]
        # # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # axs.plot(quant8bit20err[:, 0], Y, label = lb, color = 'k', linestyle = '-',  linewidth = lw)

        ##===========================================================
        # axs.grid()
        # label
        font = {'family':'Times New Roman','style':'normal','size':30 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
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
        out_fig.savefig(os.path.join(savepath, f"{model}_6bitNonIID_performance.eps") )
        out_fig.savefig(os.path.join(savepath, f"{model}_6bitNonIID_performance.pdf") )
        out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_6bitNonIID_performance.pdf") )
        plt.show()
        plt.close()
        return

    def compare_dynamicQuantize(self, model = '2nn', ):
        lw = 2
        width = 10
        high  = 8.5
        color = ['#1E90FF','#FF6347','#0000FF','#00FF00','#808000','#FF00FF','#4ea142','#FFA500','#800080','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE',]

        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        axins = axs.inset_axes((0.62, 0.56, 0.3, 0.25))
        ##================ baseline =========================================
        lb = "Error-free"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        Y1 = savgol_filter(Y1, 25, 3)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = 3)
        axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)
        i += 1

        ##================ Dynamic, 0 error =============================
        lb = "DQ, "+ r"$\epsilon = 0$"
        ##  2023-09-10-17:05:03_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-10-17:05:03_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        Y2 = savgol_filter(Y2, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        axins.plot(data[:, 0], Y2, color = color[i], linestyle = '--', linewidth = 2)
        i += 1

        ##================ 4 bit,  0 error =============================
        # lb = "4 bit, "+ r"$\epsilon = 0$"
        # ##  2023-09-14-16:46:59_FedAvg  2023-09-14-15:01:21_FedAvg
        # data = torch.load(os.path.join(self.rootdir, "2023-09-14-16:46:59_FedAvg/TraRecorder.pt"))
        # Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        # axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        # i += 1

        lb = "4-bit, "+ r"$\epsilon = 0$"
        ##  2023-09-14-15:01:21_FedAvg    2023-09-17-12:24:57_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-14-15:01:21_FedAvg/TraRecorder.pt"))
        Y3 = data[:, 2]
        Y3 = savgol_filter(Y3, 25, 3)
        Y3 = savgol_filter(Y3, 25, 1)
        axs.plot(data[:, 0], Y3, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        axins.plot(data[:, 0], Y3, color = color[i], linestyle = '--', linewidth = 2)
        i += 1

        ##================ 4 bit,  0.01 error =============================
        lb = "4-bit, "+ r"$\epsilon = 0.01$"
        ##  2023-09-14-15:01:21_FedAvg    2023-09-17-12:24:57_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-17-17:32:28_FedAvg/TraRecorder.pt"))
        Y4 = data[:, 2]
        Y4 = savgol_filter(Y4, 25, 3)
        Y4 = savgol_filter(Y4, 25, 1)
        # axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        # i += 1


        ##================ 1 bit,  0 error =============================
        lb = "1-bit, "+ r"$\epsilon = 0$"
        ##  2023-09-14-21:37:36_FedAvg     2023-09-14-22:17:04_FedAvg    2023-09-17-18:15:00_FedAvg  2023-09-17-19:55:44_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-17-20:32:40_FedAvg/TraRecorder.pt"))
        Y5 = data[:, 2]
        Y5 = savgol_filter(Y5, 25, 10)
        Y5 = savgol_filter(Y5, 25, 3)
        Y5 = savgol_filter(Y5, 25, 3)
        Y5 = savgol_filter(Y5, 25, 1)
        Y5 = savgol_filter(Y5, 25, 1)
        axs.plot(data[:, 0], Y5, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.01 error =============================
        lb = "1-bit, "+ r"$\epsilon = 0.01$"
        ##  2023-09-14-21:37:36_FedAvg     2023-09-14-22:17:04_FedAvg    2023-09-17-18:15:00_FedAvg  2023-09-17-19:55:44_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-17-22:06:23_FedAvg/TraRecorder.pt"))
        Y6 = data[:, 2]
        # Y5 = savgol_filter(Y5, 25, 10)
        Y6 = savgol_filter(Y6, 25, 3)
        Y6 = savgol_filter(Y6, 25, 3)
        Y6 = savgol_filter(Y6, 25, 1)
        Y6 = savgol_filter(Y6, 25, 1)
        axs.plot(data[:, 0], Y6, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        # axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1

        ##================ 1 bit,  0.1 error =============================
        lb = "1-bit, "+ r"$\epsilon = 0.1$"
        ##  2023-09-18-09:09:33_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-18-09:09:33_FedAvg/TraRecorder.pt"))
        Y6 = data[:, 2]
        # Y5 = savgol_filter(Y5, 25, 10)
        Y6 = savgol_filter(Y6, 25, 3)
        Y6 = savgol_filter(Y6, 25, 3)
        Y6 = savgol_filter(Y6, 25, 1)
        Y6 = savgol_filter(Y6, 25, 1)
        axs.plot(data[:, 0], Y6, label = lb, color = color[i], linestyle = '--',  linewidth = 2)
        # axins.plot(data[:, 0], Y5, color = color[i], linestyle = '--', linewidth = 2)
        i += 1

        ##================ Dynamic, 0.001 error =============================
        lb = "DQ, "+ r"$\epsilon = 0.001$"
        ##   2023-09-10-20:25:06_FedAvg  2023-09-11-09:12:45_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-11-09:12:45_FedAvg/TraRecorder.pt"))
        Y7 = data[:, 2]
        Y7 = savgol_filter(Y7, 25, 3)
        # Y = savgol_filter(Y, 25, 2)
        axs.plot(data[:, 0], Y7, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        axins.plot(data[:, 0], Y7, color = color[i], linestyle = '-', linewidth = 2)
        i += 1
        ##================ Dynamic, 0.005 error =============================
        lb = "DQ, "+ r"$\epsilon = 0.005$"
        ##  2023-09-11-00:56:52_FedAvg   2023-09-11-10:54:57_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-11-10:54:57_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 2)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        ##================ Dynamic, 0.01 error =============================
        lb = "DQ, "+ r"$\epsilon = 0.01$"
        ## 2023-09-10-21:47:49_FedAvg  2023-09-11-12:06:56_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-11-12:06:56_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        ##================ Dynamic, 0.05 error =============================
        lb = "DQ, "+ r"$\epsilon = 0.05$"
        ## 2023-09-10-23:23:23_FedAvg   2023-09-11-14:45:54_FedAvg    2023-09-11-20:50:39_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-11-20:50:39_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 1)
        axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        i += 1
        ##================ Dynamic, 0.1 error =============================
        lb = "DQ, "+ r"$\epsilon = 0.1$"
        ## 2023-09-02-02:05:04_FedAvg  2023-09-10-17:05:03_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-10-21:47:49_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        Y = savgol_filter(Y, 25, 3)
        # Y = savgol_filter(Y, 25, 2)
        #axs.plot(data[:, 0], Y, label = lb, color = color[i], linestyle = '-',  linewidth = 1)
        i += 1

        ##===========================================================
        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        ## legend
        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
        font1 = {'family':'Times New Roman','style':'normal','size':25, }
        legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.1, 0.01, 0.2, 0.5),  borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.4)
        # axs.legend(loc = 'lower left', borderaxespad = 0, edgecolor = 'black', prop = font1,)
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        ## linewidth
        bw = 2.5
        axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        ## xtick
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
        ### 局部显示并且进行连线,方法3
        zone_and_linked(axs, axins, 800, 850, data[:, 0] , [Y1, Y2, Y3, Y4, Y5, Y6], 'bottom',x_ratio = 0.3, y_ratio = 0.4)
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

        ##=======================  mother =================================
        fontt = {'family':'Times New Roman','style':'normal','size':30}
        plt.suptitle("non-IID MNIST, 2NN", fontproperties = fontt, )

        out_fig = plt.gcf()
        savepath = self.savedir
        out_fig.savefig(os.path.join(savepath, f"{model}_DynamicbitNonIID_performance.eps") )
        # out_fig.savefig(os.path.join(savepath, f"{model}_DynamicbitNonIID_performance.pdf") )
        out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_DynamicbitNonIID_performance.pdf") )
        plt.show()
        plt.close()
        return


# def main():
pl = ResultPlot( ) # 1: '2022-10-12-17:38:12'  2:'2022-10-14-09:56:05'


model = "2nn"


pl.compare_8bit1(model = model)

pl.compare_dynamicQuantize(model = model)

# pl.compare_4bit(model = model)

# pl.compare_6bit(model = model)



















