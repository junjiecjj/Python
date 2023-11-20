
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

mark  = ['s','v','d','o', '*',  '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', '$\\clubsuit$', '$\\bowtie$', '$\\boxdot$', '$\\boxtimes$', '$\\boxminus$',  '$\\boxplus$', '$\\bigodot$','$\\bigoplus$', '$\\bigotimes$', '$\\bigstar$', '$\\blacksquare$', '$\circledR$', '$\circledS$',  '$\\bigtriangledown$','$\\bigtriangleup$','$\\blacktriangleleft$', '$\\blacktriangleright$', '$\\diamondsuit',   '$\heartsuit$', '$666$', '$\\varrho$', '$\\Omega$', '$\\nabla$', '$\\Upsilon$', '$\\mho$', '$\\Phi$',  '$\\divideontimes$', '$\\otimes$', '$\\odot$', '$\\ominus$', '$\\oplus$', '$\\oslash$', '$\\otimes$']

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


    # 画出 JSCC (包括在指定信噪比 tra_test_snr 训练) 在指定测试信噪比 tra_test_snr, 指定压缩率 Rlist下训练完后, 不同攻击强度 epslist,PSNR 随 压缩率 Rlist 的曲线. 每条曲线对应一个 tra_test_snr
    def compare_dynamic_bit(self, model = '2nn', ):  ## E = 10, B = 50

        ##======================= mother ===================================
        lw = 2
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        ##======================= son =====================================
        axins = axs.inset_axes((0.7, 0.62, 0.25, 0.2))
        # ax2 = axs.twinx()
        ##================ baseline =========================================
        lb = "Error-free"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        Y1 = savgol_filter(Y1, 25, 3)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = 2)
        # axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)
        lb = "overhead, Error-free"
        # l3 = ax2.plot(data[:, 0], data[:, 0]*199210*8, color = 'k', label = lb, linestyle = '--', linewidth = 2,)

        ##================ dynamic bit, 0 error =============================
        lb = "DQ, nr, "+ r"$\epsilon = 0$"
        ###  2023-09-10-17:05:03_FedAvg   2023-09-23-21:33:44_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-23-21:33:44_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 2)
        axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y2,   color = color[1], linestyle = '-',  linewidth = 2)
        lb = "overhead, DQ"
        # l3 = ax2.plot(data[:, 0], data[:, 5], color = color[i], label = lb, linestyle = '--', linewidth = 2,)
        i += 1

        ##================ 4bit, 0 error =============================
        lb = "4-bit, nr, 0.002, "+ r"$\epsilon = 0$"
        ###  2023-08-30-21:35:09_FedAvg  2023-09-23-20:00:11_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-23-20:00:11_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 1)
        axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y2,   color = color[1], linestyle = '-',  linewidth = 2)
        lb = "overhead, 4-bit"
        # l3 = ax2.plot(data[:, 0], data[:, 5], color = color[i], label = lb, linestyle = '--', linewidth = 2,)
        i += 1

        ##================ 1 bit, 0 error =============================
        lb = "1-bit, "+ r"$\epsilon = 0$"
        ##  2023-09-18-22:15:43_FedAvg   2023-09-19-16:21:30_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-18-22:15:43_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        # Y2 = savgol_filter(Y2, 25, 3)
        Y2 = savgol_filter(Y2, 25, 3)
        Y2 = savgol_filter(Y2, 25, 1)
        # Y2 = savgol_filter(Y2, 25, 1)
        axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y2,   color = color[1], linestyle = '-',  linewidth = 2)
        lb = "overhead, 1-bit"
        # l3 = ax2.plot(data[:, 0], data[:, 5], color = color[i], label = lb, linestyle = '--', linewidth = 2,)
        i += 1


        ##======================= mother  ====================================
        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':20, }
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.6, 0.1, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
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
        # zone_and_linked(axs, axins, 800, 850, data[:, 0] , [Y1, Y2, Y3, Y4,], 'bottom',x_ratio = 0.3, y_ratio = 0.7)
        ## linewidth
        bw = 2
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
        # out_fig.savefig(os.path.join(savepath, f"{model}_DynamicBitNonIID.eps") )
        # out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.pdf") )
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_8bitNonIID_performance.pdf") )
        # plt.show()
        # plt.close()
        return



    # 画出 JSCC (包括在指定信噪比 tra_test_snr 训练) 在指定测试信噪比 tra_test_snr, 指定压缩率 Rlist下训练完后, 不同攻击强度 epslist,PSNR 随 压缩率 Rlist 的曲线. 每条曲线对应一个 tra_test_snr
    def compare_dynamic_bit_multi(self, model = '2nn', ):  ## E = 10, B = 50
        color = ['#FF0000','#0000FF','#008000', '#FF00FF','#FFA500','#800080','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE','#808000']
        ##======================= mother ===================================
        lw = 3
        # fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        fig = plt.figure(figsize=(16, 12), constrained_layout=True )
        i = 0

        ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1)
        ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
        ax2.get_xaxis().get_major_formatter().set_scientific(False)
        ax3 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
        ##==============================================================================
        ##================================  ax1 ========================================
        axins = ax1.inset_axes((0.4, 0.4, 0.5, 0.4))
        # ax2 = axs.twinx()
        ##================ baseline =========================================
        lb = "Error-free FL"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        Y1 = savgol_filter(Y1, 25, 3)
        ax1.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = lw,  marker='o', markersize = 12, markevery=100)
        axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = lw,  marker='o', markersize = 10, markevery=40)
        # lb = "8-bit"
        # ax2.plot(data[:, 0], data[:, 0]*199210*8, color = 'k', label = lb, linestyle = '--', linewidth = lw,)
        # ax3.plot(data[:, 0], [8]*data[:, 0].shape[0], color = 'k', label = lb, linestyle = '--', linewidth = 2,)

        ##================ 8 bit, 0 error =============================
        lb = "8-bit FL"

        data = torch.load(os.path.join(self.rootdir, "2023-09-06-09:28:32_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        Y2 = savgol_filter(Y2, 25, 3)
        # Y2 = savgol_filter(Y2, 25, 2)
        ax1.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = lw)
        axins.plot(data[:, 0], Y2, color = color[i], linestyle = '-',  linewidth = lw, )
        lb = "8-bit FL"
        ax2.plot(data[:, 0], data[:, 0]*199210*8, color = color[i], label = lb, linestyle = '-', linewidth = lw,  marker = mark[i], markersize = 10, markevery=100)
        i += 1

        ##================ dynamic bit, 0 error =============================
        lb = "DQ FL"
        ###  2023-09-10-17:05:03_FedAvg   2023-09-23-21:33:44_FedAvg  2023-09-24-12:11:35_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-10-23-22:26:13_FedAvg/TraRecorder.pt"))
        Y3 = data[:, 2]
        Y3 = savgol_filter(Y3, 25, 3)
        # Y3 = savgol_filter(Y3, 25, 1)
        ax1.plot(data[:, 0], Y3, label = lb, color = color[i], linestyle = '-',  linewidth = lw)
        axins.plot(data[:, 0], Y3, color = color[i], linestyle = '-',  linewidth = lw)
        lb = "DQ FL"
        data = torch.load(os.path.join(self.rootdir, "2023-09-23-21:33:44_FedAvg/TraRecorder.pt"))
        ax2.plot(data[:, 0], data[:, 5], color = color[i], label = lb, linestyle = '-', linewidth = lw,  marker = mark[i], markersize = 10, markevery=100)
        data = torch.load(os.path.join(self.rootdir, "2023-09-24-12:11:35_FedAvg/TraRecorder.pt"))
        ax3.plot(data[:, 0], data[:, 6], color = color[i], label = lb, linestyle = '-', linewidth = 1,)
        i += 1

        ##================ 4bit, 0 error =============================
        lb = "4-bit FL"
        ###  2023-08-30-21:35:09_FedAvg  2023-09-23-20:00:11_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-09-23-20:00:11_FedAvg/TraRecorder.pt"))
        Y4 = data[:, 2]
        Y4 = savgol_filter(Y4, 25, 4)
        # Y4 = savgol_filter(Y4, 25, 1)
        ax1.plot(data[:, 0], Y4, label = lb, color = color[i], linestyle = '-',  linewidth = lw)
        axins.plot(data[:, 0], Y4, color = color[i], linestyle = '-',  linewidth = lw)
        lb = "4-bit FL"
        ax2.plot(data[:, 0], data[:, 5], color = color[i], label = lb, linestyle = '-', linewidth = lw, marker = mark[i], markersize = 10, markevery=100)
        # ax3.plot(data[:, 0], [4]*data[:, 0].shape[0], color = color[i], label = lb, linestyle = '--', linewidth = 2,)
        i += 1

        ##================ 1 bit, 0 error =============================
        lb = "1-bit FL"
        ##  2023-09-18-22:15:43_FedAvg   2023-09-19-16:21:30_FedAvg
        data = torch.load(os.path.join(self.rootdir, "2023-10-21-15:54:09_FedAvg/TraRecorder.pt"))
        Y5 = data[:, 2] #- 0.008
        # Y5[-50:] = Y5[-50:].mean()
        Y5 = savgol_filter(Y5, 25, 3)
        # Y5 = savgol_filter(Y5, 25, 3)
        # Y5 = savgol_filter(Y5, 25, 2)
        Y5 = savgol_filter(Y5, 25, 1)
        # Y5 = savgol_filter(Y5, 25, 1)
        ax1.plot(data[:, 0], Y5, label = lb, color = color[i], linestyle = '-',  linewidth = lw)
        axins.plot(data[:, 0], Y5, color = color[i], linestyle = '-',  linewidth = lw)
        lb = "1-bit FL"
        ax2.plot(data[:, 0], data[:, 5], color = color[i], label = lb, linestyle = '-', linewidth = lw, marker = mark[i], markersize = 10, markevery=100)
        # ax3.plot(data[:, 0], [1]*data[:, 0].shape[0], color = color[i], label = lb, linestyle = '--', linewidth = 2,)
        i += 1

        ##======================= mother  ====================================
        x_major_locator=MultipleLocator(250)
        ax1.xaxis.set_major_locator(x_major_locator)
        ax1.set_ylim(0.3, 1.)  #拉开坐标轴范围显示投影

        ## xlabel
        ax1.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        ax1.set_xlabel("Round", fontproperties=font)
        ax1.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1
        font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 35)
        ax1.set_title('(a)',  fontproperties=font1)
        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':32, }
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        legend1 = ax1.legend(loc='best',   borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        # frame1.set_facecolor('none')  # 设置图例legend背景透明

        ## lindwidth
        bw = 2.5
        ax1.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        ax1.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        ax1.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        ax1.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        ## xtick
        ax1.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 32, width = 3 )
        labels = ax1.get_xticklabels() + ax1.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # [label.set_fontsize(35) for label in labels] #刻度值字号

        ##==================== mother and son ==================================
        ### 局部显示并且进行连线,方法3
        zone_and_linked(ax1, axins, 800, 890, data[:, 0] , [Y1, Y2, Y3, Y4, Y5], 'bottom',x_ratio = 0.3, y_ratio = 0.3)
        ## linewidth
        bw = 2
        axins.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        axins.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        axins.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        axins.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        axins.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 32, width = 1)
        labels = axins.get_xticklabels() + axins.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # [label.set_fontsize(16) for label in labels] #刻度值字号


        ##==============================================================================
        ##================================  ax2 ========================================
        x_major_locator=MultipleLocator(250)
        ax2.xaxis.set_major_locator(x_major_locator)
        ## xlabel
        ax2.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        # ax2.set_xlabel("Round", fontproperties=font)
        # ax2.set_ylabel( "Overhead (" + r"$\mathrm{1}e^9$" + " bits)", fontproperties = font )# , fontdict = font1
        ax2.set_ylabel( "Overhead (bits)", fontproperties = font )# , fontdict = font1
        font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 35)
        ax2.set_title('(b)',  fontproperties=font1)
        # ax2.set_yscale('log')#设置纵坐标的缩放

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':32, }
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        legend1 = ax2.legend(loc='best', borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        # frame1.set_facecolor('none')  # 设置图例legend背景透明

        ## lindwidth
        bw = 2.5
        ax2.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        ax2.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        ax2.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        ax2.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        ## xtick
        ax2.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 32, width = 3 )
        labels = ax2.get_xticklabels() + ax2.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # [label.set_fontsize(35) for label in labels] #刻度值字号

        ##==============================================================================
        ##================================  ax3 ========================================
        ax3.set_ylim(0, 8.5)  #拉开坐标轴范围显示投影
        x_major_locator=MultipleLocator(250)
        ax3.xaxis.set_major_locator(x_major_locator)
        # y_major_locator=MultipleLocator(2)
        # ax3.yaxis.set_major_locator(y_major_locator)
        ## xlabel
        ax3.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        ax3.set_xlabel("Round", fontproperties=font)
        ax3.set_ylabel( "Bit-width",      fontproperties = font )# , fontdict = font1
        font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 35)
        ax3.set_title('(c)',  fontproperties=font1)
        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':32, }
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        legend1 = ax3.legend(loc='best',  borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        # frame1.set_facecolor('none')  # 设置图例legend背景透明

        ## lindwidth
        bw = 2.5
        ax3.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        ax3.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        ax3.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        ax3.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        ## xtick
        ax3.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 32, width = 3 )
        labels = ax3.get_xticklabels() + ax3.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # [label.set_fontsize(35) for label in labels] #刻度值字号


        ##===================== plt =========================================
        fontt = {'family':'Times New Roman','style':'normal','size':30}
        plt.suptitle("non-IID MNIST, 2NN", fontproperties = fontt, )
        out_fig = plt.gcf()
        # savepath = self.savedir
        out_fig.savefig(f"{model}_DynamicBitNonIID.eps")
        # out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.pdf") )
        out_fig.savefig(f"{model}_DynamicBitNonIID_1.pdf")
        # plt.show()
        # plt.close()
        return


# def main():
pl = ResultPlot( ) # 1: '2022-10-12-17:38:12'  2:'2022-10-14-09:56:05'


model = "2nn"

## Paper Fig.6
pl.compare_dynamic_bit_multi(model = model)


















