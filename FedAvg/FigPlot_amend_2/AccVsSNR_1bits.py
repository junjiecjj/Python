
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

    # 画出 JSCC (包括在指定信噪比 tra_test_snr 训练) 在指定测试信噪比 tra_test_snr, 指定压缩率 Rlist下训练完后, 不同攻击强度 epslist,PSNR 随 压缩率 Rlist 的曲线. 每条曲线对应一个 tra_test_snr
    def PerformanceSNR_with1bit(self, model = '2nn', ):  ## E = 10, B = 50

        ##======================= mother ===================================
        lw = 2
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        ##======================= son =====================================
        # axins = axs.inset_axes((0.5, 0.4, 0.45, 0.4))

        ##================ baseline =========================================
        lb = "Error-free"
        data     = torch.load(os.path.join(self.rootdir, "2023-09-05-22:14:00_FedAvg/TraRecorder.pt"))
        Y1 = data[:, 2]
        Y1 = savgol_filter(Y1, 25, 3)
        axs.plot(data[:, 0], Y1, label = lb, color = 'k', linestyle = '-', linewidth = 2)
        # axins.plot(data[:, 0], Y1, color = 'k', linestyle = '-', linewidth = 2)


        # ##================ 1 bit  bit, 2 dB =============================
        lb = "1-bit, SNR = 2(dB)"
        data = torch.load(os.path.join(self.rootdir, "2023-10-22-22:10:41_FedAvg/TraRecorder.pt"))
        Y2 = data[:, 2]
        Y2 = savgol_filter(Y2, 25, 3)
        Y2 = savgol_filter(Y2, 25, 1)
        axs.plot(data[:, 0], Y2, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y2,   color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        # ##================ 1  bit, 1.75 dB =============================
        # lb = "1-bit, SNR = 1.75(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-28-15:00:43_FedAvg/TraRecorder.pt"))
        # Y3 = data[:, 2]
        # # Y3 = savgol_filter(Y3, 25, 3)
        # # Y3 = savgol_filter(Y3, 25, 2)
        # axs.plot(data[:, 0], Y3, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # # axins.plot(data[:, 0], Y3,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        # ##================ 1  bit, 1.5 dB =============================
        # lb = "1-bit, SNR = 1.5(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-28-12:46:05_FedAvg/TraRecorder.pt"))
        # Y4 = data[:, 2]
        # # Y4 = savgol_filter(Y4, 25, 3)
        # # Y4 = savgol_filter(Y4, 25, 2)
        # axs.plot(data[:, 0], Y4, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # # axins.plot(data[:, 0], Y4,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        # ##================ 1  bit, 1.25 dB =============================
        # lb = "4-bit, SNR = 1.25(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-28-11:08:20_FedAvg/TraRecorder.pt"))
        # Y5 = data[:, 2]
        # Y5 = savgol_filter(Y5, 25, 3)
        # Y5 = savgol_filter(Y5, 25, 2)
        # axs.plot(data[:, 0], Y5, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # # axins.plot(data[:, 0], Y5,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        ##================ 1  bit, 1.0 dB =============================
        lb = "1-bit, SNR = 1(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-10-22-19:58:08_FedAvg/TraRecorder.pt"))
        Y6 = data[:, 2]
        Y6 = savgol_filter(Y6, 25, 3)
        Y6 = savgol_filter(Y6, 25, 1)
        axs.plot(data[:, 0], Y6, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y6,   color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        # ##================ 1  bit, 0.75 dB =============================
        # lb = "1-bit, SNR = 0.75(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-27-19:25:23_FedAvg/TraRecorder.pt"))
        # Y7 = data[:, 2]
        # Y7 = savgol_filter(Y7, 25, 3)
        # Y7 = savgol_filter(Y7, 25, 2)
        # axs.plot(data[:, 0], Y7, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # # axins.plot(data[:, 0], Y7,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        # ##================ 1  bit, 0.5 dB =============================
        # lb = "1-bit, SNR = 0.5(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-27-17:33:56_FedAvg/TraRecorder.pt"))
        # Y8 = data[:, 2]
        # Y8 = savgol_filter(Y8, 25, 3)
        # Y8 = savgol_filter(Y8, 25, 2)
        # axs.plot(data[:, 0], Y8, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # # axins.plot(data[:, 0], Y8,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        # #================ 1 bit, 0.25 dB =============================
        # lb = "1-bit, SNR = 0.25(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-27-15:50:28_FedAvg/TraRecorder.pt"))
        # Y9 = data[:, 2]
        # Y9 = savgol_filter(Y9, 25, 3)
        # Y9 = savgol_filter(Y9, 25, 2)
        # axs.plot(data[:, 0], Y9, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # # axins.plot(data[:, 0], Y9,   color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1

        ##================ 1 bit, 0 dB =============================
        lb = "1-bit, SNR = 0(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-10-22-12:50:01_FedAvg/TraRecorder.pt"))
        Y10 = data[:, 2]
        Y10 = savgol_filter(Y10, 25, 3)
        # Y10 = savgol_filter(Y10, 25, 3)
        Y10 = savgol_filter(Y10, 25, 1)
        # Y10 = savgol_filter(Y10, 25, 1)
        axs.plot(data[:, 0], Y10, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # axins.plot(data[:, 0], Y10,   color = color[i], linestyle = '-',  linewidth = 2)
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

        ##==================== mother and son ==================================
        ### 局部显示并且进行连线,方法3
        # zone_and_linked(axs, axins, 800, 850, data[:, 0] , [Y1, Y2, Y3, Y5,  Y7,  Y9], 'bottom',x_ratio = 0.3, y_ratio = 0.3)
        ## linewidth
        # bw = 1
        # axins.spines['bottom'].set_linewidth(bw)  ###设置底部坐标轴的粗细
        # axins.spines['left'].set_linewidth(bw)    ###设置左边坐标轴的粗细
        # axins.spines['right'].set_linewidth(bw)   ###设置右边坐标轴的粗细
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
        out_fig.savefig( f"{model}_1bitAcc_SNR.eps")
        # out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.pdf") )
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_1bitAcc_SNR.pdf") )
        plt.show()
        plt.close()
        return




# def main():
pl = ResultPlot( ) # 1: '2022-10-12-17:38:12'  2:'2022-10-14-09:56:05'


model = "2nn"


pl.PerformanceSNR_with1bit(model = model)

# pl.AccVsSNR_with1bit(model = model)
















