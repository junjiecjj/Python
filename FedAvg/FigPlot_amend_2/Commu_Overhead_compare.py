
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
    def Commu_overhead_SNR(self, model = '2nn', ):  ## E = 10, B = 50

        ##======================= mother ===================================
        lw = 2
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        ##======================= son =====================================
        # axins = axs.inset_axes((0.5, 0.4, 0.45, 0.4))

        ##========================================================================
        ##                            DQ
        ##========================================================================

        cv_dq = []
        ##================ DQ  bit, 0 dB =============================
        lb = "DQ, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-24-17:14:49_FedAvg/TraRecorder.pt"))
        cv_dq.append(data[-1, 5])

        ##================ DQ  bit, 0.25 dB =============================
        lb = "DQ, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-24-21:54:46_FedAvg/TraRecorder.pt"))
        cv_dq.append(data[-1, 5])



        ##================ DQ  bit, 0.5 dB =============================
        lb = "DQ, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-25-09:14:03_FedAvg/TraRecorder.pt"))
        cv_dq.append(data[-1, 5])



        ##================ DQ  bit, 0.75 dB =============================
        lb = "DQ, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-25-10:56:38_FedAvg/TraRecorder.pt"))
        cv_dq.append(data[-1, 5])



        ##================ DQ  bit, 1 dB =============================
        lb = "DQ, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-25-13:18:13_FedAvg/TraRecorder.pt"))
        cv_dq.append(data[-1, 5])



        ##================ DQ  bit, 1.25 dB =============================
        lb = "DQ, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-25-15:28:38_FedAvg/TraRecorder.pt"))
        cv_dq.append(data[-1, 5])



        ##================ DQ  bit, 1.5 dB =============================
        lb = "DQ, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-25-16:56:18_FedAvg/TraRecorder.pt"))
        cv_dq.append(data[-1, 5])



        ##================ DQ  bit, 1.75 dB =============================
        lb = "DQ, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-25-19:42:42_FedAvg/TraRecorder.pt"))
        cv_dq.append(data[-1, 5])



        ##================ DQ  bit, 2 dB =============================
        lb = "DQ, SNR = 2(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-24-14:48:57_FedAvg/TraRecorder.pt"))
        cv_dq.append(data[-1, 5])


        SNR = np.arange(0, 2.1, 0.25)
        lb = "DQ"
        axs.plot(SNR, cv_dq, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        i += 1
        ##========================================================================
        ##                           8 bit
        ##========================================================================

        SNR = np.arange(0, 2.1, 0.25)
        lb = "8-bit"
        rounds = 1000
        cv_dq = [199210*8*1000]*len(SNR)
        axs.plot(SNR, cv_dq, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        ##========================================================================
        ##                           4 bit
        ##========================================================================

        SNR = np.arange(0, 2.1, 0.25)
        lb = "4-bit"
        rounds = 1000
        cv_dq = [199210*4*1000]*len(SNR)
        axs.plot(SNR, cv_dq, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        ##========================================================================
        ##                           1 bit
        ##========================================================================

        SNR = np.arange(0, 2.1, 0.25)
        lb = "1-bit"
        rounds = 1000
        cv_dq = [199210*1*1000]*len(SNR)
        axs.plot(SNR, cv_dq, label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        i += 1

        ##===========================================================
        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel("SNR (dB)", fontproperties=font)
        axs.set_ylabel( "Overhead (bits)",      fontproperties = font )# , fontdict = font1

        ## legend
        font1 = {'family':'Times New Roman','style':'normal','size':30, }
        # legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.16, 0.02, 0.2, 0.5), borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',) ## loc = 'lower left',
        legend1 = axs.legend(loc='lower left', bbox_to_anchor = (0.7, 0.6, 0.2, 0.5),  borderaxespad = 0, edgecolor = 'black', prop = font1, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
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
        # ### 局部显示并且进行连线,方法3
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
        out_fig.savefig(os.path.join(savepath, f"{model}_commu_overhead.eps") )
        # out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.pdf") )
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_DQAcc_SNR.pdf") )
        plt.show()
        plt.close()
        return


    def Commu_overhead_1SNR(self, model = '2nn', ):  ## E = 10, B = 50
        color = ['#FF0000', '#1E90FF', '#00FF00', '#0000FF','#4ea142','#FF00FF','#FFA500','#800080','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE','#808000']
        ##======================= mother ===================================
        lw = 2
        width = 10
        high  = 8.5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        ##======================= son =====================================
        # axins = axs.inset_axes((0.5, 0.4, 0.45, 0.4))
        c1 = '#FF00FF'  #'#FF0000'
        c2 = '#FF8C00'  # '#1E90FF'
        c3 = '#008080'
        c4 = '#B22222'
        ##========================================================================
        ##                            DQ
        ##========================================================================
        SNR = 1
        ##================ DQ  bit, 1 dB =============================
        lb = f"Proposed FL, SNR = {SNR}(dB)"

        data = torch.load(os.path.join(self.rootdir, "2023-09-25-13:18:13_FedAvg/TraRecorder.pt"))
        axs.plot(data[:,0], data[:,5], label = lb, color = c1, linestyle = '-',  linewidth = 3, marker = 'o', markerfacecolor='white', markersize = 12, markevery=100)
        i += 1


        # ##================ DQ  bit, 1.5 dB =============================
        # lb = "DQ, SNR = 1.5(dB)"

        # data = torch.load(os.path.join(self.rootdir, "2023-09-25-16:56:18_FedAvg/TraRecorder.pt"))
        # axs.plot(data[:,0], data[:,5], label = lb, color = color[i], linestyle = '-',  linewidth = 2)
        # i += 1


        ##========================================================================
        ##                           8 bit
        ##========================================================================

        lb = f"8-bit FL, SNR = {SNR}(dB)"
        axs.plot(data[:,0], data[:, 0]*199210*8, label = lb, color = c4, linestyle = '-', linewidth = 3, marker = "v", markersize = 12, markevery=100)
        i += 1

        ##========================================================================
        ##                           4 bit
        ##========================================================================

        lb = f"4-bit FL, SNR = {SNR}(dB)"

        axs.plot(data[:,0], data[:, 0]*199210*4, label = lb, color = c2, linestyle = '-',  linewidth = 3, marker = "^", markersize = 12, markevery=100)
        i += 1

        ##========================================================================
        ##                           1 bit
        ##========================================================================

        lb = f"1-bit FL, SNR = {SNR}(dB)"
        axs.plot(data[:,0], data[:, 0]*199210*1, label = lb, color = c3, linestyle = '-', linewidth = 3, marker = "*", markersize = 12,  markevery=100)
        i += 1

        ##===========================================================
        ## xlabel
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':35 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Overhead (bits)",      fontproperties = font )# , fontdict = font1

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
        # ### 局部显示并且进行连线,方法3
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
        out_fig.savefig(f"./figures/{model}_1SNR_commu_overhead.eps")
        # out_fig.savefig(os.path.join(savepath, f"{model}_8bitNonIID_performance.pdf") )
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_1SNR_commu_overhead.pdf") )
        plt.show()
        plt.close()
        return



# def main():
pl = ResultPlot( ) # 1: '2022-10-12-17:38:12'  2:'2022-10-14-09:56:05'


model = "2nn"


# pl.Commu_overhead_SNR(model = model)

## Paper Fig.8(b)
pl.Commu_overhead_1SNR(model = model)









