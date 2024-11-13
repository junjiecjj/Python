#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:16:13 2023

@author: jack
"""



import os
import sys


import matplotlib
# matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
import socket, getpass
from scipy.signal import savgol_filter

# 获取当前系统主机名
# host_name = socket.gethostname()
# 获取当前系统用户名
# user_name = getpass.getuser()
# 获取当前系统用户目录
# user_home = os.path.expanduser('~')
home = os.path.expanduser('~')

# 本项目自己编写的库
# from option import args
sys.path.append("..")
# checkpoint
# import Utility
# Utility.set_printoption(5)


fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', '$\\clubsuit$', '$\\bowtie$', '$\\boxdot$', '$\\boxtimes$', '$\\boxminus$',  '$\\boxplus$', '$\\bigodot$','$\\bigoplus$', '$\\bigotimes$', '$\\bigstar$', '$\\blacksquare$', '$\circledR$', '$\circledS$',  '$\\bigtriangledown$','$\\bigtriangleup$','$\\blacktriangleleft$', '$\\blacktriangleright$', '$\\diamondsuit',   '$\heartsuit$', '$666$', '$\\varrho$', '$\\Omega$', '$\\nabla$', '$\\Upsilon$', '$\\mho$', '$\\Phi$',  '$\\divideontimes$', '$\\otimes$', '$\\odot$', '$\\ominus$', '$\\oplus$', '$\\oslash$', '$\\otimes$']

color = ['#1E90FF', '#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE', '#00CED1', '#CD5C5C', '#FF0000',  '#0000FF', '#7B68EE', '#808000' ]
lsty = [(0, (3, 10, 1, 10, 1, 10)), (0, (1, 1)), (0, (1, 2)), (0, (5, 1)), (0, (1, 10)), (0, (1, 2)),  (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),  '-', ':', '--', '-.', ]
alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

# mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
color = ['#000000','#0000FF', '#DC143C', '#006400', '#9400D3', '#ADFF2F', '#FF00FF', '#00CED1' ,'#FFA500', ]
# color = ['#1E90FF','#FF6347','#00FF00','#0000FF','#4ea142','#FF00FF','#FFA500','#800080','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE','#808000']



class ResultPlot():
    def __init__(self, ):
        ## savedir
        self.rootdir = f"{home}/FedAvg_DataResults/results/"
        self.home = f"{home}"
        self.savedir = os.path.join(self.home, 'FedAvg_DataResults/Figures_plot')
        os.makedirs(self.savedir, exist_ok=True)
        ### local_batchsize=128, loc_epochs=10:
        # self.noiid_unbla  = torch.load(os.path.join(self.rootdir, "2023-08-26-21:41:35_FedAvg/MeanVarL12OfClients.pt"))
        # ##
        # self.noiid_bla    = torch.load(os.path.join(self.rootdir, "2023-08-27-10:03:12_FedAvg/MeanVarL12OfClients.pt"))
        # ##
        # self.iid_bla      = torch.load(os.path.join(self.rootdir, "2023-08-27-01:08:19_FedAvg/MeanVarL12OfClients.pt"))

        self.noiid_bla_t = "2023-09-05-15:24:12_FedAvg" # "2023-09-05-15:24:12_FedAvg"   "2023-10-13-20:08:50_FedAvg"
        self.iid_bla_t = "2023-09-11-22:16:54_FedAvg"   # "2023-09-05-16:35:38_FedAvg" "2023-09-11-22:16:54_FedAvg" "2023-10-13-22:07:17_FedAvg"


        ## local_batchsize=50, loc_epochs=5
        # self.noiid_unbla  = torch.load(os.path.join(self.rootdir, "2023-08-26-21:41:35_FedAvg/MeanVarL12OfClients.pt"))
        ##  2023-09-05-15:24:12_FedAvg
        self.noiid_bla    = torch.load(os.path.join(self.rootdir, f"{self.noiid_bla_t}/MeanVarL12OfClients.pt"), weights_only = False)
        ## 2023-09-05-16:35:38_FedAvg    2023-09-11-22:16:54_FedAvg
        self.iid_bla      = torch.load(os.path.join(self.rootdir, f"{self.iid_bla_t}/MeanVarL12OfClients.pt"), weights_only = False)
        return


    ##统计 每一轮通信过程，各个客户端所有参数数组(融合为一个数组)的均值、方差、L1范数的最大值最小值的差值，为了验证每轮各个客户端之间的差异
    def compare_no_iid_unblance_minmax(self, num_clients, stastic = ["Mean", "1-norm", "2-norm", "Variance"], model = "cnn" ):

        width = 8
        high = 7
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True, sharex=True) # constrained_layout=True
        # cnt = 0
        L = len(stastic)
        ## Mean
        # for idx, key in enumerate(stastic):
        idx = 2
        cols = [1 + idx + L*j for j in range(num_clients)]


        ## Non-IID, Unblance
        # X = self.noiid_unbla[:, 0]
        # tmp = self.noiid_unbla[:, cols]
        # Max = np.max(tmp, axis=1)
        # Min = np.min(tmp, axis=1)
        # diff =  Max - Min
        # # diff = savgol_filter(diff, 25, 10)
        # # diff = savgol_filter(diff, 25, 3)
        # # diff = savgol_filter(diff, 25, 3)
        # # diff = savgol_filter(diff, 25, 1)
        # print(diff.shape)
        # # axs.plot(X, diff, color = color[0], label = "Non-IID, Unblance", linewidth = 2,)

        ## Non-IID, blance
        X = self.noiid_bla[:, 0]
        tmp = self.noiid_bla[:, cols]
        Max = np.max(tmp, axis=1)
        Min = np.min(tmp, axis=1)
        diff =   Max - Min
        # diff =  savgol_filter(diff, 25, 3)
        diff = savgol_filter(diff, 25, 3)
        # diff = savgol_filter(diff, 25, 1)
        # diff = savgol_filter(diff, 25, 1)
        axs.plot(X, diff, color = color[1], label = "Non-IID", linewidth = 5,)

        ##  IID,
        X = self.iid_bla[:, 0]
        tmp = self.iid_bla[:, cols]
        Max = np.max(tmp, axis=1)
        Min = np.min(tmp, axis=1)
        diff =   Max - Min
        diff =  savgol_filter(diff, 25, 3)
        # diff = savgol_filter(diff, 25, 8)
        # diff = savgol_filter(diff, 25, 3)
        # diff = savgol_filter(diff, 25, 1)
        axs.plot(X, diff, color = color[2], label = "IID", linewidth = 5,)

        # axs.grid()
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font1 = {'family':'Times New Roman','style':'normal','size':50, }
        legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        ## border
        bw = 4
        axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        axs.tick_params(direction='in',axis='both',top=True,right=True,labelsize=24, width=3 )
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(45) for label in labels] #刻度值字号

        font = {'family':'Times New Roman','style':'normal','size':50, }
        axs.set_xlabel("Round " + r"${t}$", fontproperties = font)
        axs.set_ylabel(r"$\beta(t)$" , fontproperties = font,) #  fontdict = font1
        fontt = {'family':'Times New Roman','style':'normal','size':40}
        plt.suptitle("MNIST, 2NN", fontproperties = fontt, )


        out_fig = plt.gcf()
        savepath = self.savedir
        out_fig.savefig(os.path.join(savepath, f"{model}_NonIID_UnBlance_minmax.eps") )
        out_fig.savefig(os.path.join(savepath, f"{model}_NonIID_UnBlance_minmax.pdf") )
        # plt.show()
        plt.close()
        return

    ##统计 每一轮通信过程，各个客户端所有参数数组(融合为一个数组)的均值、方差、L1范数的和的均值，为了验证各轮之间的差异性越来越小
    def compare_no_iid_unblance_avg(self, num_clients, stastic = ["Mean", "1-norm", "2-norm", "Variance"], model = "cnn" ):

        width = 8
        high = 7
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True, sharex=True) # constrained_layout=True
        # cnt = 0
        L = len(stastic)
        ## Mean
        # for idx, key in enumerate(stastic):
        idx = 2
        cols = [1 + idx + L*j for j in range(num_clients)]

        ## Non-IID, Unblance
        # X = self.noiid_unbla[:, 0]
        # tmp = self.noiid_unbla[:, cols]
        # Mean = np.mean(tmp, axis=1)
        # Mean = savgol_filter(Mean, 21, 3)
        # # axs.plot(X, Mean, color = color[0], label = "Non-IID, Unblance", linewidth = 2,)

        s = 2
        ## Non-IID, blance
        X = self.noiid_bla[:, 0][s:]
        tmp = self.noiid_bla[:, cols]
        Mean = np.mean(tmp, axis=1)
        # 进行均值滤波去除高频噪声
        # window_size = 6  # 滤波窗口大小
        # Mean = np.convolve(Mean, np.ones(window_size) / window_size, mode='same')
        # Mean = savgol_filter(Mean, 30, 8)
        # Mean = savgol_filter(Mean, 25, 3)
        # Mean = savgol_filter(Mean, 25, 1)
        # Mean = savgol_filter(Mean, 25, 1)
        axs.plot(X, Mean[s:], color = color[1], label = "Non-IID", linewidth = 5,)

        ##  IID,
        X = self.iid_bla[:, 0][s:]
        tmp = self.iid_bla[:, cols][s:]
        Mean = np.mean(tmp, axis=1)
        # 进行均值滤波去除高频噪声
        # window_size = 6  # 滤波窗口大小
        # Mean = np.convolve(Mean, np.ones(window_size) / window_size, mode='same')
        # Mean = savgol_filter(Mean, 30, 8)
        # Mean = savgol_filter(Mean, 25, 3)
        # Mean = savgol_filter(Mean, 25, 1)
        # Mean = savgol_filter(Mean, 25, 1)
        axs.plot(X, Mean, color = color[2], label = "IID", linewidth = 5,)

        ## legend
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font1 = {'family':'Times New Roman','style':'normal','size':50, }
        legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        ## border
        bw = 4
        axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        ## ticks
        axs.tick_params(direction='in',axis='both',top=True,right=True,labelsize=24, width=3 )
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(45) for label in labels] #刻度值字号

        ## xlabel
        font = {'family':'Times New Roman','style':'normal','size':50, }
        axs.set_xlabel("Round " + r"${t}$", fontproperties = font)
        # axs.set_ylabel(r"$\mathrm{{2\_Norm}}_{mean}$", fontproperties = font,) #  fontdict = font1
        axs.set_ylabel(r"$\alpha(t)$", fontproperties = font,) #  fontdict = font1
        fontt = {'family':'Times New Roman','style':'normal','size':40}
        plt.suptitle("MNIST, 2NN", fontproperties = fontt, )
        out_fig = plt.gcf()

        savepath = self.savedir
        # out_fig.savefig(f"./figures/{model}_NonIID_UnBlance_avg.eps")
        # out_fig.savefig(os.path.join(savepath, f"{model}_NonIID_UnBlance_avg.pdf") )
        plt.show()
        plt.close()
        return


    def performanceIIDvsNonIID(self, model = '2nn', ):
        lw = 2
        width = 8
        high  = 7
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        i = 0
        ##==========================================================
        lb = "Non-IID"
        ## 2023-09-05-15:24:12_FedAvg
        data  =  torch.load(os.path.join(self.rootdir, "2023-09-05-15:24:12_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        Y = savgol_filter(Y, 25, 3)
        axs.plot(data[:, 0], Y, label = lb, color = color[1], linestyle = '-', linewidth = 5)
        i = i+1

        ##================ baseline =========================================
        lb = "IID"
        ## 2023-09-11-22:16:54_FedAvg
        data  =   torch.load(os.path.join(self.rootdir, "2023-09-11-22:16:54_FedAvg/TraRecorder.pt"))
        Y = data[:, 2]
        # Y = savgol_filter(Y, 25, 3)
        axs.plot(data[:, 0], Y, label = lb, color = color[2], linestyle = '-', linewidth = 5)
        i = i+1

        ##===========================================================
        ## legend
        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
        font1 = {'family':'Times New Roman','style':'normal','size':50, }
        legend1 = axs.legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = font1,)
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        ## border
        bw = 4
        axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

        ## ticks
        axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 16, width = bw)
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(45) for label in labels] #刻度值字号

        # axs.set_xlim(0.05, 0.94)  #拉开坐标轴范围显示投影
        # axs.set_ylim(0.0, 1.001)  #拉开坐标轴范围显示投影
        # x_major_locator=MultipleLocator(0.1)
        # axs.xaxis.set_major_locator(x_major_locator)
        y_major_locator=MultipleLocator(0.2)
        axs.yaxis.set_major_locator(y_major_locator)

        ## xlabel
        # axs.grid()
        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        font = {'family':'Times New Roman','style':'normal','size':50 }
        axs.set_xlabel("Round " + r"${t}$", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1
        fontt = {'family':'Times New Roman','style':'normal','size':40}
        plt.suptitle("MNIST, 2NN", fontproperties = fontt, )

        out_fig = plt.gcf()
        savepath = self.savedir
        out_fig.savefig(f"./figures/{model}_IIDvsNonIID_perform.eps")
        # out_fig.savefig(os.path.join(savepath, f"{model}_IIDvsNonIID_perform.pdf") )
        # out_fig.savefig(os.path.join(savepath, f"{model}_Param_debug.pdf") )
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_8bitNonIID_performance.pdf") )
        # plt.show()
        plt.close()
        return

    def compare_noiid_unblance_avg_minmax(self, num_clients, stastic = ["Mean", "1-norm", "2-norm", "Variance"], model = "cnn"):
        width = 8
        high = 6
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True, ) # constrained_layout=True
        # cnt = 0
        L = len(stastic)
        ## Mean
        ## for idx, key in enumerate(stastic):
        idx = 2
        cols = [1 + idx + L*j for j in range(num_clients)]

        ##================================  left Y ===========================================
        # ## Non-IID, Unblance

        ## Non-IID, blance
        X = self.noiid_bla[:, 0]
        tmp = self.noiid_bla[:, cols]
        Mean = np.mean(tmp, axis=1)
        Mean = savgol_filter(Mean, 25, 3)
        Mean = savgol_filter(Mean, 25, 3)
        Mean = savgol_filter(Mean, 25, 1)
        Mean = savgol_filter(Mean, 25, 1)
        l1 = axs.plot(X, Mean, color = color[1], label = r"$\alpha(t)$,"+" Non-IID", linestyle = '-', linewidth = 3,)

        ##  IID,
        X = self.iid_bla[:, 0]
        tmp = self.iid_bla[:, cols]
        Mean = np.mean(tmp, axis=1)
        Mean = savgol_filter(Mean, 25, 3)
        Mean = savgol_filter(Mean, 25, 3)
        Mean = savgol_filter(Mean, 25, 1)
        Mean = savgol_filter(Mean, 25, 1)
        l2 = axs.plot(X, Mean, color = color[1], label = r"$\alpha(t)$,"+" IID", linestyle = '--', linewidth = 3,)

        ##================================  right Y ===========================================
        ax2 = axs.twinx()
        ### Non-IID, Unblance

        ## Non-IID, blance
        X = self.noiid_bla[:, 0]
        tmp = self.noiid_bla[:, cols]
        Max = np.max(tmp, axis=1)
        Min = np.min(tmp, axis=1)
        diff =   Max - Min
        diff =  savgol_filter(diff, 25, 3)
        diff = savgol_filter(diff, 25, 3)
        diff = savgol_filter(diff, 25, 1)
        diff = savgol_filter(diff, 25, 1)
        l3 = ax2.plot(X, diff, color = color[2], label = r"$\beta(t)$,"+" non-IID", linestyle = '-', linewidth = 3,)

        ##  IID,
        X = self.iid_bla[:, 0]
        tmp = self.iid_bla[:, cols]
        Max = np.max(tmp, axis=1)
        Min = np.min(tmp, axis=1)
        diff =   Max - Min
        diff =  savgol_filter(diff, 25, 10)
        diff = savgol_filter(diff, 25, 3)
        diff = savgol_filter(diff, 25, 3)
        diff = savgol_filter(diff, 25, 1)
        l4 = ax2.plot(X, diff, color = color[2], label = r"$\beta(t)$,"+" IID", linestyle = '--', linewidth = 3,)

        ##==============================  left X ================================================
        # axs.grid()
        # font1 = {'family':'Times New Roman','style':'normal','size':30, }
        # legend1 = axs.legend(loc='upper left', borderaxespad=0, edgecolor='black', prop=font1,)
        # frame1 = legend1.get_frame()
        # frame1.set_alpha(1)
        # frame1.set_facecolor('none')  # 设置图例legend背景透明

        axs.spines['bottom'].set_linewidth(2.5) ### 设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(2.5)   ### 设置左边坐标轴的粗细
        axs.spines['left'].set_color(color[1])  ### 设置边框线颜色
        axs.spines['top'].set_linewidth(2.5)    ### 设置上部坐标轴的粗细


        font = {'family':'Times New Roman','style':'normal','size':32 }
        axs.set_xlabel("Communication Round " + r"${t}$", fontproperties = font)
        font = {'family':'Times New Roman','style':'normal','size':32, 'color': color[1]}
        axs.set_ylabel(r"$\alpha(t)$", fontdict = font,) #  fontdict = font1

        axs.tick_params(direction='in', axis='y',   width=4, labelcolor = color[1], colors= color[1],)
        axs.tick_params(direction='in', axis='x', top=True, width=4,  )
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(32) for label in labels] #刻度值字号

        ax2.spines['left'].set_color(color[1])  ### 设置边框线颜色
        ax2.spines['right'].set_linewidth(2.5)  ###设置右边坐标轴的粗细
        # ax2.spines['top'].set_linewidth(2.5)    ###设置上部坐标轴的粗细
        ax2.spines['right'].set_color(color[2]) ## 设置边框线颜色

        font = {'family':'Times New Roman','style':'normal','size':32, 'color': color[2]}
        ax2.set_ylabel(r"$\beta(t)$", fontdict = font ) #  fontdict = font1

        ax2.tick_params(direction='in', axis='y', top=True, right=True, width=4, colors=color[2],)
        labels =  ax2.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]  #刻度值字体
        [label.set_fontsize(32) for label in labels] #刻度值字号

        lines = l1 + l2 + l3 + l4
        labels = [h.get_label() for h in lines]
        font1 = {'family':'Times New Roman','style':'normal','size':30 }
        plt.legend(lines, labels, loc='best', borderaxespad=0, edgecolor='black', prop=font1, facecolor = 'none',  )

        ##=============================================
        out_fig = plt.gcf()

        savepath = self.savedir
        out_fig.savefig(os.path.join(savepath, f"{model}_NonIID_avg_minmax_twin.eps") )
        out_fig.savefig(os.path.join(savepath, f"{model}_NonIID_avg_minmax_twin.pdf") )
        # plt.show()
        plt.close()
        return




# cnn_key_order = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
nn2_key_order = []

# key_order = cnn_key_order

model = "2nn"
num_clients  = 10
smooth = True   #  True   False

rs = ResultPlot()


# rs.compare_no_iid_unblance_minmax(num_clients, stastic = ["Mean", "1-norm", "2-norm", "Variance"], model = model)

# Fig 2a
rs.compare_no_iid_unblance_avg(num_clients, stastic = ["Mean", "1-norm", "2-norm", "Variance"], model = model)

# Fig 2b
# rs.performanceIIDvsNonIID(model = model)

# rs.compare_noiid_unblance_avg_minmax(num_clients, stastic = ["Mean", "1-norm", "2-norm", "Variance"], model = model)













































































































