#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:16:13 2023

@author: jack
"""



import os
import sys


import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
import socket, getpass
import scipy

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
import Utility
Utility.set_printoption(5)


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



class ResultPlot():
    def __init__(self, ):
        ## savedir
        self.rootdir = f"{home}/FedAvg_DataResults/results/"
        self.home = f"{home}"
        self.savedir = os.path.join(self.home, 'FedAvg_DataResults/Figures_plot')
        os.makedirs(self.savedir, exist_ok=True)
        self.MeanVarCnn  = torch.load(os.path.join(self.rootdir, "2023-08-22-22:43:04_FedAvg/Mean_VarOfClients.pt"))[:200,:]
        return

    ## 1：将 FL 分别画出每轮所有客户端的各个参数更新的均值的均值，和方差的均值随着通信轮数的变化，画在一幅图上，上面是均值下面是方差，每个参数一条曲线。
    def Cnn_Clients_MeanVar_avg(self, num_clients, param_name_list, savepath = "", savename = "", smooth = False):
        X = self.MeanVarCnn[:, 0]

        width = 10
        high = 6*2
        fig, axs = plt.subplots(2, 1, figsize=(width, high), constrained_layout = True, sharex=True) # constrained_layout=True
        # cnt = 0
        param_len = len(param_name_list)

        ## Mean
        for kidx, key in enumerate(param_name_list):
            cols = [1 + 2*kidx + 2*param_len*j for j in range(num_clients)]
            tmp = self.MeanVarCnn[:, cols]
            avg = np.mean(tmp, axis=1)
            if smooth == True:
                avg = scipy.signal.savgol_filter(avg, 30, 10)
            axs[0].plot(X, avg, color = color[kidx], label = key, linewidth=2,)

        font1 = {'family':'Times New Roman','style':'normal','size':18, }
        legend1 = axs[0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        axs[0].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
        axs[0].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
        axs[0].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
        axs[0].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

        axs[0].tick_params(direction='in',axis='both',top=True,right=True,labelsize=20, width=3 )
        labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(18) for label in labels] #刻度值字号

        font = {'family':'Times New Roman','style':'normal','size':20, }
        axs[0].set_xlabel("Communication Round", fontproperties = font)
        axs[0].set_ylabel("Mean", fontproperties = font,) #  fontdict = font1

        ## Var
        for kidx, key in enumerate(param_name_list):
            cols = [1 + 2*kidx + 1 + 2*param_len*j for j in range(num_clients)]
            tmp = self.MeanVarCnn[:, cols]
            avg = np.mean(tmp, axis=1)
            if smooth == True:
                avg = scipy.signal.savgol_filter(avg, 30, 10)
            axs[1].plot(X, avg, color = color[kidx], label = key, linewidth=2,)

        font1 = {'family':'Times New Roman','style':'normal','size':18, }
        legend1 = axs[1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        axs[1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
        axs[1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
        axs[1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
        axs[1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

        axs[1].tick_params(direction='in',axis='both',top=True,right=True,labelsize=20,width=3)
        labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(18) for label in labels] #刻度值字号

        font = {'family':'Times New Roman','style':'normal','size':20, }
        axs[1].set_xlabel("Communication Round", fontproperties = font)
        axs[1].set_ylabel("Variance", fontproperties = font,) #  fontdict = font1

        # fontt = {'family':'Times New Roman','style':'normal','size':16}
        # if self.title != '':
            # plt.suptitle(self.title, fontproperties = fontt, )
        out_fig = plt.gcf()

        # out_fig.savefig( os.path.join(savepath, f"{self.basename}.pdf") )
        savepath = self.savedir
        out_fig.savefig(os.path.join(savepath, f"{savename}.eps") )
        out_fig.savefig(os.path.join(savepath, f"{savename}.pdf") )
        # plt.show()
        plt.close()
        return


    ## 2：将FL 分别画出每轮所有客户端的各个参数更新的均值(方差)的均值 随着通信轮数的变化，一幅图，每条曲线是一个参数矩阵的变化，要么是均值要么是方差。其实，这里的均值的均值相当于所有客户端同一个参数矩阵的均值。
    ## 实际上就是把1的两个子图单独拿出来
    def Cnn_Clients_VarOrMean_avg(self, num_clients, param_name_list, statistics = "var", model = "cnn",  smooth = False):
        X = self.MeanVarCnn[:, 0]

        width = 10
        high = 6
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True, sharex=True) # constrained_layout=True
        # cnt = 0
        param_len = len(param_name_list)

        ## Mean or Var
        if statistics.lower() == "var":
            k = 1
        else:
            k = 0
        for kidx, key in enumerate(param_name_list):
            cols = [1 + 2*kidx + k + 2*param_len*j for j in range(num_clients)]
            tmp = self.MeanVarCnn[:, cols]
            avg = np.mean(tmp, axis=1)
            if smooth == True:
                avg = scipy.signal.savgol_filter(avg, 30, 10)
            axs.plot(X, avg, color = color[kidx], label = key, linewidth=2,)

        font1 = {'family':'Times New Roman','style':'normal','size':24, }
        if statistics.lower() == "var":
            legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
        else:
            legend1 = axs.legend(loc='lower right', borderaxespad=0, edgecolor='black', prop=font1,)

        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

        axs.tick_params(direction='in',axis='both',top=True,right=True,labelsize=20, width=3 )
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(24) for label in labels] #刻度值字号

        font = {'family':'Times New Roman','style':'normal','size':28, }
        axs.set_xlabel("Communication Round", fontproperties = font)
        if statistics.lower() == "var":
            axs.set_ylabel("Variance", fontproperties = font,) #  fontdict = font1
        else:
            axs.set_ylabel("Mean", fontproperties = font,) #  fontdict = font1

        # fontt = {'family':'Times New Roman','style':'normal','size':16}
        # if self.title != '':
            # plt.suptitle(self.title, fontproperties = fontt, )
        out_fig = plt.gcf()

        # out_fig.savefig( os.path.join(savepath, f"{self.basename}.pdf") )
        savepath = self.savedir
        out_fig.savefig(os.path.join(savepath, f"{model}_{statistics}_avg.eps") )
        out_fig.savefig(os.path.join(savepath, f"{model}_{statistics}_avg.pdf") )
        # plt.show()
        plt.close()
        return


    ## 3：将FL 分别画出每轮其中[一个客户端]的所有参数更新的均值(方差) 随着通信轮数的变化，一幅图，要么是均值要么是方差
    def Cnn_Client_VarOrMean(self, num_clients, param_name_list, which_client = "1-10,number", model = "Cnn", statistics = "var", smooth = False):
        X = self.MeanVarCnn[:, 0]

        width = 10
        high = 6
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True, sharex=True) # constrained_layout=True
        # cnt = 0
        param_len = len(param_name_list)

        ## Mean or Var
        if statistics.lower() == "var":
            k = 1
        else:
            k = 0
        cols = range(1 + which_client*2*param_len, 1 + (which_client + 1)*2*param_len)
        for kidx, key in enumerate(param_name_list):
            col = cols[2 * kidx + k]
            tmp = self.MeanVarCnn[:, col]
            if smooth == True:
                tmp = scipy.signal.savgol_filter(tmp, 30, 3)
            if statistics.lower() == "var":
                tmp[np.where(tmp < 0) ] = 0
            axs.plot(X, tmp, color = color[kidx], label = key, linewidth=2,)
        font1 = {'family':'Times New Roman','style':'normal','size':24, }
        if statistics.lower() == "var":
            legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
        else:
            legend1 = axs.legend(loc='lower right', borderaxespad=0, edgecolor='black', prop=font1,)

        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

        axs.tick_params(direction='in',axis='both',top=True,right=True,labelsize=20, width=3 )
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(24) for label in labels] #刻度值字号

        font = {'family':'Times New Roman','style':'normal','size':28, }
        axs.set_xlabel("Communication Round", fontproperties = font)
        if statistics.lower() == "var":
            axs.set_ylabel("Variance", fontproperties = font,) #  fontdict = font1
        else:
            axs.set_ylabel("Mean", fontproperties = font,) #  fontdict = font1

        # fontt = {'family':'Times New Roman','style':'normal','size':16}
        # if self.title != '':
            # plt.suptitle(self.title, fontproperties = fontt, )
        out_fig = plt.gcf()

        # out_fig.savefig( os.path.join(savepath, f"{self.basename}.pdf") )
        savepath = self.savedir
        out_fig.savefig(os.path.join(savepath, f"{model}_client{which_client}_{statistics}.eps") )
        out_fig.savefig(os.path.join(savepath, f"{model}_client{which_client}_{statistics}.pdf") )
        # plt.show()
        plt.close()
        return


    ## 4:将FL 分别画出每轮各个参数更新在所有客户端之间的均值(方差)的最大值和最小值 随着通信轮数的变化，一幅图，要么是均值要么是方差。
    def Cnn_Clients_VarOrMean_minmax(self, num_clients, param_name_list, statistics = "var", model = "cnn", smooth = False):
        X = self.MeanVarCnn[:, 0]

        width = 10
        high = 6
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True, sharex=True) # constrained_layout=True
        # cnt = 0
        param_len = len(param_name_list)

        ## Mean or Var
        if statistics.lower() == "var":
            k = 1
        else:
            k = 0
        for kidx, key in enumerate(param_name_list):
            cols = [1 + 2*kidx + k + 2*param_len*j for j in range(num_clients)]
            tmp = self.MeanVarCnn[:, cols]
            Max = np.max(tmp, axis=1)
            if smooth == True:
                Max = scipy.signal.savgol_filter(Max, 30, 10)
            axs.plot(X, Max, color = color[kidx], label = "max " + key, linestyle = '-', linewidth=2,)

            Min = np.min(tmp, axis=1)
            if smooth == True:
                Min = scipy.signal.savgol_filter(Min, 30, 10)
            axs.plot(X, Min, color = color[kidx], label = "min " + key, linestyle = '--', linewidth=2,)


        font1 = {'family':'Times New Roman','style':'normal','size':12, }
        if statistics.lower() == "var":
            legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
        else:
            legend1 = axs.legend(loc='lower right', borderaxespad=0, edgecolor='black', prop=font1,)

        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

        axs.tick_params(direction='in',axis='both',top=True,right=True,labelsize=20, width=3 )
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(24) for label in labels] #刻度值字号

        font = {'family':'Times New Roman','style':'normal','size':28, }
        axs.set_xlabel("Communication Round", fontproperties = font)
        if statistics.lower() == "var":
            axs.set_ylabel("Variance", fontproperties = font,) #  fontdict = font1
        else:
            axs.set_ylabel("Mean", fontproperties = font,) #  fontdict = font1

        # fontt = {'family':'Times New Roman','style':'normal','size':16}
        # if self.title != '':
            # plt.suptitle(self.title, fontproperties = fontt, )
        out_fig = plt.gcf()

        # out_fig.savefig( os.path.join(savepath, f"{self.basename}.pdf") )
        savepath = self.savedir
        out_fig.savefig(os.path.join(savepath, f"{model}_{statistics}_minmax.eps") )
        out_fig.savefig(os.path.join(savepath, f"{model}_{statistics}_minmax.pdf") )
        # plt.show()
        plt.close()
        return


    ## 将FL 分别画出每轮所有客户端的[某个参数更新]的均值(方差)的最大值和最小值 随着通信轮数的变化，一幅图，要么是均值要么是方差。
    def Cnn_Clients_1Param_VarOrMean_minmax(self, num_clients, param_name_list, param_name, statistics = "var", model = "cnn", smooth = False):
        X = self.MeanVarCnn[:, 0]

        width = 10
        high = 6
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True, sharex=True) # constrained_layout=True
        # cnt = 0
        param_len = len(param_name_list)

        ## Mean or Var
        if statistics.lower() == "var":
            k = 1
        else:
            k = 0
        # for kidx, key in enumerate(param_name_list):
        kidx = param_name_list.index(param_name)
        cols = [1 + 2*kidx + k + 2*param_len*j for j in range(num_clients)]
        tmp = self.MeanVarCnn[:, cols]
        Max = np.max(tmp, axis=1)
        if smooth == True:
            Max = scipy.signal.savgol_filter(Max, 30, 10)
        axs.plot(X, Max, color = color[kidx], label = "max " + param_name, linestyle = '-', linewidth=2,)

        Min = np.min(tmp, axis=1)
        if smooth == True:
            Min = scipy.signal.savgol_filter(Min, 30, 10)
        axs.plot(X, Min, color = color[kidx], label = "min " + param_name, linestyle = '--', linewidth=2,)


        font1 = {'family':'Times New Roman','style':'normal','size':12, }
        if statistics.lower() == "var":
            legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
        else:
            legend1 = axs.legend(loc='lower right', borderaxespad=0, edgecolor='black', prop=font1,)

        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

        axs.tick_params(direction='in',axis='both',top=True,right=True,labelsize=20, width=3 )
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(24) for label in labels] #刻度值字号

        font = {'family':'Times New Roman','style':'normal','size':28, }
        axs.set_xlabel("Communication Round", fontproperties = font)
        if statistics.lower() == "var":
            axs.set_ylabel("Variance", fontproperties = font,) #  fontdict = font1
        else:
            axs.set_ylabel("Mean", fontproperties = font,) #  fontdict = font1

        # fontt = {'family':'Times New Roman','style':'normal','size':16}
        # if self.title != '':
            # plt.suptitle(self.title, fontproperties = fontt, )
        out_fig = plt.gcf()

        # out_fig.savefig( os.path.join(savepath, f"{self.basename}.pdf") )
        savepath = os.path.join(self.savedir, "1Param")
        os.makedirs(savepath, exist_ok=True)
        # out_fig.savefig(os.path.join(savepath, f"{model}_{statistics}_{param_name}_minmax.eps") )
        # out_fig.savefig(os.path.join(savepath, f"{model}_{statistics}_{param_name}_minmax.pdf") )
        out_fig.savefig(os.path.join(savepath, f"{model}_{statistics}_{param_name}_minmax.png") )
        # plt.show()
        plt.close()
        return





cnn_key_order = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
nn2_key_order = []

key_order = cnn_key_order

model = "Cnn"
num_clients  = 10
smooth = True   #  True   False

rs = ResultPlot()
# rs.Cnn_Clients_MeanVar_avg(10, key_order, savename =  f"{model}_mean_var_avg")  ##  Cnn_mean_var_avg   2nn_mean_var_avg

# rs.Cnn_Clients_VarOrMean_avg(10, key_order, model = model, statistics = "mean", smooth = False  )
# rs.Cnn_Clients_VarOrMean_avg(10, key_order, model = model, statistics = "var", smooth = False )

for c in range(10):
    rs.Cnn_Client_VarOrMean(10, key_order, which_client = c, model = model,  statistics = "var", smooth = smooth  )


rs.Cnn_Clients_VarOrMean_minmax(num_clients, key_order, statistics = "var", model = "cnn", smooth = smooth )


for param_name in key_order:
    rs.Cnn_Clients_1Param_VarOrMean_minmax(num_clients, key_order, param_name, statistics = "mean", model = "cnn", smooth = smooth )




















































































































