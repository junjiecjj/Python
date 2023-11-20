
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
        # ##================ 1 bit,  0.5 error =============================
        # lb = "1-bit, lr=0.001, G=8, "+ r"$\epsilon = 0.5$"
        # data = torch.load(os.path.join(self.rootdir, "2023-10-16-14:17:59_FedAvg/TraRecorder.pt"))
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
        out_fig.savefig(os.path.join(savepath, f"{model}_1bit_G8.eps") )
        # out_fig.savefig(os.path.join(savepath, f"{model}_Param_debug.pdf") )
        # out_fig.savefig(os.path.join("/home/jack/文档/中山大学/00 我的论文/Federate_learning_Com/Figures", f"{model}_8bitNonIID_performance.pdf") )
        plt.show()
        plt.close()
        return

# def main():
pl = ResultPlot( ) # 1: '2022-10-12-17:38:12'  2:'2022-10-14-09:56:05'

model = "2nn"

pl.Onebit_G7(model = model)

pl.Onebit_G8(model = model)


















