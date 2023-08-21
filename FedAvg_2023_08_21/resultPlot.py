
# -*- coding: utf-8 -*-
"""
Created on 2023/04/25
@author: Junjie Chen

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

color = ['#1E90FF', '#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE', '#00CED1', '#CD5C5C', '#FF0000',  '#0000FF', '#7B68EE', '#808000' ]
lsty = [(0, (3, 10, 1, 10, 1, 10)), (0, (1, 1)), (0, (1, 2)), (0, (5, 1)), (0, (1, 10)), (0, (1, 2)),  (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),  '-', ':', '--', '-.', ]
alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

Utility.set_printoption(5)

# 功能：
class ResultPlot():
    def __init__(self, ):
        #### savedir
        self.rootdir = f"{user_home}/FedAvg_DataResults/results/"
        self.home = f"{user_home}"
        self.savedir = os.path.join(self.home, 'FedAvg_DataResults/Figures_plot')
        os.makedirs(self.savedir, exist_ok=True)

        self.diff_publ_base         = torch.load(os.path.join(self.rootdir, "2023-07-08-21:11:49_FedAvg/TraRecorder.pt"))
        self.diff_publ_mask07       = torch.load(os.path.join(self.rootdir, "2023-07-09-21:56:33_FedAvg/TraRecorder.pt"))
        self.diff_publ_mask08       = torch.load(os.path.join(self.rootdir, "2023-07-09-15:45:34_FedAvg/TraRecorder.pt"))
        self.diff_publ_comp05       = torch.load(os.path.join(self.rootdir, "2023-07-09-17:14:50_FedAvg/TraRecorder.pt"))
        self.diff_publ_comp09       = torch.load(os.path.join(self.rootdir, "2023-07-09-17:49:28_FedAvg/TraRecorder.pt"))
        self.diff_lcoalDP           = torch.load(os.path.join(self.rootdir, "2023-07-12-15:18:23_FedAvg/TraRecorder.pt"))
        self.diff_quant8bit         = torch.load(os.path.join(self.rootdir, "2023-07-15-15:10:00_FedAvg/TraRecorder.pt"))
        self.diff_quant4bit         = torch.load(os.path.join(self.rootdir, "2023-07-15-15:51:50_FedAvg/TraRecorder.pt"))
        self.diff_quant2bit         = torch.load(os.path.join(self.rootdir, "2023-07-15-16:26:34_FedAvg/TraRecorder.pt"))

        return

    # 画出 JSCC (包括在指定信噪比 tra_test_snr 训练) 在指定测试信噪比 tra_test_snr, 指定压缩率 Rlist下训练完后, 不同攻击强度 epslist,PSNR 随 压缩率 Rlist 的曲线. 每条曲线对应一个 tra_test_snr
    def compare(self, tra_test_snr = [ 3 ], epslist = [0, 0.1, 0.2, 0.3, 0.4],  Rlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], PSNR = True, col_idx = 3):
        width = 10
        high  = 8
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True

        lb = "baseline"
        axs.plot(self.diff_publ_base[:, 0], self.diff_publ_base[:, 2], label = lb, color = 'purple', linestyle = '-', linewidth = 3)

        lb = "mask:0.7"
        axs.plot(self.diff_publ_mask07[:, 0], self.diff_publ_mask07[:, 2], label = lb, color = 'b', linestyle = '-', )
        lb = "mask:0.8"
        axs.plot(self.diff_publ_mask08[:, 0], self.diff_publ_mask08[:, 2], label = lb, color = 'b', linestyle = '--', )

        lb = "compress:0.5"
        axs.plot(self.diff_publ_comp05[:, 0], self.diff_publ_comp05[:, 2], label = lb, color = '#FFA500', linestyle = '-', )
        lb = "compress:0.9"
        axs.plot(self.diff_publ_comp09[:, 0], self.diff_publ_comp09[:, 2], label = lb, color = '#FFA500', linestyle = '--', )

        lb = "local DP"
        axs.plot(self.diff_lcoalDP[:, 0], self.diff_lcoalDP[:, 2], label = lb, color = 'r', linestyle = '-', )

        lb = "Quant: 8 bit"
        axs.plot(self.diff_quant8bit[:, 0], self.diff_quant8bit[:, 2], label = lb, color = 'g', linestyle = '-', )

        lb = "Quant: 4 bit"
        axs.plot(self.diff_quant4bit[:, 0], self.diff_quant4bit[:, 2], label = lb, color = 'g', linestyle = '--', )

        lb = "Quant: 2 bit"
        axs.plot(self.diff_quant2bit[:, 0], self.diff_quant2bit[:, 2], label = lb, color = 'g', linestyle = '-.', )

        # label
        font = {'family':'Times New Roman','style':'normal','size':22 }
        axs.set_xlabel("Communication Round", fontproperties=font)
        axs.set_ylabel( "Test Accuracy",      fontproperties = font )# , fontdict = font1

        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
        font1 = {'family':'Times New Roman','style':'normal','size':22, }
        legend1 = axs.legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = font1,)
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        axs.spines['bottom'].set_linewidth(1) ###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(1)   ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(1)  ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(1)    ###设置上部坐标轴的粗细

        axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 16, width = 2 )
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(18) for label in labels] #刻度值字号

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
        out_fig.savefig( os.path.join(self.savedir, f"FederateLearning.eps") )
        out_fig.savefig( os.path.join(self.savedir, f"FederateLearning.png") )
        # plt.show()
        plt.close()
        return

# def main():
pl = ResultPlot( ) # 1: '2022-10-12-17:38:12'  2:'2022-10-14-09:56:05'

pl.compare()




















