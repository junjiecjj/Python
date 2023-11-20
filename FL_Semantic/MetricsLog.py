#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023/04/25
@author: Junjie Chen

"""

# 系统库
import math
import os, sys
import time, datetime
import numpy as np
from scipy import stats
import torch

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
#内存分析工具
# from memory_profiler import profile
# import objgraph


# 本项目自己编写的库
# sys.path.append("../")
# from Option import args
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
plt.rc('font', family='Times New Roman')

mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]
color = ['#000000','#0000FF', '#DC143C', '#006400', '#9400D3', '#ADFF2F', '#FF00FF', '#00CED1' ,'#FFA500', ]

#============================================================================================================================
#                                                   统计 正确率
#============================================================================================================================
class Recorder(object,):
    def __init__(self, Len = 2,  metname = "MSE loss_ Acc"):
        self.metrics = [i.strip() for i in metname.split("/")]
        self.len = Len
        if len(self.metrics) != self.len:
            print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")
            raise ValueError("len is inconsistent")
        self.data = np.empty((0,  self.len))
        return

    def addline(self, firstcol):
        self.data = np.append(self.data , np.zeros( (1, self.len) ), axis = 0 )
        self.data[-1, 0] = firstcol
        return

    def assign(self, met):
        if len(met) != self.len - 1:
            print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")
            raise ValueError("len is inconsistent")
        self.data[-1, 1:] = met
        return

    def __getitem__(self, idx):
        return self.data[-1, idx]

    def save(self, path, name):
        torch.save(self.data, os.path.join(path, name))
        return


# acc = Recorder(3, metname='eps/acc/psnr')
# print(f"acc.metrics = {acc.metrics}")

# for i in range(10):
#     acc.addline(i)
#     acc.assign([i+1, i+2])


#============================================================================================================================
## 记录联邦学习通信轮次和各个客户端模型参数的均值和方差的变化情况表
class RecorderFL(object,):
    def __init__(self, Len = 2,  ):
        self.len = Len
        self.data = np.empty((0,  self.len))
        return

    def addline(self, cround):
        self.data = np.append(self.data , np.zeros( (1, self.len) ), axis = 0 )
        self.data[-1, 0] = cround
        return

    def assign(self, met):
        if len(met) != self.len - 1:
            print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")
            raise ValueError("len is inconsistent")
        self.data[-1, 1:] = met
        return

    def __getitem__(self, idx):
        return self.data[-1, idx]

    def save(self, path, name):
        torch.save(self.data, os.path.join(path, name))
        return

    ##统计 每一轮通信过程，各个参数矩阵在所有被选择的客户端的均值和方差
    def Clients_MeanVar_avg(self, num_clients, param_name_list, savepath = '', savename = ''):
        X = self.data[:, 0]

        width = 10
        high = 6*2
        fig, axs = plt.subplots(2, 1, figsize=(width, high), constrained_layout = True, sharex=True) # constrained_layout=True
        # cnt = 0
        param_len = len(param_name_list)

        ## Mean
        for kidx, key in enumerate(param_name_list):
            cols = [1 + 2*kidx + 2*param_len*j for j in range(num_clients)]
            tmp = self.data[:, cols]
            avg = np.mean(tmp, axis=1)
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
            tmp = self.data[:, cols]
            avg = np.mean(tmp, axis=1)
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
        out_fig.savefig(os.path.join(savepath, f"{savename}.eps") )
        out_fig.savefig(os.path.join(savepath, f"{savename}.pdf") )
        # plt.show()
        plt.close()
        return


    ##统计 每一轮通信过程，各个客户端所有参数数组(融合为一个数组)的均值、方差、L1范数的最大值最小值的差值，为了验证每轮各个客户端之间的差异
    def Client_mean_var_L12(self, num_clients, stastic = ["Mean", "1-norm", "2-norm", "Variance"], savepath = '', savename = ''):
        X = self.data[:, 0]
        width = 4*4
        high = 3
        fig, axs = plt.subplots(1, 4, figsize=(width, high), constrained_layout = True, sharex=True) # constrained_layout=True
        # cnt = 0
        L = len(stastic)
        ## Mean
        for idx, key in enumerate(stastic):
            cols = [1 + idx + L*j for j in range(num_clients)]
            tmp = self.data[:, cols]
            Max = np.max(tmp, axis=1)
            Min = np.min(tmp, axis=1)
            axs[idx].plot(X, Max - Min, color = color[idx], label = key, linewidth=2,)

            font1 = {'family':'Times New Roman','style':'normal','size':18, }
            legend1 = axs[idx].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs[idx].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
            axs[idx].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
            axs[idx].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
            axs[idx].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

            axs[idx].tick_params(direction='in',axis='both',top=True,right=True,labelsize=20, width=3 )
            labels = axs[idx].get_xticklabels() + axs[idx].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(18) for label in labels] #刻度值字号

            font = {'family':'Times New Roman','style':'normal','size':20, }
            axs[idx].set_xlabel("Communication Round", fontproperties = font)
            axs[idx].set_ylabel(key, fontproperties = font,) #  fontdict = font1
        # fontt = {'family':'Times New Roman','style':'normal','size':16}
        # if self.title != '':
            # plt.suptitle(self.title, fontproperties = fontt, )
        out_fig = plt.gcf()

        # out_fig.savefig( os.path.join(savepath, f"{self.basename}.pdf") )
        out_fig.savefig(os.path.join(savepath, f"{savename}.eps") )
        out_fig.savefig(os.path.join(savepath, f"{savename}.pdf") )
        # plt.show()
        plt.close()
        return

    ##统计 每一轮通信过程，各个客户端所有参数数组(融合为一个数组)的均值、方差、L1范数的和的均值，为了验证各轮之间的差异性越来越小
    def Client_mean_var_L12_avg(self, num_clients, stastic = ["Mean", "1-norm", "2-norm", "Variance"], savepath = '', savename = ''):
        X = self.data[:, 0]
        width = 4*4
        high = 3
        fig, axs = plt.subplots(1, 4, figsize=(width, high), constrained_layout = True, sharex=True) # constrained_layout=True
        # cnt = 0
        L = len(stastic)
        ## Mean
        for idx, key in enumerate(stastic):
            cols = [1 + idx + L*j for j in range(num_clients)]
            tmp = self.data[:, cols]
            avg = np.mean(tmp, axis=1)
            axs[idx].plot(X, avg, color = color[idx], label = key, linewidth=2,)

            font1 = {'family':'Times New Roman','style':'normal','size':18, }
            legend1 = axs[idx].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs[idx].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
            axs[idx].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
            axs[idx].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
            axs[idx].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

            axs[idx].tick_params(direction='in',axis='both',top=True,right=True,labelsize=20, width=3 )
            labels = axs[idx].get_xticklabels() + axs[idx].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(18) for label in labels] #刻度值字号

            font = {'family':'Times New Roman','style':'normal','size':20, }
            axs[idx].set_xlabel("Communication Round", fontproperties = font)
            axs[idx].set_ylabel(key, fontproperties = font,) #  fontdict = font1
        # fontt = {'family':'Times New Roman','style':'normal','size':16}
        # if self.title != '':
            # plt.suptitle(self.title, fontproperties = fontt, )
        out_fig = plt.gcf()

        # out_fig.savefig( os.path.join(savepath, f"{self.basename}.pdf") )
        out_fig.savefig(os.path.join(savepath, f"{savename}.eps") )
        out_fig.savefig(os.path.join(savepath, f"{savename}.pdf") )
        # plt.show()
        plt.close()
        return
#============================================================================================================================
#                                        记录当前每个epcoh的 相关指标, 但是不记录历史
#============================================================================================================================

class Accumulator:
    """For accumulating sums over n variables."""
    def __init__(self,  n):
        self.data = [0.0] * n
        return

    def add(self, *Args):
        self.data = [a + float(b) for a, b in zip(self.data, Args)]
        return

    def reset(self):
        self.data = [0.0] * len(self.data)
        return

    def __getitem__(self, idx):
        return self.data[idx]


#============================================================================================================================
#                                                训练时 统计 PSNR和MSE
#============================================================================================================================

class TraRecorder(object):
    def __init__(self,  Len = 3,  name = "Train", compr = '', tra_snr = 'noiseless'):
        self.name =  name
        self.len = Len
        self.metricLog = np.empty((0, self.len))
        self.cn = self.__class__.__name__
        if compr != '' :
            self.title = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}} = {}\mathrm{{(dB)}}$'.format(compr, tra_snr)
            self.basename = f"{self.cn}_compr={compr:.1f}_trainSnr={tra_snr}(dB)"
        else:
            self.title = ""
            self.basename = f"{self.cn}"
        return

    def reset(self):
        self.metricLog = np.empty((0, self.len))
        return

    def addlog(self, epoch):
        # self.metricLog.shape= (n, 4) # 每列分为别
        # self.metricLog = np.vstack( ( self.metricLog, np.zeros( (1, len(self.args.metrics)) ) ) )
        self.metricLog = np.append(self.metricLog , np.zeros( (1, self.len )), axis=0)
        self.metricLog[-1, 0] = epoch
        return

    def assign(self,  metrics = ''):
        if len(metrics) != self.len - 1:
            print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")
            raise ValueError("len is inconsistent")
        self.metricLog[-1, 1:] = metrics
        return

    def __getitem__(self, idx):
        return self.metricLog[-1, idx + 1]

    def save(self, path, prefix = None):
        if prefix == None:
            torch.save(self.metricLog, os.path.join(path, f"{self.basename}.pt"))
        else:
            torch.save(self.metricLog, os.path.join(path, f"{prefix}.pt"))
        return

    def plot_inonefig(self, savepath, metric_str = ['loss','batimg_PSNR', 'imgae_PSNR','acc',], ):
        if len(metric_str) != self.len - 1:
            print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")
            raise ValueError("len is inconsistent")

        X = self.metricLog[:, 0]
        cols = 2;        rows =  math.ceil(len(metric_str) / cols)
        width = 4*cols;  high = 3*rows

        fig, axs = plt.subplots(rows, cols, figsize = (width, high), constrained_layout=True, sharex = True) # constrained_layout=True
        # cnt = 0
        for idx, met in  enumerate(metric_str):
            i = idx // cols
            j = idx % cols

            axs[i, j].plot(X, self.metricLog[:, idx + 1], color = color[idx], linestyle = '-',  label = met,) # marker = 'd', markersize = 12,

            # font = FontProperties(fname=fontpath1 + "Times_New_Roman.ttf", size = 20)
            font = {'family':'Times New Roman','style':'normal','size':12, }
            axs[i, j].set_xlabel("Communication Round", fontproperties=font)

            if "psnr" in met.lower():
                axs[i, j].set_ylabel(f"{met} (dB)", fontproperties = font,) #  fontdict = font1
            else:
                axs[i, j].set_ylabel(f"{met}", fontproperties = font )# , fontdict = font1
            #plt.title(label, fontproperties=font)

            #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
            # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
            font1 = {'family':'Times New Roman','style':'normal','size':12, }
            legend1 = axs[i, j].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs[i, j].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
            axs[i, j].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
            axs[i, j].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
            axs[i, j].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

            axs[i, j].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
            labels = axs[i, j].get_xticklabels() + axs[i, j].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(12) for label in labels] #刻度值字号

        fontt = {'family':'Times New Roman','style':'normal','size':16}
        if self.title != '':
            plt.suptitle(self.title, fontproperties = fontt, )
        out_fig = plt.gcf()

        # out_fig.savefig( os.path.join(savepath, f"{self.basename}.pdf") )
        out_fig.savefig(os.path.join(savepath, f"{self.basename}_Plot.eps") )
        # plt.show()
        plt.close()
        return




























