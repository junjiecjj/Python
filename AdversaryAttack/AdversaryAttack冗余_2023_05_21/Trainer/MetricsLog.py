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
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
#内存分析工具
from memory_profiler import profile
import objgraph


# 本项目自己编写的库
sys.path.append("../")
from Option import args
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


#============================================================================================================================
#                                                   统计 PSNR和MSE
#============================================================================================================================


class MetricsRecorder(object):
    def __init__(self, Args, Len, metricsname = "TrainPSNRMSE"):
        self.args = Args
        self.len = Len
        self.m_samples = 0
        self.m_batchs = 0
        self.metricLog = np.empty((0, self.len))
        return

    def reset(self):
        self.m_samples = 0
        self.m_batchs = 0
        self.metricLog = np.empty((0, self.len))
        return


    def addlog(self, ):
        # self.metricLog.shape= (n, 4) # 每列分为别
        # self.metricLog = np.vstack( ( self.metricLog, np.zeros( (1, len(self.args.metrics)) ) ) )
        self.metricLog = np.append(self.metricLog , np.zeros( (1, self.len )), axis=0)
        self.m_samples = 0
        self.m_batchs = 0
        return

    def add(self, metrics, samples):
        self.metricLog[-1] += metrics
        self.m_samples += samples
        self.m_batchs += 1
        return

    def avg(self, ):
        self.metricLog[-1, 0] /= self.m_batchs  # 以 batch 为单位计算的 psnr 的均值
        self.metricLog[-1, 1] /= self.m_batchs  # 以 batch内 每张图片单独计算的 psnr 的和的均值
        self.metricLog[-1, 2] /= self.m_samples # 以 每张图片单独计算的 psnr 的和的均值
        # self.metricLog[-1, 3] /= self.m_samples
        return

    def __getitem__(self, idx):
        return self.metricLog[-1, idx]

    def save(self, path, name):
        torch.save(self.metricLog, os.path.join(path, name))
        
        return

    """
    两张大图；每张图对应一个指标，PSNR或者MSE，下面以PSNR为例；

    """
    #@profile
    def plot(self, savepath, ):
        width = 6
        high = 5
        for idx, met in  enumerate(self.args.metrics):
            fig, axs = plt.subplots(1, 1, figsize=(width, high),  constrained_layout=True)# constrained_layout=True
            epoch = len(self.metricLog)
            X = np.linspace(1, epoch, epoch)

            label = f"Avg batch {met}"
            axs.plot(X, self.metricLog[:, idx], color = "r", linestyle = '-', marker = '*', markersize = 12, label = label,)

            label = f"Avg sample {met}"
            axs.plot(X, self.metricLog[:, idx + 1], color = "b", linestyle = '-', marker = 'd', markersize = 12, label = label,)

            label = f"Avg image {met}"
            axs.plot(X, self.metricLog[:, idx + 2], color = "#FF8C00", linestyle = '-', marker = 's', markersize = 12, label = label,)

            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
            #font1 = {'family':'Times New Roman','style':'normal','size':20, 'color':'blue',}
            
            axs.set_xlabel('Epoch',fontproperties=font)
            if met == "PSNR":
                axs.set_ylabel(f"{met}(dB)",fontproperties=font,) #  fontdict = font1
            else:
                axs.set_ylabel(f"{met}",fontproperties=font )# , fontdict = font1
            #plt.title(label, fontproperties=font)

            #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
            font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
            font1 = {'family':'Times New Roman','style':'normal','size':16, }
            legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
            axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
            axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
            axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

            axs.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
            labels = axs.get_xticklabels() + axs.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(20) for label in labels] #刻度值字号


            #调节两个子图间的距离
            # plt.subplots_adjust(top=0.92,bottom=0.1, left=0.1, right=0.97, wspace=0.4, hspace=0.2)
            # plt.tight_layout()#  使得图像的四周边缘空白最小化
            out_fig = plt.gcf()
            out_fig.savefig( os.path.join(savepath, f"Train_{met}_Plot.pdf") )
            out_fig.savefig( os.path.join(savepath, f"Train_{met}_Plot.eps") )
            plt.show()
            plt.close()
        return

# me = MetricsRecorder(args,3 )

# print(f"me.metricLog = \n{me.metricLog}\n")

# me.addlog()
# print(f"me.metricLog = \n{me.metricLog}\n")

# me.addlog()
# print(f"me.metricLog = \n{me.metricLog}\n")

# me.addlog()
# print(f"me.metricLog = \n{me.metricLog}\n")
# me.metricLog[-1, 0] = 121

# me.addlog()
# print(f"me.metricLog = \n{me.metricLog}\n")

# me.add([1,2,3 ], 2)
# me.add([2,9,0, ], 3)
# print(f"me.metricLog = \n{me.metricLog}\n")

# me.avg()
# print(f"me.metricLog = \n{me.metricLog}\n")

# me.addlog()
# print(f"me.metricLog = \n{me.metricLog}\n")

# me.add([12, 42,  44], 12)
# me.add([22, 29,  41], 13)
# me.avg()
# print(f"me.metricLog = \n{me.metricLog}\n")


# # torch.save(me.metricLog, "/home/jack/snap/metrics.pt")

# # mt = torch.load( "/home/jack/snap/metrics.pt")



#============================================================================================================================
#                                                   统计 正确率
#============================================================================================================================
class AccuracyRecorder(object,):
    def __init__(self, Len,  metricsname = "MSE loss_ Acc"):
        self.m_samples = 0
        self.metrics = [i.strip() for i in metricsname.split("_")]
        self.len = Len
        self.data = np.empty((0,  self.len))
        return

    def reset(self):
        self.data = np.empty((0,  self.len ))
        return

    def addlog(self, ):
        self.m_samples = 0
        self.data = np.append(self.data , np.zeros( (1, self.len) ), axis=0 )
        return

    def add(self, data, samples):
        self.m_samples += samples
        self.data[-1] += data
        return

    def avg(self, ):
        self.data[-1] /= self.m_samples
        return

    def __getitem__(self, idx):
        return self.data[-1, idx]

    def save(self, path, name):
        torch.save(self.data, os.path.join(path, name))
        return

    #@profile
    def plot(self, savepath, ):
        width = 6
        high = 5
        for idx, met in  enumerate(self.metrics):
            fig, axs = plt.subplots(1, 1, figsize=(width, high),  constrained_layout=True)# constrained_layout=True
            epoch = len(self.data)
            X = np.linspace(1, epoch, epoch)

            label = f"{met}"
            axs.plot(X, self.data[:, idx], 'r-', marker = '*', markersize = 12, label = label,)

            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
            axs.set_xlabel('Epoch',fontproperties=font)

            axs.set_ylabel(f"{met}",fontproperties=font)

            #plt.title(label, fontproperties=font)

            #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
            font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
            font1 = {'family':'Times New Roman','style':'normal','size':16}
            legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
            axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
            axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
            axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

            axs.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
            labels = axs.get_xticklabels() + axs.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(20) for label in labels] #刻度值字号

            #调节两个子图间的距离
            # plt.subplots_adjust(top=0.92,bottom=0.1, left=0.1, right=0.97, wspace=0.4, hspace=0.2)
            # plt.tight_layout()#  使得图像的四周边缘空白最小化
            out_fig = plt.gcf()
            out_fig.savefig( os.path.join(savepath, f"Train_{met}_Plot.pdf") )
            out_fig.savefig( os.path.join(savepath, f"Train_{met}_Plot.eps") )
            # plt.show()
            plt.close()
        return

# acc = AccuracyRecorder(1, metricsname='MSE loss')
# print(f"acc.metrics = {acc.metrics}")

# for i in range(10):
#     acc.addlog()
#     for j in range(4):
#         acc.add(np.random.randn(1), 2)
#     acc.avg()
# acc.plot("/home/jack/snap/")

#============================================================================================================================
#                                                    记录当前每个epcoh的 相关指标, 但是不记录历史
#============================================================================================================================


class Accumulator:
    """For accumulating sums over n variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]













































