#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 01:30:11 2025

@author: jack
"""


# 系统库
import math
import os, sys
# import time, datetime
import numpy as np
# from scipy import stats
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]



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

    def save(self, path,   ):
        torch.save(self.metricLog, os.path.join(path, f"{self.basename}.pt"))
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

            font = {'family':'Times New Roman','style':'normal','size':12, }
            axs[i, j].set_xlabel("Epoch", fontproperties=font)

            if "psnr" in met.lower():
                axs[i, j].set_ylabel(f"{met} (dB)", fontproperties = font,) #  fontdict = font1
            else:
                axs[i, j].set_ylabel(f"{met}", fontproperties = font )# , fontdict = font1

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

        out_fig.savefig( os.path.join(savepath, f"{self.basename}_Plot.eps") )
        # plt.show()
        plt.close()
        return


#============================================================================================================================
#                                                测试时 统计 PSNR 和 MSE
#============================================================================================================================


class TesRecorder(object):
    def __init__(self,  Len = 2,  name = "Test"):
        self.name =  name
        self.len = Len
        self.TeMetricLog = {}
        self.cn = self.__class__.__name__
        return

    # 增加 某个压缩率和信噪比下训练的模型的测试结果条目
    def add_item(self,  tra_compr = 0.1, tra_snr = 1,):
        #tmpS = "TestMetricLog:Dataset={},CompRatio={}".format(dataset,comprateTmp)
        tmpS = "TestMetrics:Compr={:.1f},SNRtrain={}(dB)".format(  tra_compr, tra_snr)
        if tmpS not in self.TeMetricLog.keys():
            self.TeMetricLog[tmpS] = torch.Tensor()
        else:
            pass
        return

    def add_snr(self, tra_compr = 0.1, tra_snr = 1, test_snr = 1):
        tmpS = "TestMetrics:Compr={:.1f},SNRtrain={}(dB)".format(  tra_compr, tra_snr)
        # 第一列为snr, 后面各列为各个指标, 每一行第一列是测试snr, 其他列是在该 snr 下测试数据集的指标.
        self.TeMetricLog[tmpS] = torch.cat([self.TeMetricLog[tmpS], torch.zeros(1, self.len )], dim=0)
        self.TeMetricLog[tmpS][-1, 0] = test_snr
        return

    # assign 是直接赋值，而add_metric 和 avg 是联合起来赋值
    def assign(self,  tra_compr = 0.1, tra_snr = 1, met = ''):
        if len(met) != self.len - 1:
            print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")
            raise ValueError("len is inconsistent")
        tmpS = "TestMetrics:Compr={:.1f},SNRtrain={}(dB)".format(  tra_compr, tra_snr)
        self.TeMetricLog[tmpS][-1, 1:] = met
        return

    def __getitem__(self, pox):
        tra_compr, tra_snr, idx = pox
        tmpS = "TestMetrics:Compr={:.1f},SNRtrain={}(dB)".format(  tra_compr, tra_snr)
        return self.TeMetricLog[tmpS][-1, idx]

    def save(self, path, ):
        basename = f"{self.cn}_TeMetricLog"
        torch.save(self.TeMetricLog, os.path.join(path, basename + '.pt'))
        # self.plot(path, compr = compr, snr = snr)
        return

    def plot_inonefig1x2(self, savepath, metric_str = ['batimg_PSNR', 'imgae_PSNR'], tra_compr = 0.1, tra_snr = 1,):
        if len(metric_str) != self.len - 1:
            print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")
            raise ValueError("len is inconsistent")

        tmpS = "TestMetrics:Compr={:.1f},SNRtrain={}(dB)".format(  tra_compr, tra_snr)
        if tmpS not in self.TeMetricLog.keys():
            print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")
            raise ValueError(f"{tmpS} is nonexistent")

        data = self.TeMetricLog[tmpS]
        SNRlist = data[:, 0]

        width = 10
        high = 6
        fig, axs = plt.subplots(1, 2, figsize = (width, high), constrained_layout=True, sharex = True )  # constrained_layout = True  figsize=(width, high),

        ## accuaracy
        axs[0].plot(SNRlist, data[:, 1], color = color[0], linestyle = '-', marker = mark[0], markersize = 12, label = "accuracy",) # marker = 'd', markersize = 12,

        # font = {'family':'Times New Roman','style':'normal','size':20, }
        axs[0].set_xlabel(r"$\mathrm{SNR}_\mathrm{test}$ (dB)",)  # fontdict = font
        axs[0].set_ylabel("Accuracy",)# , fontdict = font1   fontproperties = font


        # font1 = {'family':'Times New Roman','style':'normal','size':16, }
        legend1 = axs[0].legend(loc='best', borderaxespad=0, edgecolor='black',) #  prop=font1,
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        ## PSNR
        for idx, met in enumerate(metric_str[1:]):
            axs[1].plot(SNRlist, data[:, idx + 2], color = color[idx + 1], linestyle = '-', marker = mark[idx + 1], markersize = 12, label = met,) # marker = 'd', markersize = 12,

        # font = {'family':'Times New Roman','style':'normal','size':20, }
        axs[0].set_xlabel(r"$\mathrm{SNR}_\mathrm{test}$ (dB)", ) # fontdict = font
        axs[1].set_ylabel("PSNR (dB)", ) # , fontdict = font1  fontproperties = font
        #plt.title(label, fontproperties=font)

        font1 = {'family':'Times New Roman','style':'normal','size':16, }
        legend1 = axs[1].legend(loc='best', borderaxespad = 0, edgecolor = 'black', prop = font1,)
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        ## public
        title = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(tra_compr, tra_snr)
        if title != '':
            # fontt  = {'family':'Times New Roman','style':'normal','size':22}
            plt.suptitle(title,) #  fontproperties = fontt,
        out_fig = plt.gcf()
        # out_fig.savefig( os.path.join(savepath, f"{self.cn}_Plot_{met}{basename}.pdf") )
        basename = f"compr={tra_compr:.1f}_trainSnr={tra_snr}(dB)"
        out_fig.savefig( os.path.join(savepath, f"{self.cn}_Plot_{basename}.eps") )
        # plt.show()
        plt.close()
        return


# ts = TesRecorder(3)
# ts.add_item(tra_compr=0.1, tra_snr=1)
# ts.add_snr(tra_compr=0.1, tra_snr=1,  test_snr = -2)
# ts.assign(tra_compr=0.1, tra_snr=1, met = torch.tensor([0.988, 0.121]))


# ts.add_snr(tra_compr=0.1, tra_snr=1,  test_snr = 1)
# ts.assign(tra_compr=0.1, tra_snr=1, met = torch.tensor([0.18, 0.532]))

# print(f"ts.TeMetricLog = {ts.TeMetricLog}")
# print(f"ts[0.1, 1, 1] = {ts[0.1, 1, 1]}")
































