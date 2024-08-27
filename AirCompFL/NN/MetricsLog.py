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
# matplotlib.use('Agg')
# 设置backend为pgf
# matplotlib.use('pdf')
import pickle
import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# from matplotlib.pyplot import MultipleLocator
import warnings
warnings.filterwarnings("ignore")


# 本项目自己编写的库
# sys.path.append("../")
# from Option import args
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
plt.rc('font', family='Times New Roman')

mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
colors = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]
# colors = ['#000000','#0000FF', '#DC143C', '#006400', '#9400D3', '#ADFF2F', '#FF00FF', '#00CED1' ,'#FFA500', ]

#===============================================================================================================
#                                                训练时
#===============================================================================================================
class TraRecorder(object):
    def __init__(self,  Len = 3,  name = "Train",  tra_snr = 'noiseless'):
        self.name =  name
        self.len = Len
        self.metricLog = np.empty((0, self.len + 1))
        self.cn = self.__class__.__name__

        self.title = ""
        self.basename = f"{self.cn}"
        return

    def reset(self):
        self.metricLog = np.empty((0, self.len + 1))
        return

    def addlog(self, epoch):
        self.metricLog = np.append(self.metricLog, np.zeros( (1, self.len + 1)), axis=0)
        self.metricLog[-1, 0] = epoch
        return

    def assign(self,  metrics = ''):
        if len(metrics) != self.len:
            print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")
            raise ValueError("len is inconsistent")
        self.metricLog[-1, 1:] = metrics
        return

    def __getitem__(self, idx):
        return self.metricLog[-1, idx + 1]

    def save(self, path, args, prefix = None):
        ## 1
        # if prefix == None:
        #     torch.save(self.metricLog, os.path.join(path, f"{self.basename}.pt"))
        # else:
        #     torch.save(self.metricLog, os.path.join(path, f"{prefix}.pt"))

        ## 3
        np.save(os.path.join(path, f"{self.basename}.npy"), self.metricLog)

        ## 3
        # with open(os.path.join(path, f"{self.basename}.pickle"), 'wb') as f:
        #     pickle.dump(self.metricLog, f)
        self.plot(path, args)
        return
    def plot(self, path, args):
        self.metricLog[ self.metricLog > 1e20] = 1e20

        label = 'Optimality gap'
        fig, ax = plt.subplots(1, 1, figsize = (8, 6))
        # colors = plt.cm.cool(np.linspace(0, 1, len(results)))
        ax.semilogy(self.metricLog[:,0], self.metricLog[:,1], color = colors[0], lw = 3, label = label)

        font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
        ax.set_xlabel("Communication Round", fontdict = font, labelpad = 2)
        ax.set_ylabel("Optimality gap", fontdict = font, labelpad = 2)
        title = f"{args.case}, E = {args.local_up}, {args.channel}, SNR = {args.SNR}"
        ax.set_title(title, fontproperties = font, )

        font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,}
        legend1 = ax.legend(loc='best', prop =  font )
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        ax.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

        ax.tick_params(which = 'major', axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)

        ax.tick_params(which = 'minor', axis='x', direction='in', top=True,  width=2, length = 2,  )
        ax.tick_params(which = 'minor', axis='y', direction='in', color = 'red',  width=2, length = 2,  )
        ax.grid(color = 'black', alpha = 0.3, linestyle = (0, (5, 10)), linewidth = 1.5 )
        out_fig = plt.gcf()
        out_fig.savefig(os.path.join(path, "OptimalityGap.eps"),  )
        # plt.show()
        plt.close()

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        label = 'lr'
        fig, ax = plt.subplots(1, 1, figsize = (8, 6))
        # colors = plt.cm.cool(np.linspace(0, 1, len(results)))
        ax.plot(self.metricLog[:,0], self.metricLog[:,2], color = colors[0], lw = 3, label = label)

        font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
        ax.set_xlabel("Communication Round", fontdict = font, labelpad = 2)
        ax.set_ylabel("Learning rate", fontdict = font, labelpad = 2)
        title = f"{args.case}, E = {args.local_up}, {args.channel}, SNR = {args.SNR}"
        ax.set_title(title, fontproperties = font, )

        font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,}
        legend1 = ax.legend(loc='best', prop =  font )
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        ax.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

        ax.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)

        ax.grid(color = 'black', alpha = 0.3, linestyle = (0, (5, 10)), linewidth = 1.5 )

        out_fig = plt.gcf()
        out_fig.savefig(os.path.join(path, "Lr.eps"), )
        # plt.show()
        plt.close()
        return




#=====================================================================================================================
#                                        记录当前每个epcoh的 相关指标, 但是不记录历史
#=====================================================================================================================


class Accumulator(object):
    """ For accumulating sums over n variables. """
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


















