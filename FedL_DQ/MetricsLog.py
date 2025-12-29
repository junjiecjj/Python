#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023/04/25
@author: Junjie Chen

"""

# 系统库

import os, sys

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


# 本项目自己编写的库
# sys.path.append("../")
# from Option import args
# fontpath = "/usr/share/fonts/truetype/windows/"
# fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
# fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
# plt.rc('font', family='Times New Roman')

mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
colors = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]
# colors = ['#000000','#0000FF', '#DC143C', '#006400', '#9400D3', '#ADFF2F', '#FF00FF', '#00CED1' ,'#FFA500', ]

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



#===============================================================================================================
#                                                训练时
#===============================================================================================================
class TraRecorder(object):
    def __init__(self,  Len = 3,  name = "TraRecorder",  tra_snr = 'noiseless'):
        self.name =  name
        self.len = Len
        self.metricLog = np.empty((0, self.len + 1))
        self.cn = self.__class__.__name__

        self.title = ""
        self.basename = f"{name}"
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

    def save(self, path, prefix = None):
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
        # self.plot(path, args)
        return

    def plot(self, path, labels = ['train acc', 'train Loss', 'lr', 'bit width'], name = "TrainLog.eps"):

        i = 0
        label = labels[i]
        fig, ax = plt.subplots(2, 2, figsize = (10, 8))
        # colors = plt.cm.cool(np.linspace(0, 1, len(results)))
        ax[0, 0].plot(self.metricLog[:,0], self.metricLog[:,1], color = colors[0], lw = 3, )
        font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
        ax[0, 0].set_xlabel("Communication Round", fontdict = font, labelpad = 2)
        ax[0, 0].set_ylabel(label, fontdict = font, labelpad = 2)
        ax[0, 0].grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        i += 1
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        label = labels[i]
        ax[0, 1].plot(self.metricLog[:,0], self.metricLog[:,2], color = colors[0], lw = 3, label = label)
        font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
        ax[0, 1].set_xlabel("Communication Round", fontdict = font, labelpad = 2)
        ax[0, 1].set_ylabel(label, fontdict = font, labelpad = 2)
        ax[0, 1].grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        i += 1
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        label = labels[i]
        ax[1, 0].plot(self.metricLog[:,0], self.metricLog[:,3], color = colors[0], lw = 3, )
        font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
        ax[1, 0].set_xlabel("Communication Round", fontdict = font, labelpad = 2)
        ax[1, 0].set_ylabel(label, fontdict = font, labelpad = 2)
        ax[1, 0].grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        i += 1
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        label = labels[i]
        ax[1, 1].plot(self.metricLog[:,0], self.metricLog[:,4], color = colors[0], lw = 3, label = label)
        font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
        ax[1, 1].set_xlabel("Communication Round", fontdict = font, labelpad = 2)
        ax[1, 1].set_ylabel(label, fontdict = font, labelpad = 2)
        ax[1, 1].grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

        out_fig = plt.gcf()
        out_fig.savefig(os.path.join(path, name),  )
        # plt.show()
        plt.close()

        return





































