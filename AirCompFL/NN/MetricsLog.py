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

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator



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
#                                                训练时
#============================================================================================================================
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
        self.metricLog = np.append(self.metricLog , np.zeros( (1, self.len + 1)), axis=0)
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
        if prefix == None:
            torch.save(self.metricLog, os.path.join(path, f"{self.basename}.pt"))
        else:
            torch.save(self.metricLog, os.path.join(path, f"{prefix}.pt"))
        return


























