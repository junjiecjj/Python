#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 13:25:00 2022

@author: jack
"""
import scipy.stats as st
import scipy.stats as stats
import matplotlib
# matplotlib.get_backend()
# matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'

font = FontProperties(
    fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)


fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)


fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
fonte1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=24)


fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
fonttX = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
fonttY = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
fonttitle = {'style': 'normal', 'size': 17}
fontt2 = {'style': 'normal', 'size': 19, 'weight': 'bold'}
fontt3 = {'style': 'normal', 'size': 16, }

# ===========================================================================
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

u = 0
sigma = np.sqrt(2)
plt.rcParams['font.sans-serif'] = ['SimHei']

x = np.linspace(-10, 10, 1000000)  # 设定分割区间
y1 = stats.norm.pdf(x, u, sigma)
axs[0].plot(x, y1, color='r', linestyle='-', label='real pdf',)


y2 = np.exp(- ((x - u)**2/(2.0 * sigma**2))) *  1.0/(sigma * np.sqrt(2.0 * np.pi))
axs[0].plot(x, y2, color='b', linestyle=':', label='sim pdf',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[0].set_xlabel(r'x', fontproperties=font3)
axs[0].set_ylabel(r'PDF', fontproperties=font3)
axs[0].set_title('PDF', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[0].legend(loc='best', borderaxespad=0,
                        edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[0].tick_params(direction='in', axis='both', labelsize=16, width=3,)
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号
# ==============================================================
# 绘制正态分布CDF
y3 = stats.norm.cdf(x, u, sigma**2)
axs[1].plot(x, y3,  color='r', linestyle='-', label='real cdf',)


y4 = 1.0/(1.0 + np.exp(-2.0*x/sigma**2))
axs[1].plot(x, y4,  color='b', linestyle=':', label='sim cdf',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1].set_xlabel(r'x', fontproperties=font3)
axs[1].set_ylabel(r'CDF', fontproperties=font3)
axs[1].set_title('CDF', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[1].legend(loc='best', borderaxespad=0,
                        edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[1].tick_params(direction='in', axis='both', labelsize=16, width=3,)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号
# ===========================================================================
fig.subplots_adjust(hspace=0.6)  # 调节两个子图间的距离

fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
plt.suptitle('PDF and CDF', fontproperties=fontt)
plt.tight_layout()
plt.show()
# ===========================================================================
