#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 00:22:31 2025

@author: jack
"""

import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 20



#%% Program 3.1: Shannon limit.m: Dependency of spectral and power efﬁciencies for AWGN channel
k = np.arange(0.1, 15, 0.001)
EbN0 = (2**k-1)/k
EbN0dB = 10 * np.log10(EbN0)

##### plot
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

axs.semilogy(EbN0dB, k, color = 'b', label = 'capacity boundary')
axs.set_xlabel(r'$\mathrm{E_b}/\mathrm{N_0}$(dB)',)
axs.set_ylabel('Spectral Efficiency (Bit/s/Hz)',)
axs.set_title("Channel Capacity & Power efficiency limit")

axs.vlines(x = -1.59, ymin = 0.08 ,ymax = 13, color = 'red', ls = '--')

axs.text( 4, 8, "R > C\nUnattainable", color='red', size = 20, rotation = 0., ha = "center", va="center",)
axs.text(16, 0.3, "R < C\nPractical systems", color='g', size = 20, rotation = 0., ha = "center", va="center",)
axs.text( 16, 6, "R = C\ncapacity boundary", color='k', size = 20, rotation = 16., ha = "center", va="center",)
# axs.text( 1.1, 0.04, "Shannon limit\n-1.59dB", color='k', size = 20, rotation = 0., ha = "center",  va = "center", wrap = True)
axs.annotate("Shannon limit\n-1.59dB", xy = (0.155, 0.2) , xytext = (0.06,0.01) , size = 20, xycoords = 'figure fraction', textcoords='figure fraction', arrowprops=dict(facecolor='fuchsia',arrowstyle='<|-',connectionstyle='arc3'))
# axs.legend()
plt.show()
plt.close()




#%% Program 3.7: ergodic capacity limits.m: Simulating the ergodic capacity of a fading channel
snrdB = np.arange(-10, 30, 1/2)
h = (np.random.randn( 10000) + 1j * np.random.randn( 10000))/np.sqrt(2)
sigma_z = 1
snr = 10**(snrdB/10)
P = (sigma_z**2) * snr / np.mean(np.abs(h)**2)

C_agwn = 1





















































