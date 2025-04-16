#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:12:47 2025

@author: jack
"""

import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

## my lib
from Xcorrs import xcorr, correlate_maxlag, correlate_template, get_lags

filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12



#%% 2.3 互相关函数在信号处理中的应用

fs = 2000     # 采样频率
t = np.arange(0, 1, 1/fs)  #时间向量
c = 1e4       # 光速，假设信号以光速传播
d = 2000      # 目标实际距离（米）
delay = 2*d/c # 实际延迟时间（秒）


X = scipy.signal.chirp(t, 0, 1, 100);
Y = np.hstack((np.zeros(int(np.round(delay*fs))), X[:-int(np.round(delay*fs))])) + 0.5 * np.random.randn(X.size)

##>>>>>>  method 1
acf, lag = xcorr(Y, X, normed = True, detrend = True, maxlags = Y.size - 1)

time_delay = lag[np.argmax(np.abs(acf))]/fs
estimated_distance = time_delay * c / 2;

print(f'实际的目标距离为 {d} 米\n', )
print(f'估计的目标距离为 {estimated_distance} 米\n', )

##### plot
fig, axs = plt.subplots(3, 1, figsize = (12, 8), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, X, label = 'x')
axs[0].set_title("发射信号 X(t)")
axs[0].set_xlabel('时间 (秒)',)
axs[0].set_ylabel('幅度',)

axs[1].plot(t, Y[0:len(t)], label = 'PSD',  )
axs[1].set_title("接收信号 Y(t)")
axs[1].set_xlabel('时间 (秒)')
axs[1].set_ylabel('幅度')

axs[2].plot(lag/fs, acf, label = 'PSD',  )
axs[2].set_title(r"互相关函数 $R_{XY}(\tau)$")
axs[2].set_xlabel('时间延迟 (秒)')
axs[2].set_ylabel('互相关值')

plt.show()
plt.close()

















































































