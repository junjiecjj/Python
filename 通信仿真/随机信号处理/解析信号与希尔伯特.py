#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 01:10:22 2025

@author: jack
实值信号的傅里叶变换是复对称的。这意味着负频率的内容相对于正频率是冗余的。在他们的工作中，Gabor[12]和Ville[13]，旨在通过去除傅立叶变换产生的冗余负频率内容来创建一个分析信号。解析信号是复值信号，但其频谱是单侧的（只有正频率），保留了原始实值信号的频谱内容。用解析信号代替原来的实值信号，已被证明是有用的

"""
import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

## my lib
from Xcorrs import xcorr, correlate_maxlag, correlate_template, get_lags

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



def analytic_signal(x):
    # Generate analytic signal using frequency domain approach
    x = x[:]
    N = x.size
    X = np.fft.fft(x, n = N)
    spectrum = np.hstack((X[0], 2*X[1:int(N/2)],X[int(N/2)+1], np.zeros(int(N/2)-1)))
    z = np.fft.ifft(spectrum, n = N)
    return z







































































































































