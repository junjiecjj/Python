#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:15:28 2025

@author: jack
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
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12


# https://www.cnblogs.com/gjblog/p/13494103.html#
# https://blog.csdn.net/weixin_42553916/article/details/122225988
#%% 频率调制（FM）是一种广泛应用于广播和通信系统的调制方式。其基本概念是通过改变信号的频率来传递信息。
# 相干解调
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
fs = 500  # 采样频率
dt = 1/fs
T  = 3
t  = np.arange(0, T, dt)  # 时间向量
fc = 40    # 载波频率
Ac = 1     # 载波幅度
kf = 20    # 频率偏移常数,频偏常数, 表示调频器的调频灵敏度. 这个参数相当重要，直接决定解调的效果，需要学习一下确定这个参数的方法
Am = 1     # 调制信号幅度
fm = 3     # 调制信号频率

# 调制信号（假设为正弦波）
mt = Am * np.cos(2 * np.pi * fm * t)
# 频率调制
ct = Ac * np.cos(2 * np.pi * fc * t)  # 载波信号
x = Ac * np.cos(2 * np.pi * fc * t +   kf * Am * np.sin(2 * np.pi * fm * t) / (fm) ) # + 0.01 * np.random.randn(t.size)

## PLL 锁相环相干解调, 相干解调法(同步解调),用锁相环同步
L = t.size
vco_phase = np.zeros(L) # 初始化vco相位
rt = np.zeros(L)        # 初始化压控振荡器vco输出
et = np.zeros(L)        # 初始化乘法鉴相器pd输出
vt = np.zeros(L)        # 初始化环路滤波器lf输出
Av = 1                  # vco输出幅度
kv = 100                # vco频率灵敏度
km = 1                  # 鉴相器增益pd增益
k0 = 1                  # lf增益
rt[0] = Av * np.cos(vco_phase[0])  #
et[0] = km * x[0] * rt [0]         #

# [Bb, Ba] = scipy.signal.butter(1, 2 * 4 * fm / fs, 'low')
# b0 = Bb[0]
# b1 = Bb[1]
# a1 = Ba[1]
## b0 = 0.07295965726826667;       # Fs = 40000, fcut = 1000的1阶巴特沃斯低通滤波器系数, 由FDA生成
## b1 = 0.07295965726826667;
## a1 = -0.8540806854634666;
#================================ IIR -巴特沃兹低通滤波器  =====================================
lf = 10     # 通带截止频率200Hz
Fc = 200    # 阻带截止频率1000Hz
Rp = 1      # 通带波纹最大衰减为1dB
Rs = 40     # 阻带衰减为40dB
#-----------------------计算最小滤波器阶数-----------------------------
na = np.sqrt(10**(0.1 * Rp) - 1)
ea = np.sqrt(10**(0.1 * Rs) - 1)
order = np.ceil(np.log10(ea/na)/np.log10(Fc/lf))  #巴特沃兹阶数
Wn = lf * 2 / fs
#------- 低通滤波 ---------
[Bb, Ba] = scipy.signal.butter(order, Wn, 'low')
vt = scipy.signal.filtfilt(Bb, Ba, et)

# vt[0] = k0 * b0 * et[0]
for i in range(1, L):
    vco_phase_change = 2 * np.pi * fc * dt + 2 * np.pi * kv * vt[i-1] * dt
    vco_phase[i] = vco_phase[i-1] + vco_phase_change

    rt[i] = Av * np.cos(vco_phase[i]) # vco输出（会跟踪st的相位）
    et[i] = km * rt[i] * x[i]         # 乘法鉴相器输出，式(16)

    # vt = scipy.signal.lfilter(Bb, Ba, et) * 2 # 进行滤波
    vt = scipy.signal.filtfilt(Bb, Ba, et)

    # vt[i] = k0 * (b0 * et[i] + b1 * et[i-1] - a1 * vt[i-1])

# 绘制调制信号和FM信号
fig, axs = plt.subplots(5, 1, figsize = (8, 10), constrained_layout = True)

axs[0].plot(t, mt, color = 'b', lw = 2, label = '原始波形 (时域)')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("原始波形 (时域)")
axs[0].legend()

axs[1].plot(t, ct, color = 'b', lw = 0.5, label = '载波信号')
axs[1].set_xlabel('时间 (s)',)
axs[1].set_ylabel('幅度',)
axs[1].set_title("载波信号 (时域)")
axs[1].legend()

axs[2].plot(t, x, color = 'b', lw = 0.2, label = '已调信号 (AM, 时域)')
axs[2].set_xlabel('时间 (s)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("已调信号 (幅度调制AM, 时域)")
axs[2].legend()

axs[3].plot(t, rt, color = 'b', lw = 0.2, label = 'vco信号 (时域)')
axs[3].set_xlabel('时间 (s)',)
axs[3].set_ylabel('幅度',)
axs[3].set_title("vco信号 (时域)")
axs[3].legend()

axs[4].plot(t, vt * 2, color = 'b', label = '解调信号(时域)')
axs[4].plot(t, mt, color = 'r', label = '原始信号 (时域)')
axs[4].set_xlabel('时间 (s)',)
axs[4].set_ylabel('幅度',)
axs[4].set_title("解调信号(时域)")
axs[4].legend()

plt.show()
plt.close()
























































