#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 11:09:23 2025

@author: jack
% https://blog.csdn.net/qq_35844208/article/details/128547667



"""


import numpy as np
import matplotlib.pyplot as plt
import scipy

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18

# %% Radar parameters
c = 3e8                    # speed of light
BW = 150e6;                 # bandwidth 有效
f0 = 77e9;                 # carrier frequency
numADC = 256;              # of adc samples
numChirps = 256;           # of chirps per frame
numCPI = 10;
T = 10e-6;                   # PRI，默认不存在空闲时间
PRF = 1/T;
Fs = numADC/T;               # sampling frequency
dt = 1/Fs;                   # sampling interval
slope = BW/T;
lamba = c/f0;
N = numChirps*numADC*numCPI                       #  total of adc samples
t = np.linspace(0, T*numChirps*numCPI, N)         #  time axis, one frame 等间隔时间/点数
t_onePulse = np.arange(0, dt*numADC, dt)          #  单chirp时间
numTX = 1
numRX = 8                             # 等效后
Vmax = lamba/(T*4)                    # Max Unamb velocity m/s
DFmax = 1/2*PRF                       # = Vmax/(c/f0/2); % Max Unamb Dopp Freq
dR = c/(2*BW)                         # range resol
Rmax = Fs*c/(2*slope)                 # TI's MIMO Radar doc
Rmax2 = c/2/PRF                       # lecture 2.3
dV = lamba/(2*numChirps*T)            # velocity resol, lambda/(2*framePeriod)
d_rx = lamba/2                        # dist. between rxs
d_tx = 4*d_rx                         # dist. between txs
N_Dopp = numChirps                    # length of doppler FFT
N_range = numADC                      # length of range FFT
N_azimuth = numTX*numRX
R = np.arange(0, Rmax, dR)                   # range axis
V = np.linspace(-Vmax, Vmax, numChirps)      # Velocity axis
ang_ax = np.arange(-90, 91)                  # angle axis

# %%目标参数
r1_radial = 50
v1_radial = 10     #  velocity 1
tar1_angle = -10
r1_x = np.sin(tar1_angle * np.pi/180)*r1_radial
r1_y = np.cos(tar1_angle * np.pi/180)*r1_radial
v1_x = np.sin(tar1_angle * np.pi/180)*v1_radial
v1_y = np.cos(tar1_angle * np.pi/180)*v1_radial
r1 = [r1_x, r1_y, 0]

r2_radial = 100
v2_radial = -15    #  velocity 2
tar2_angle = 10
r2_x = np.sin(tar2_angle * np.pi/180)*r2_radial
r2_y = np.cos(tar2_angle * np.pi/180)*r2_radial
v2_x = np.sin(tar2_angle * np.pi/180)*v2_radial
v2_y = np.cos(tar2_angle * np.pi/180)*v2_radial
r2 = [r2_x, r2_y, 0]


































