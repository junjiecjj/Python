#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:38:28 2024

@author: jack

exp^{1j * 2pi * k * n/N}
画出DFT整数频率的时频曲线

"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import scipy
# from scipy.fftpack import fft,ifft,fftshift,fftfreq

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300      # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22

mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', '$\\clubsuit$', '$\\bowtie$', '$\\boxdot$', '$\\boxtimes$', '$\\boxminus$',  '$\\boxplus$', '$\\bigodot$','$\\bigoplus$', '$\\bigotimes$', '$\\bigstar$', '$\\blacksquare$', '$\circledR$', '$\circledS$',  '$\\bigtriangledown$','$\\bigtriangleup$','$\\blacktriangleleft$', '$\\blacktriangleright$', '$\\diamondsuit',   '$\heartsuit$', '$666$', '$\\varrho$', '$\\Omega$', '$\\nabla$', '$\\Upsilon$', '$\\mho$', '$\\Phi$',  '$\\divideontimes$', '$\\otimes$', '$\\odot$', '$\\ominus$', '$\\oplus$', '$\\oslash$', '$\\otimes$']

color = ['#800080','#FF6347','#00FF00','#0000FF','#4ea142','#FF00FF','#FFA500','#1E90FF','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE','#808000']

#===================================================================================
# ============================  变量参数定义
#===================================================================================
basicunit = 1                                          # 1Hz, 1kHz, 1MHz
f0 = 15 * basicunit                                    # 最低子载波频率(基带)
M  = 64                                                # 子载波数
subcarr_interval = f0                                  # 子载波频率间隔
f_list = np.arange(f0, f0 * (1 + M), f0)               # 子载波频率序列

Tsym = 1/f0                      # OFDM符号长度
fs = 2000 * basicunit            # 信号采样频率,Hz; 要求 f0 * num_subcar *2 <= fs
Ts = 1/fs                        # 采样时间间隔
t = np.arange(0, Tsym, Ts)       # 定义信号采样的时间点 t.
N = t.size                       # 采样信号的长度, N为偶数

#======================================================================================
# =============================   生成子载波信号，并进行频谱分析
#======================================================================================

X = np.zeros( (M, N), dtype = complex)     # X 是一个大小为 M × length(t) 的矩阵，用于存储 M 个子载波的信号
FFTN = 3000
hatX = np.zeros((M, FFTN), dtype = complex)  # 对每个子载波的信号进行快速傅里叶变换（FFT）得到其频域表示，并将结果存储在矩阵 hatX 中
for i in range(M):
    X[i]     = np.exp(1j * 2 * np.pi * f_list[i] * t)
    hatX[i]  = np.fft.fft(X[i,:], n = FFTN) / N  # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
    hatX[i]  = scipy.fftpack.fftshift(hatX[i] )
# 画频谱图时的频率刻度,N为偶数
NN = hatX.shape[1]
df = fs/NN
f = np.arange(-int(NN/2), int(NN/2))*df

#%%%================================= 开始画图 ======================================
width = 10
high = 6
horvizen = 1
vertical = 1
fig, axs = plt.subplots(vertical, horvizen, figsize = (horvizen*width, vertical*high), constrained_layout = True)
labelsize = 20
## 时域波形图
#===================================================================================
Numscr = 4                          # 绘制的子载波数量
for i in range(Numscr):
    axs.plot(t, np.real(X[i]), color = color[i], linestyle = '-', label = f"{f_list[i]} Hz", lw = 2)

axs.set_xlabel(r'时间(s)', )
axs.set_ylabel(r'实部', )

legend1 = axs.legend(loc='best', borderaxespad=0,  edgecolor='black',  )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#================ super ====================
out_fig = plt.gcf()
# out_fig.savefig('sin.png', dpi=1000,)
plt.show()
plt.close()


#%% 频域波形图
#===================================================================================
width = 10
high = 6
horvizen = 1
vertical = 1
fig, axs = plt.subplots(vertical, horvizen, figsize = (horvizen*width, vertical*high), constrained_layout = True)
labelsize = 20

Numscr = 4                          # 绘制的子载波数量
for i in range(Numscr):
    axs.plot(f, np.real(hatX[i]), color = color[i], linestyle = '-', label = f"{f_list[i]} Hz",)

axs.set_xlabel(r'频率/Hz', )
axs.set_ylabel(r'幅度', )

legend1 = axs.legend(loc='best', borderaxespad=0,  edgecolor='black', )

K = 7
axs.set_xlim(f0- K * f0, f0 * (K + Numscr))  #拉开坐标轴范围显示投影
#================ super ====================
out_fig = plt.gcf()
# out_fig.savefig('freq.png', dpi=1000,)
plt.show()
plt.close()


































