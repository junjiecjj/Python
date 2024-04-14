#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:38:28 2024

@author: jack


exp^{1j * 2pi * f_k * t}
画出实际频率的时频曲线

"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import scipy
# from scipy.fftpack import fft,ifft,fftshift,fftfreq

filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 14)

fontpath = "/usr/share/fonts/truetype/windows/"
font = FontProperties(fname=fontpath+"simsun.ttf", size = 22)

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size = 20)

mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', '$\\clubsuit$', '$\\bowtie$', '$\\boxdot$', '$\\boxtimes$', '$\\boxminus$',  '$\\boxplus$', '$\\bigodot$','$\\bigoplus$', '$\\bigotimes$', '$\\bigstar$', '$\\blacksquare$', '$\circledR$', '$\circledS$',  '$\\bigtriangledown$','$\\bigtriangleup$','$\\blacktriangleleft$', '$\\blacktriangleright$', '$\\diamondsuit',   '$\heartsuit$', '$666$', '$\\varrho$', '$\\Omega$', '$\\nabla$', '$\\Upsilon$', '$\\mho$', '$\\Phi$',  '$\\divideontimes$', '$\\otimes$', '$\\odot$', '$\\ominus$', '$\\oplus$', '$\\oslash$', '$\\otimes$']

color = ['#800080','#FF6347','#00FF00','#0000FF','#4ea142','#FF00FF','#FFA500','#1E90FF','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C',  '#7B68EE','#808000']

#===================================================================================
# ============================  变量参数定义
#===================================================================================
basicunit = 1000                                          # 1Hz, 1kHz, 1MHz
f0 = 15 * basicunit                                    # 最低子载波频率(基带)
M  = 64                                                # 子载波数
subcarr_interval = f0                                  # 子载波频率间隔
f_list = np.arange(f0, f0 * (1 + M), f0)               # 子载波频率序列

Tsym = 1/f0                      # OFDM符号长度
fs = f_list[-1] * 4            # 信号采样频率,Hz; 要求 f0 * num_subcar *2 <= fs
Ts = 1/fs                        # 采样时间间隔
t = np.arange(0, Tsym, Ts)       # 定义信号采样的时间点 t.
N = t.size                       # 采样信号的长度, N为偶数

#======================================================================================
# =============================   生成子载波信号，并进行频谱分析
#======================================================================================

X = np.zeros( (M, N), dtype = complex)     # X 是一个大小为 M × length(t) 的矩阵，用于存储 M 个子载波的信号
FFTN = N * 100                                # 扩展后的 FFT 数，为了提高频谱计算的分辨率
hatX = np.zeros((M, FFTN), dtype = complex)  # 对每个子载波的信号进行快速傅里叶变换（FFT）得到其频域表示，并将结果存储在矩阵 hatX 中
for i in range(M):
    X[i]     = np.exp(1j * 2 * np.pi * f_list[i] * t)
    hatX[i]  = np.fft.fft(X[i,:], n = FFTN) / N  # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
    hatX[i]  = scipy.fftpack.fftshift(hatX[i] )
# 画频谱图时的频率刻度,N为偶数
NN = hatX.shape[1]
df = fs/NN
f = np.arange(-int(NN/2), int(NN/2))*df

#%%%====================================== 开始画图 ===============================================
width = 10
high = 8
horvizen = 1
vertical = 1
fig, axs = plt.subplots(vertical, horvizen, figsize = (horvizen*width, vertical*high), constrained_layout = True)
labelsize = 20

#  时域波形图
#===================================================================================

Numscr = 4                          # 绘制的子载波数量
for i in range(Numscr):
    axs.plot(t, np.real(X[i]), color = color[i], linestyle = '-', label = f"{f_list[i]} Hz", lw = 2)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size = 20)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs.set_xlabel(r'时间(s)', fontproperties=font3)
axs.set_ylabel(r'实部', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 20}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 16)
legend1 = axs.legend(loc='best', borderaxespad=0,  prop=font2, bbox_to_anchor=(1, 1.2),  ncol = 4, facecolor = 'y', edgecolor = 'b',   )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = labelsize, width = 3,)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#================ super ====================
out_fig = plt.gcf()
out_fig.savefig('sin.png', dpi=1000,)
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
    axs.plot(f, np.real(hatX[i]), color = color[i], linestyle = '-', label = f"{f_list[i]} Hz", lw = 2)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs.set_xlabel(r'频率/Hz', fontproperties=font3)
axs.set_ylabel(r'幅度', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
legend1 = axs.legend(loc='best', borderaxespad=0,  edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize=labelsize, width=3,)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

K = 7
axs.set_xlim(f0- K * f0, f0 * (K + Numscr))  #拉开坐标轴范围显示投影
#================ super ====================
out_fig = plt.gcf()
out_fig.savefig('freq.png', dpi=1000,)
plt.show()
plt.close()
























