#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:35:34 2024
reproduction of Paper: Intelligent Reﬂecting Surface Enhanced Wireless Network via Joint Active and Passive Beamforming

@author: Junjie Chen
Note: 在通信优化问题的仿真中，信道建模尤其关键，如果信道选择不好会极大地影响优化的性能，可能调试很久都没有出现较为理想的曲线都是因为信道模型选取的问题，

https://github.com/ken0225/RIS-Codes-Collection
https://zhuanlan.zhihu.com/p/582128377

https://blog.csdn.net/liujun19930425/article/details/127862357
"""

import sys
import numpy as np
import scipy
import cvxpy as cpy
import matplotlib.pyplot as plt
import math
import matplotlib
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
from matplotlib.pyplot import MultipleLocator
# import scipy.constants as CONSTANTS


sys.path.append("../")
from Solver import SDRsolver, AU_MRT, AI_MRT, AlternatingOptim
from Utility import set_random_seed, set_printoption

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


#%%
epsilon = 1e-4
d0 = 100
dv = 1                 # user到AP-RIS垂直距离
D0 = 1.0
C0 = -30               # dB
C0 = 10**(C0/10.0)     # 参考距离的路损
sigmaK2 = -80          # dB
sigmaK2 = 10**(sigmaK2/10.0)  # 噪声功率
gamma = 10             # dB
gamma = 10**(gamma/10.0)    #  信干噪比约束10dB
M = 1     # AP天线数量
# N = 30    # RIS天线数量
L = 1000  # Gaussian随机化次数
frame = 500

#%% 路损参数
alpha_AI = 3.2      #  AP 和 IRS 之间path loss exponent
alpha_Iu = 2      # IRS 和 User 之间path loss exponent

set_random_seed(10000)
set_printoption(10)

N = np.arange(0, 1610, 100) # user到AP-RIS连线的垂线距离AP的距离
N[0] += 1
P = 5/1000

#%%
sigmaR2 = -80          # dB
sigmaR2 = 10**(sigmaR2/10.0)  # 噪声功率

d = d0
dIu = np.sqrt((d0-d)**2 + dv**2)
## g
AI_large_fading = C0 * ((d0/D0)**(-alpha_AI))
## h_r
Iu_large_fading = C0 * ((dIu/D0)**(-alpha_Iu))

RIS = np.zeros(len(N))
FDAF = np.zeros(len(N))

for i, n in enumerate(N):
    for j in range(frame):
        g = np.sqrt(AI_large_fading) * np.sqrt(1 / (2 * sigmaK2)) * ( np.random.randn(n, M) + 1j * np.random.randn(n, M) )
        hr = np.sqrt(Iu_large_fading) * np.sqrt(1 / 2) * ( np.ones((1,n)) )
        Pu = P * (np.sum(np.array([np.abs(i) for i in hr.flatten()]) * np.array([np.abs(i) for i in g.flatten()])))**2
        RIS[i] += Pu
RIS = RIS/frame
RIS = np.log2(1 + RIS)

#%% 画图
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
axs.plot(N, RIS, color = 'k', linestyle='-',  marker = "o",  markersize = 20, label = 'RIS',  )
# axs.plot(N, SDR, color='b', linestyle='-',  lw = 3, marker = "d", markersize = 12, label = 'SDR',  )



# font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs.set_xlabel( "N", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
axs.set_ylabel('rate', fontproperties=font2, )
# axs.set_title(f'd = {d}(m)', fontproperties=font2)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 20}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

x_major_locator = MultipleLocator(200)               #把x轴的刻度间隔设置为1，并存在变量里
axs.xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs.tick_params(direction='in', axis='both', top=True, right=True,labelsize=16, width=3,)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
axs.spines['bottom'].set_linewidth(1.5)    ###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(1.5)      ####设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(1.5)     ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(1.5)       ####设置上部坐标轴的粗细

out_fig = plt.gcf()
# out_fig.savefig('fig5.eps' )
plt.show()














































