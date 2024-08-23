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
# import scipy
import cvxpy as cp
import matplotlib.pyplot as plt
# import math
# import matplotlib
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
from matplotlib.pyplot import MultipleLocator
# import scipy.constants as CONSTANTS

from Tools import  ULA2UPA_Los, RIS2UserSteerVec
sys.path.append("../")
# from Solver import SOCPforW, SDPforV,
from Solver import AlternatingOptim, TwoStageAlgorithm
from Utility import set_random_seed, set_printoption

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

set_random_seed(10000)
# set_printoption()

#%%
pi = np.pi
epsilon = 1e-4
d0 = 51
D0 = 1.0

d1 = 20                         # AP半径
d2 = 3                          # RIS半径
C0 = -30                        # dB
C0 = 10**(C0/10.0)              # 参考距离的路损
sigmaK2 = -80                   # dBm
sigmaK2 = 10**(sigmaK2/10.0)/1000    # 噪声功率
gamma = 20                      # dB
gamma = 10**(gamma/10.0)        #  信干噪比约束20dB
M = 4                           # AP天线数量
Nx = 5
Ny = 6
N = Nx * Ny                     # RIS天线数量
L = 1000                        # Gaussian随机化次数
frame = 500

#%% 路损参数
alpha_AI = 2.8           # AP 和 IRS 之间path loss exponent
alpha_Iu = 2.8           # IRS 和 User 之间path loss exponent
alpha_Au = 3.5           # AP 和 User 之间path loss exponent
beta_AI = 10**(3/10)     # IRS到User考虑瑞利衰落信道，AP和IRS之间为纯LoS信道
beta_Iu = 10**(3/10)     # IRS到User考虑瑞利衰落信道，AP和IRS之间为纯LoS信道
beta_Au = 0              #  AP和 User 之间为Rayleigh信道

Uk = 4                   # 4个用户，图7仿真，假设U_k, k=1,2,3,4是活跃用户

#%% AP-User和RIS-User之间的距离和角度
d_Au = [d1, np.sqrt((d2*np.cos(pi/5))**2 + (d0 - d2*np.sin(pi/5))**2), d1, np.sqrt((d2*np.sin(pi/10))**2 + (d0 - d2*np.cos(pi/10))**2)]
theta_Au = [-pi/4, 2*pi - np.arctan(d2*np.cos(pi/5) / (d0 - d2*np.sin(pi/5))), pi/4, 2*pi - np.arctan(d2*np.sin(pi/10) / (d0 - d2*np.cos(pi/10)))]
d_Iu = [np.sqrt((d1*np.sin(pi/4))**2 + (d0-d1*np.cos(pi/4))**2), d2, np.sqrt((d1*np.sin(pi/4))**2 + (d0-d1*np.cos(pi/4))**2), d2]
theta_Iu = [pi + np.arctan(d1*np.sin(pi/4)/(d0 - d1*np.cos(pi/4))), 3*pi/2 - pi/5, pi - np.arctan(d1*np.sin(pi/4)/(d0 - d1*np.cos(pi/4))), pi+pi/10]


# #%% AP-User和RIS-User之间的距离和角度
# d_Au = [d1, np.sqrt((d2*np.cos(pi/5))**2 + (d0 - d2*np.sin(pi/5))**2), d1, np.sqrt((d2*np.sin(pi/10))**2 + (d0 - d2*np.cos(pi/10))**2)]
# theta_Au = [pi/4, np.arctan(d2*np.cos(pi/5) / (d0 - d2*np.sin(pi/5))), -pi/4, np.arctan(d2*np.sin(pi/10) / (d0 - d2*np.cos(pi/10)))]
# d_Iu = [np.sqrt((d1*np.sin(pi/4))**2 + (d0-d1*np.cos(pi/4))**2), d2, np.sqrt((d1*np.sin(pi/4))**2 + (d0-d1*np.cos(pi/4))**2), d2]
# theta_Iu = [-np.arctan(d1*np.sin(pi/4)/(d0 - d1*np.cos(pi/4))), -3*pi/10, np.arctan(d1*np.sin(pi/4)/(d0 - d1*np.cos(pi/4))), -pi/10]



#%% AP-RIS 信道G
AI_large_fading = C0 * ((d0/D0)**(-alpha_AI))
GLos = ULA2UPA_Los(M = M, Nx = Nx, Ny = Ny, azi_AP = 0, ele_AP = 0, azi_RIS = -np.pi, ele_RIS = 0)
GNLos = np.sqrt(1/2) * ( np.random.randn(N, M) + 1j * np.random.randn(N, M) )
G = np.sqrt(AI_large_fading) * (np.sqrt(beta_AI/(1+beta_AI)) * GLos + np.sqrt(1/(1+beta_AI)) * GNLos )

#%% AP/RIS-User信道Hr, Hd
Hr = np.zeros((N, Uk), dtype = complex)
Hd = np.zeros((M, Uk), dtype = complex)
for i in range(Uk):
    ## h_r
    Iu_large_fading = C0 * ((d_Iu[i]/D0)**(-alpha_Iu))
    ## h_d
    Au_large_fading = C0 * ((d_Au[i]/D0)**(-alpha_Au))

    IULos = RIS2UserSteerVec(Nx = Nx, Ny = Ny, azi = theta_Iu[i], ele = 0)
    IUNLos = np.sqrt(1/2) * ( np.random.randn(N, ) + 1j * np.random.randn(N, ) )
    # RIS-User
    Hr[:,i] = np.sqrt(Iu_large_fading/sigmaK2) * (np.sqrt(beta_Iu/(beta_Iu+1)) * IULos + np.sqrt(1/(beta_Iu+1)) * IUNLos)
    # AP-User
    AU_NLos = np.sqrt(1/2) * ( np.random.randn(M, ) + 1j * np.random.randn(M, ))
    Hd[:,i] = np.sqrt(Au_large_fading/sigmaK2) * AU_NLos

#%% Alternating Optim
iternum,  Pow, _, _ = AlternatingOptim(Hr, Hd, G, M, N, Uk, gamma, epsilon, L)

#%% TwoStage Algorithm
Pow2, W, v = TwoStageAlgorithm(Hr, Hd, G, M, N, Uk, gamma, epsilon, L)


#%% 画图
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
axs.plot(np.arange(iternum), Pow, color = 'k', linestyle='-', lw = 3,  marker = "o", markerfacecolor='white',markersize = 10, label = "Solving P4'",  )
axs.axhline(y = Pow2, ls=':', color='r', lw = 3, label='Two Stage Solution')

# font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs.set_xlabel( "Number of iterations", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
axs.set_ylabel('Transmit power at the AP(dBm)', fontproperties=font2, )
# axs.set_title(f'd = {d}(m)', fontproperties=font2)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 20}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# x_major_locator = MultipleLocator(5)               #把x轴的刻度间隔设置为1，并存在变量里
# axs.xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
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
# out_fig.savefig('fig7.eps' )
plt.show()














































