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
from Utility import set_random_seed

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


#%%
epsilon = 1e-4
d0 = 51
dv = 2                 # user到AP-RIS垂直距离
D0 = 1.0
C0 = -30               # dB
C0 = 10**(C0/10.0)     # 参考距离的路损
sigmaK2 = -80          # dB
sigmaK2 = 10**(sigmaK2/10.0)  # 噪声功率
gamma = 10             # dB
gamma = 10**(gamma/10.0)    #  信干噪比约束10dB
M = 4     # AP天线数量
N = 30    # RIS天线数量
L = 1000  # Gaussian随机化次数
frame = 500

#%% 路损参数
alpha_AI = 2      #  AP 和 IRS 之间path loss exponent
alpha_Iu = 2.8    # IRS 和 User 之间path loss exponent
alpha_Au = 3.5    # AP 和 User 之间path loss exponent
# beta_AI = 100   # IRS到User考虑瑞利衰落信道，AP和IRS之间为纯LoS信道
# beta_Au = 0   # IRS到User考虑瑞利衰落信道，AP和IRS之间为纯LoS信道
# beta_Iu = 0   # IRS到User考虑瑞利衰落信道，AP和IRS之间为纯LoS信道

set_random_seed(1)

D = np.arange(20, 51, 5) # user到AP-RIS连线的垂线距离AP的距离

SDR = np.zeros(len(D))
AO = np.zeros(len(D))
AuMRT = np.zeros(len(D))
AIMRT = np.zeros(len(D))
LowBound = np.zeros(len(D))
RANDOMphase = np.zeros(len(D))
WithoutRIS = np.zeros(len(D))

#%%
G = np.sqrt(C0 * ((d0/D0)**(-alpha_AI))) * np.ones((N, M))
for i, d in enumerate(D):
    print(f"distance = {d}")
    dAu = np.sqrt(d**2 + dv**2)
    dIu = np.sqrt((d0-d)**2 + dv**2)
    ## h_r
    Iu_large_fading = C0 * ((dIu/D0)**(-alpha_Iu))
    ## h_d
    Au_large_fading = C0 * ((dAu/D0)**(-alpha_Au))

    for j in range(frame):
        if (j + 1) % 100 == 0: print(f"  {j+1} ", end = "  ")

        hr = np.sqrt(Iu_large_fading) * np.sqrt(1 / (2 * sigmaK2)) * ( np.random.randn(1,N) + 1j * np.random.randn(1,N) )
        hd = np.sqrt(Au_large_fading) * np.sqrt(1 / (2 * sigmaK2)) * ( np.random.randn(1,M) + 1j * np.random.randn(1,M) )

        #%% SDR and Lowbound
        lowbound, v = SDRsolver(G, hr, hd, N, L)
        opti = gamma/(np.linalg.norm(v.T.conjugate() @ (np.diag(hr.flatten()) @ G) + hd, ord = 2)**2 )
        SDR[i] = SDR[i] + opti
        LowBound[i] += gamma/lowbound

        #%% AO 交替迭代算法
        P_AO = AlternatingOptim(hd,hr,G, epsilon, gamma)
        AO[i] += P_AO

        #%% AP-User MRT
        v_aumrt, w_aumrt = AU_MRT(hd, hr, G)
        ## 注意这里的相位对齐后，对v只需要转置即可，不需要共轭转置
        P_aumrt = gamma/(np.linalg.norm((v_aumrt.T @ (np.diag(hr.flatten()) @ G) + hd) @ w_aumrt)**2)
        AuMRT[i] += P_aumrt;

        #%% AP-IRS MRT
        v_aimrt, w_aimrt = AI_MRT(hd, hr, G)
        ## 注意这里的相位对齐后，对v只需要转置即可，不需要共轭转置
        P_aimrt = gamma/(np.linalg.norm((v_aimrt.T @(np.diag(hr.flatten()) @ G) + hd) @ w_aimrt)**2)
        AIMRT[i] += P_aimrt;

        #%% RANDOMphase
        theta = 2 * np.pi * np.random.rand(1, N)
        Theta = np.diag(np.exp(1j * theta.flatten()))
        P_rand = gamma/(np.linalg.norm(hr @ Theta @ G + hd, 2)**2)
        RANDOMphase[i] += P_rand

        #%% WithoutRIS
        WithoutRIS[i] += gamma/(np.linalg.norm(hd, 2)**2)

    print("\n")
    SDR[i] = 10 * np.log10(SDR[i]/frame)
    LowBound[i] = 10 * np.log10(LowBound[i]/frame)
    AO[i] = 10 * np.log10(AO[i]/frame)
    AuMRT[i] = 10 * np.log10(AuMRT[i]/frame)
    AIMRT[i] = 10 * np.log10(AIMRT[i]/frame)
    RANDOMphase[i] = 10 * np.log10(RANDOMphase[i]/frame)
    WithoutRIS[i] = 10 * np.log10(WithoutRIS[i]/frame)
    print(f"  SDR = {SDR[i]}, LowBound = {LowBound[i]}, AO = {AO}, AuMRT = {AuMRT}, AIMRT = {AIMRT}, RANDOMphase = {RANDOMphase}, WithoutRIS = {WithoutRIS}")


# SDR = 10 * np.log10(SDR/frame )
# LowBound = 10 * np.log10(LowBound/frame )
# AO = 10 * np.log10(AO/frame )
# AuMRT = 10 * np.log10(AuMRT/frame )
# AIMRT = 10 * np.log10(AIMRT/frame )
# RANDOMphase = 10 * np.log10(RANDOMphase/frame )
# WithoutRIS = 10 * np.log10(WithoutRIS/frame )


#%% 画图
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
axs.plot(D, LowBound, color = 'k', linestyle='none',  marker = "o", markerfacecolor='white',markersize = 20, label = 'Lower Bound',  )
axs.plot(D, SDR, color='b', linestyle='-',  lw = 3, marker = "d", markersize = 12, label = 'SDR',  )

axs.plot(D, AO, color='r', linestyle='--', lw = 3, marker = "*", markersize = 12, label = 'AO',  )
axs.plot(D, AuMRT, color='#28a428', linestyle='--', lw = 3, marker = "v", markersize = 12, label = 'AP-User MRT',  )
axs.plot(D, AIMRT, color='#FF00FF', linestyle='-', lw = 3, marker = "^", markersize = 12, label = 'AP_RIS MRT',  )
axs.plot(D, RANDOMphase, color='k', linestyle='--', lw = 3, marker = "P", markersize = 12, label = 'Random Phase',  )
axs.plot(D, WithoutRIS, color='k', linestyle='--', lw = 3, marker = "s", markersize = 14, markerfacecolor='none',  label = 'Without RIS',  )

# font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs.set_xlabel( "AP-User horizonal distance(m)", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
axs.set_ylabel('Transmit power at the AP(dBm)', fontproperties=font2, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 20}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

x_major_locator = MultipleLocator(5)               #把x轴的刻度间隔设置为1，并存在变量里
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
# out_fig.savefig('fig4a.eps' )
plt.show()














































