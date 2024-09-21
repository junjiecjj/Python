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
from Solver import SOCPforW, SDPforV
from Solver import AlternatingOptim, TwoStageAlgorithm
from Utility import set_random_seed, set_printoption

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

# set_random_seed(10000)
# set_printoption()

#%%
pi = np.pi
epsilon = 1e-4
d0 = 51
dv = 2                               # user到AP-RIS垂直距离
D0 = 1.0

d1 = 20                              # AP半径
d2 = 3                               # RIS半径
C0 = -30                             # dB
C0 = 10**(C0/10.0)                   # 参考距离的路损
sigmaK2 = -80                        # dBm
sigmaK2 = 10**(sigmaK2/10.0)/1000    # 噪声功率
GammaDB = np.arange(-4, 25, 4)       # dB
Gamma = 10**(GammaDB/10.0)           # 信干噪比约束10dB
M = 4                                # AP天线数量
Nx = 5
Ny = 6
N = Nx * Ny                          # RIS天线数量
L = 1000                             # Gaussian随机化次数
frame = 500

#%% 路损参数
alpha_AI = 2.8           # AP 和 IRS 之间path loss exponent
alpha_Iu = 2.8           # IRS 和 User 之间path loss exponent
alpha_Au = 3.5           # AP 和 User 之间path loss exponent
beta_AI = 10**(3/10)     # IRS到User考虑瑞利衰落信道，AP和IRS之间为纯LoS信道
beta_Iu = 10**(3/10)     # IRS到User考虑瑞利衰落信道，AP和IRS之间为纯LoS信道
beta_Au = 0              # AP和 User 之间为Rayleigh信道

Uk = 2                   # 2个用户，图8仿真，假设 U_k, k = 1,2 是活跃用户

#%% AP-User和RIS-User之间的距离和角度
d_Au = [d1, np.sqrt((d2*np.cos(pi/5))**2 + (d0 - d2*np.sin(pi/5))**2), ]
theta_Au = [-pi/4, 2*pi - np.arctan(d2*np.cos(pi/5) / (d0 - d2*np.sin(pi/5))), ]
d_Iu = [np.sqrt((d1*np.sin(pi/4))**2 + (d0-d1*np.cos(pi/4))**2), d2,  ]
theta_Iu = [pi + np.arctan(d1*np.sin(pi/4)/(d0 - d1*np.cos(pi/4))), 3*pi/2 - pi/5, ]

# #%% AP-User和RIS-User之间的距离和角度
# d_Au = [d1, np.sqrt((d2*np.cos(pi/5))**2 + (d0 - d2*np.sin(pi/5))**2), ]
# theta_Au = [-pi/4, 2*pi - np.arctan(d2*np.cos(pi/5) / (d0 - d2*np.sin(pi/5))), ]
# d_Iu = [np.sqrt((d1*np.sin(pi/4))**2 + (d0-d1*np.cos(pi/4))**2), d2,  ]
# theta_Iu = [pi + np.arctan(d1*np.sin(pi/4)/(d0 - d1*np.cos(pi/4))), 3*pi/2 - pi/5, ]

## results, User1
CombPowUser1 = np.zeros(Gamma.size)
DirPowUser1 = np.zeros(Gamma.size)
CombInterfUser1 = np.zeros(Gamma.size)
DirInterUser1 = np.zeros(Gamma.size)
## results, User2
CombPowUser2 = np.zeros(Gamma.size)
DirPowUser2 = np.zeros(Gamma.size)
CombInterfUser2 = np.zeros(Gamma.size)
DirInterUser2 = np.zeros(Gamma.size)


for i, gamma in enumerate(Gamma):
    print(f"gamma = {GammaDB[i]}(dB)")
    for j in range(frame):
        print("\r   " + "▇"*int(j/frame*100) + f"{j/frame*100:.5f}%", end="")
        # if (j + 1) % 50 == 0: print(f"  {j+1} ", end = "  ")
        #%% AP-RIS 信道G
        AI_large_fading = C0 * ((d0/D0)**(-alpha_AI))
        GLos = ULA2UPA_Los(M = M, Nx = Nx, Ny = Ny, azi_AP = 0, ele_AP = 0, azi_RIS = -np.pi, ele_RIS = 0)
        GNLos = np.sqrt(1/2) * ( np.random.randn(N, M) + 1j * np.random.randn(N, M) )
        G = np.sqrt(AI_large_fading) * (np.sqrt(beta_AI/(1+beta_AI)) * GLos + np.sqrt(1/(1+beta_AI)) * GNLos )

        #%% AP/RIS-User信道Hr, Hd
        Hr = np.zeros((N, Uk), dtype = complex)
        Hd = np.zeros((M, Uk), dtype = complex)
        for k in range(Uk):
            # RIS-User channel
            Iu_large_fading = C0 * ((d_Iu[k]/D0)**(-alpha_Iu))
            IULos = RIS2UserSteerVec(Nx = Nx, Ny = Ny, azi = theta_Iu[k], ele = 0)
            IUNLos = np.sqrt(1/2) * ( np.random.randn(N, ) + 1j * np.random.randn(N, ) )
            Hr[:,k] = np.sqrt(Iu_large_fading/sigmaK2) * (np.sqrt(beta_Iu/(beta_Iu+1)) * IULos + np.sqrt(1/(beta_Iu+1)) * IUNLos)
            # AP-User channel
            Au_large_fading = C0 * ((d_Au[k]/D0)**(-alpha_Au))
            AU_NLos = np.sqrt(1/2) * ( np.random.randn(M, ) + 1j * np.random.randn(M, ))
            Hd[:,k] = np.sqrt(Au_large_fading/sigmaK2) * AU_NLos

        #%% 交替优化
        iternum, Pow, W, v = AlternatingOptim(Hr, Hd, G, M, N, Uk, gamma, epsilon, L)

        #%% 两阶段优化
        # Pow2, W, v = TwoStageAlgorithm(Hr, Hd, G, M, N, Uk, gamma, epsilon, L)

        #%% calcuate the combine/direct pow/interference
        Theta = np.diag(v.flatten().conjugate())
        H = Hr.T.conjugate() @ Theta @ G + Hd.T.conjugate()
        HdH = Hd.T.conjugate()
        HW = 10*np.log10(np.power(np.abs(H@W), 2))
        HdHW = 10*np.log10(np.power(np.abs(HdH@W), 2))

        ## User1
        CombPowUser1[i]    = CombPowUser1[i] + HW[0,0]
        DirPowUser1[i]     = DirPowUser1[i] + HdHW[0,0]
        CombInterfUser1[i] = CombInterfUser1[i] + HW[0,1]
        DirInterUser1[i]   = DirInterUser1[i] + HdHW[0,1]
        ## User2
        CombPowUser2[i]    = CombPowUser2[i] + HW[1,1]
        DirPowUser2[i]     = DirPowUser2[i] + HdHW[1,1]
        CombInterfUser2[i] = CombInterfUser2[i] + HW[1,0]
        DirInterUser2[i]   = DirInterUser2[i] + HdHW[1,0]

    ## User1
    CombPowUser1[i]    = CombPowUser1[i] / frame
    DirPowUser1[i]     = DirPowUser1[i] / frame
    CombInterfUser1[i] = CombInterfUser1[i] / frame
    DirInterUser1[i]   = DirInterUser1[i] / frame
    ## User2
    CombPowUser2[i]    = CombPowUser2[i] / frame
    DirPowUser2[i]     = DirPowUser2[i] / frame
    CombInterfUser2[i] = CombInterfUser2[i] / frame
    DirInterUser2[i]   = DirInterUser2[i] / frame

    print("\n")

    print(f"  CombPowUser1 = {CombPowUser1} \n  DirPowUser1 = {DirPowUser1} \n  CombInterfUser1 = {CombInterfUser1} \n  DirInterUser1 = {DirInterUser1} \n")
    print(f"  CombPowUser2 = {CombPowUser2} \n  DirPowUser2 = {DirPowUser2} \n  CombInterfUser2 = {CombInterfUser2} \n  DirInterUser2 = {DirInterUser2} ")


#%% Alternating Optimization Algorithm

CombPowUser1 = [-2.0799772, 2.10636887, 5.8448003, 9.03406649, 12.55630389, 16.22771828, 20.09668351, 24.04169945]
DirPowUser1 = [-2.08055845, 2.10578219, 5.8440206, 9.03323689, 12.55530399, 16.22622727, 20.09523708, 24.04019478]
CombInterfUser1 = [ -4.83219208, -4.24665882, -5.05985184, -8.27634419, -11.12062527, -15.35908344, -19.49777904, -23.2319149 ]
DirInterUser1 = [ -4.82822009, -4.23752622, -5.04063946, -8.23469137, -11.00171942,-15.04446886, -18.47020826, -20.6489935 ]

CombPowUser2 = [-3.95370216, 0.05736258, 4.03197019, 8.01273852, 12.00512966, 16.00190653, 20.00078589, 24.00091855]
DirPowUser2 = [-7.69243841 -3.82740164, 0.20169768, 3.90261353, 8.02084504, 12.06242531, 16.02467885, 20.17734629]
CombInterfUser2 = [-23.83860108, -23.35860197, -25.13280465, -28.92625093, -32.57850102, -36.82255008 -40.96796123 -43.40271314]
DirInterUser2 = [-25.61378372, -22.84222952, -21.23208638, -17.70959918, -14.80038396, -11.34295932, -6.7980275, -3.70797191]

#%% Two stage Algorithm

CombPowUser1 = 10*np.log10( np.array([ 0.73614686, 2.09673112, 4.89168308, 9.4201684, 19.58381834, 42.94352514, 103.18613877, 254.64462118]))
DirPowUser1 = 10*np.log10(np.array([ 0.73591066, 2.09623748, 4.8902981, 9.41788966, 19.57929452, 42.93441118, 103.16173038, 254.59314357]))
CombInterfUser1 = 10*np.log10(np.array([0.84908611, 1.09671724, 0.94740815, 0.49299385, 0.23565441, 0.07869223, 0.03186115, 0.01375836]))
DirInterUser1 = 10*np.log10(np.array([0.84862257, 1.09609212, 0.94638435, 0.49197506, 0.23451948, 0.07848578, 0.03194468, 0.01388827]))

CombPowUser2 = 10*np.log10(np.array([ 0.4035666, 1.02364521, 2.54711695, 6.34129117, 15.87769044, 39.83977692, 100.02918548, 251.21553493]))
DirPowUser2 = 10*np.log10(np.array([0.17457879, 0.45206891, 1.1279531, 2.74709967, 6.93436564, 17.82080905, 44.84914786, 110.38235518]))
CombInterfUser2 = 10*np.log10(np.array([0.01371096, 0.02364409, 0.01402506, 0.00502675, 0.00181448, 0.00072993, 0.00029185, 0.00010706]))
DirInterUser2 = 10*np.log10(np.array([0.00743672, 0.01358386, 0.01903976, 0.02775385, 0.06078126, 0.14360453, 0.3755818, 0.80195195]))



#%% fig9(a)
fig, axs = plt.subplots(1, 1, figsize=(10, 8))

axs.plot(GammaDB, CombPowUser1 , mec='r', linestyle = 'none', marker = "o", mfc='white', ms = 12, mew = 3, label = 'Combined desired signal', )
axs.plot(GammaDB, DirPowUser1, color = 'b', linestyle='--', lw = 3,  label = "Desired signal from AP-user link", )
axs.plot(GammaDB, CombInterfUser1, linestyle = 'none', mec='k',marker = "o", mfc='white', ms = 12, mew = 3, label = 'Combined interference', )
axs.plot(GammaDB, DirInterUser1, color='#DAA520', linestyle='--', lw = 3,  label = 'Interference from AP-user link', )

# font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs.set_xlabel( "User SINR target, (dB)", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
axs.set_ylabel('Normalized singal power(dB)', fontproperties=font2, )
axs.set_title('User 1', fontproperties=font2)

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
out_fig.savefig('fig9a_AO.eps' )

plt.show()

#%% fig9(b)
fig, axs = plt.subplots(1, 1, figsize=(10, 8))

axs.plot(GammaDB, CombPowUser2 , color='r', marker = "o", mfc='white', ms = 12, mew = 3, label = 'Combined desired signal 1', )
axs.plot(GammaDB, DirPowUser2, color = 'b', linestyle='--', lw = 3,  label = "Desired signal from AP-user link", )
axs.plot(GammaDB, CombInterfUser2, color='k',marker = "o", mfc='white',ms = 12, mew = 3, label = 'Combined interference', )
axs.plot(GammaDB, DirInterUser2, color='#DAA520', linestyle='--', lw = 3, marker = "d", markersize = 12,  label = 'Interference from AP-user link', )

# font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs.set_xlabel( "User SINR target, (dB)", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
axs.set_ylabel('Normalized singal power(dB)', fontproperties=font2, )
axs.set_title('User 2', fontproperties=font2)

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
out_fig.savefig('fig9b_AO.eps' )
plt.show()















































