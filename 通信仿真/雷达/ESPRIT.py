#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:07:05 2024

@author: jack
# https://blog.csdn.net/weixin_44705592/article/details/131503963

# https://www.zhihu.com/question/270353751
"""


import sys
import numpy as np
import scipy
import cvxpy as cp
import matplotlib.pyplot as plt
# import math
# import matplotlib
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
from matplotlib.pyplot import MultipleLocator
# import scipy.constants as CONSTANTS

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

#%%%%%%%% ESPRIT for Uniform Linear Array%%%%%%%%
pi = np.pi
derad = pi/180                        # 角度->弧度
N = 8                                 # 阵元个数
K = 3                                 # 信源数目
thetaTrue = [-30, 0, 60]
theta = np.deg2rad(thetaTrue)         # 待估计角度
snr = 10                              # 信噪比
T = 512                               # 快拍数

d = np.arange(0, N).reshape(-1, 1)
A = np.exp(-1j * pi * d @ np.sin(theta).reshape(1,-1) )   # 方向矢量

#%%%% 构建信号模型 %%%%%
S = np.random.randn(K, T) + 1j * np.random.randn(K, T)            # 信源信号，入射信号
X = A@S                                                          # 构造接收信号
SigPow = np.power(np.abs(X), 2).mean()
noise_pwr = SigPow/(10**(snr/10))
noise = np.sqrt(noise_pwr ) * ( np.random.randn(*(X.shape)) + 1j * np.random.randn(*(X.shape)))
X1 = X + noise                  # 将白色高斯噪声添加到信号中
# 计算协方差矩阵
Rxx = X1 @ X1.T.conjugate() / T
# 特征值分解
D, U = np.linalg.eigh(Rxx)             # 特征值分解
idx = np.argsort(D)                    # 将特征值排序 从小到大
U = U[:, idx]
U = U[:,::-1]                          # 对应特征矢量排序
Us = U[:, 0:K]

## 角度估计
Ux = Us[0:K, :]
Uy = Us[1:K+1, :]

# ## 方法一:最小二乘法
# Psi = np.linalg.inv(Ux)@Uy
# Psi = np.linalg.solve(Ux,Uy)    # or Ux\Uy

## 方法二：完全最小二乘法
Uxy = np.hstack([Ux,Uy])
Uxy = Uxy.T.conjugate() @ Uxy
eigenvalues, eigvector = np.linalg.eigh(Uxy)          # 特征值分解
idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
eigvector = eigvector[:, idx]
eigvector = eigvector[:,::-1]                          # 对应特征矢量排序
F0 = eigvector[0:K, K:2*K]
F1 = eigvector[K:2*K, K:2*K]
Psi = -F0 @ np.linalg.inv(F1)

# 特征值分解
D, U = np.linalg.eig(Psi)          # 特征值分解
Theta = np.rad2deg(np.arcsin(-np.angle(D)/pi ))

Theta = np.sort(Theta)
print(f"Theta = {Theta}")



#%% 画图
# fig, axs = plt.subplots(1, 1, figsize=(10, 8))
# axs.plot(np.arange(-90, 90.1, 0.5), Pesprit , color = 'b', linestyle='-', lw = 3, label = "ESPRIT", )
# Theta = np.arange(-90, 90.1, 0.5)
# axs.plot(Theta[peaks], Pesprit[peaks], linestyle='', marker = 'o', color='r', markersize = 12)


# # font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
# font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
# # font2 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
# axs.set_xlabel( "DOA/(degree)", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
# axs.set_ylabel('Normalized Spectrum/(dB)', fontproperties=font2, )

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 20}
# # font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
# legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

# # x_major_locator = MultipleLocator(5)               #把x轴的刻度间隔设置为1，并存在变量里
# # axs.xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
# axs.tick_params(direction='in', axis='both', top=True, right=True,labelsize=16, width=3,)
# labels = axs.get_xticklabels() + axs.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(24) for label in labels]  # 刻度值字号

# axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
# axs.spines['bottom'].set_linewidth(1.5)    ###设置底部坐标轴的粗细
# axs.spines['left'].set_linewidth(1.5)      ####设置左边坐标轴的粗细
# axs.spines['right'].set_linewidth(1.5)     ###设置右边坐标轴的粗细
# axs.spines['top'].set_linewidth(1.5)       ####设置上部坐标轴的粗细

# out_fig = plt.gcf()
# # out_fig.savefig('ESPRIT.eps' )
# plt.show()









































































































































































