#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:35:34 2024
reproduction of Paper: Intelligent Reﬂecting Surface Enhanced Wireless Network via Joint Active and Passive Beamforming
@author: jack
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
from pylab import tick_params
from matplotlib.pyplot import MultipleLocator
import scipy.constants as CONSTANTS


sys.path.append("../")
from SDR import SDRsolver
from Utility import set_random_seed

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


epsilon = 1e-4
d0 = 51
dv = 2   # user到AP-RIS垂直距离
D0 = 1.0
C0 = -30   # dB
C0 = 10**(C0/10.0)     # 参考距离的路损
sigmaK2 = -80   # dB
sigmaK2 = 10**(sigmaK2/10.0) # 噪声功率
gamma = 10   # dB
gamma = 10**(gamma/10.0)    #  信干噪比约束10dB
M = 4    # AP天线数量
N = 16   # RIS天线数量
L = 200  # Gaussian随机化次数
frame = 500


# 路损参数
alpha_AI = 2      #  AP 和 IRS 之间path loss exponent
alpha_Iu = 2.8    # IRS 和 User 之间path loss exponent
alpha_Au = 3.5    # AP 和 User 之间path loss exponent
beta_AI = 100   # IRS到User考虑瑞利衰落信道，AP和IRS之间为纯LoS信道
beta_Au = 0   # IRS到User考虑瑞利衰落信道，AP和IRS之间为纯LoS信道
beta_Iu = 0   # IRS到User考虑瑞利衰落信道，AP和IRS之间为纯LoS信道


SDR = []
AO = []
AuMRT = []
AIMRT = []
LB = []
RANDOMphase = []
WithoutRIS = []


set_random_seed(100000)

G = np.sqrt(C0 * (d0/D0)**(-alpha_AI)) * np.ones((N, M))
D = np.arange(20, 51, 5) # user到AP-RIS连线的垂线距离AP的距离


for d in D:
    dAu = np.sqrt(d**2 + dv**2)
    dIu = np.sqrt((d0-d)**2 + dv**2)
    hr = np.sqrt(C0 * (dIu/D0)**(-alpha_Iu)) * np.sqrt(1 / (2 * sigmaK2)) * ( np.random.randn(N) + 1j * np.random.randn(N) )
    hd = np.sqrt(C0 * (dAu/D0)**(-alpha_Au)) * np.sqrt(1 / (2 * sigmaK2)) * ( np.random.randn(M) + 1j * np.random.randn(M) )
    hr = hr.reshape(-1, 1)
    hd = hd.reshape(-1, 1)

    for j in range(frame):
        SDRsolver(G, hr, hd )




















































