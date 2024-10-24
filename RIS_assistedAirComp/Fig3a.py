#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 22:29:12 2024

@author: jack
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
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

from DC_Solver import DC_RIS
from DC_Solver1 import DC_RIS1
from SCA_solver import SCA_RIS
from DC_wo_RIS import DC_woRIS
from Utility import set_random_seed, set_printoption
set_random_seed(1)

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


SNR = 30 # dB, P0/sigma^2 = SNR
SNR = 10**(SNR/10)
T0 = 30  # dB
T0 = 10**(T0/10)
d0 = 1
alpha_Au = 3.5  # AP 和 User 之间的衰落因子
alpha_Iu = 2.8  # RIS 和 User 之间的衰落因子
alpha_AI = 2.2  # RIS 和 AP 之间的衰落因子
### K =16,M =30,N =20
K = 16 # 用户个数
L = 30 # RIS 元素
N = 20 # AP 天线个数
rho = 5
epsilon = 1e-3
epsilon_dc = 1e-8
verbose = 2
maxiter = 50
iter_num = 50

BS_locate = np.array([[0, 0, 25]])
RIS_locate = np.array([[50, 50, 40]])

users_locate_x = np.random.rand(K, 1) * 100 - 50
users_locate_y = np.random.rand(K, 1) * 100 + 50
users_locate_z = np.zeros((K, 1))
users_locate = np.hstack((users_locate_x, users_locate_y, users_locate_z))

d_Au = pairwise_distances(users_locate, BS_locate, metric = 'euclidean',)
d_Iu = pairwise_distances(users_locate, RIS_locate, metric = 'euclidean',)
d_AI = pairwise_distances(BS_locate, RIS_locate, metric = 'euclidean',)

PL_Au = T0 * (d_Au/d0)**(-alpha_Au)
PL_Iu = T0 * (d_Iu/d0)**(-alpha_Iu)
PL_AI = T0 * (d_AI/d0)**(-alpha_AI)

# trial_num = 50
# MSE_log = np.zeros((trial_num, maxiter+1))
# for t in range(trial_num):
    # print(f"{t}-th trial: ")
h_d  = np.sqrt(1/2) * (np.random.randn(N, K) + 1j * np.random.randn(N, K))
h_d = h_d @ np.diag(np.sqrt(PL_Au.flatten()))

H_UR = np.sqrt(1/2) * (np.random.randn(L, K) + 1j * np.random.randn(L, K))
H_UR = H_UR @ np.diag(np.sqrt(PL_Iu.flatten()))

H_RA = np.sqrt(PL_AI) * np.sqrt(1/2) * (np.random.randn(N, L) + 1j*np.random.randn(N, L))

G = np.zeros([N, L, K], dtype = complex) #  (5, 40, 40)
for j in range(K):
    G[:, :, j] = H_RA @ np.diag(H_UR[:,j])

## Solver
# f_DC, theta_DC, MSE_DC = DC_RIS(N, L, K, h_d, G, epsilon, epsilon_dc, SNR, maxiter, iter_num, rho, verbose, )
# print(f"MSE_DC = {MSE_DC}, ||f_DC||_2 = {np.linalg.norm(f_DC)}, |theta_DC| = f{np.abs(theta_DC)}")

# f_DC1, theta_DC1, MSE_DC1 = DC_RIS1(N, L, K, h_d, G, epsilon, epsilon_dc, SNR, maxiter, iter_num, rho, verbose)
# print(f"MSE = {MSE_DC1}, \n||f_DC1||_2 = {np.linalg.norm(f_DC1)}, \n|theta_DC1| = {np.abs(theta_DC1)}")

Imax = 10000000000
tau = 1
threshold = 1e-3
f_sca, theta_sca, MSE_sca = SCA_RIS(N, L, K, h_d, G, threshold, SNR, Imax, tau, verbose, RISON = 1)
print(f"MSE = {MSE_sca[-1]}, ||f||_2 = {np.linalg.norm(f_sca)}, |theta| = {np.abs(theta_sca)}")
# MSE_log[t] = MSE

# DC without RIS
f_woRIS, MSE_wo = DC_woRIS(N, L, K, h_d, G, epsilon, epsilon_dc, SNR, maxiter, iter_num, rho, verbose, )
print(f"MSE = {MSE_wo[-1]}, ||f||_2 = {np.linalg.norm(f_woRIS)}, ")

# %% 画图
# fig, axs = plt.subplots(1, 1, figsize=(10, 8))
# axs.semilogy(np.arange(len(MSE_sca)), MSE_sca, color = 'r', linestyle='-',  label = 'SCA',  )
# axs.semilogy(np.arange(len(MSE_DC)), MSE_DC, color = 'b', linestyle='--',  label = 'DC',  )
# # axs.semilogy(np.arange(len(MSE_DC1)), MSE_DC1, color = 'k', linestyle='--',  label = 'DC 1',  )
# axs.semilogy(np.arange(len(MSE_wo)), MSE_wo, color = 'k', linestyle='--',  label = 'DC w/o RIS',  )
# # font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
# font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs.set_xlabel( "Iterations", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
# axs.set_ylabel('MSE', fontproperties=font2, )

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 20}
# # font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
# legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')                         # 设置图例legend背景透明

# # x_major_locator = MultipleLocator(5)               # 把x轴的刻度间隔设置为1，并存在变量里
# # axs.xaxis.set_major_locator(x_major_locator)       # 把x轴的主刻度设置为1的倍数
# axs.tick_params(direction='in', axis='both', top=True, right=True,labelsize=16, width=3,)
# labels = axs.get_xticklabels() + axs.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(24) for label in labels]  # 刻度值字号

# axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
# axs.spines['bottom'].set_linewidth(1.5)    ### 设置底部坐标轴的粗细
# axs.spines['left'].set_linewidth(1.5)      #### 设置左边坐标轴的粗细
# axs.spines['right'].set_linewidth(1.5)     ### 设置右边坐标轴的粗细
# axs.spines['top'].set_linewidth(1.5)       #### 设置上部坐标轴的粗细

# out_fig = plt.gcf()
# # out_fig.savefig('fig3.eps' )
# plt.show()






















































































