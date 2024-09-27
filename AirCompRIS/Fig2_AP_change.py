


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:28:33 2024

@author: jack
"""



import scipy
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

from Channel import Generate_hd, Generate_hr, Generate_hAI


from DC_Solver import DC_RIS
from DC_Solver1 import DC_RIS1
from SCA_solver import SCA_RIS
from DC_wo_RIS import DC_woRIS
from DC_randomtheta import DC_random_theta
from Utility import set_random_seed, set_printoption
set_random_seed(999)

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


C0 = -30                             # dB
C0 = 10**(C0/10.0)                   # 参考距离的路损
d0 = 1
### K =16, M =30, N =20
N = 20  # Ap antenna
K = 16 # User number
Nx = 8
Ny = 8
L = Nx * Ny  # RIS antenna

## path loss exponents
alpha_Au = 3.5
alpha_AI = 2.2
alpha_Iu = 2.8

## Rician factor
beta_Au = 0   # dB
beta_AI = 3   # dB
beta_Iu = 3   # dB
beta_Au = 10**(beta_Au/10)
beta_AI = 10**(beta_AI/10)
beta_Iu = 10**(beta_Iu/10)

sigmaK2 = -80                        # dBm
sigmaK2 = 10**(sigmaK2/10.0)/1000    # 噪声功率
P0 = 30 # dBm
P0 = 10**(P0/10.0)/1000

## Location, Case I
BS_locate = np.array([[-50, 0, 10]])
RIS_locate = np.array([[0, 0, 10]])
users_locate_x1 = np.random.rand(int(K/2), 1) * (-20)
users_locate_x2 = np.random.rand(int(K/2), 1) * 20 + 100
users_locate_x = np.vstack((users_locate_x1, users_locate_x2))
users_locate_y = np.random.rand(K, 1) * 20 - 10
users_locate_z = np.zeros((K, 1))
users_locate = np.hstack((users_locate_x, users_locate_y, users_locate_z))


## Distance
d_Au = pairwise_distances(users_locate, BS_locate, metric = 'euclidean',)
d_Iu = pairwise_distances(users_locate, RIS_locate, metric = 'euclidean',)
d_AI = pairwise_distances(BS_locate, RIS_locate, metric = 'euclidean',)

## generate path-loss
PL_Au = C0 * (d_Au/d0)**(-alpha_Au)
PL_Iu = C0 * (d_Iu/d0)**(-alpha_Iu)
PL_AI = C0 * (d_AI/d0)**(-alpha_AI)


#%%
rho = 5
epsilon = 1e-3
epsilon_dc = 1e-8
verbose = 2
maxiter = 50
iter_num = 50

## Solver 3
Imax = 100000
tau = 1
threshold = 1e-5
APlst = np.arange(5, 31, 5)
# APlst = [25, 30]

sca_res = np.zeros((len(APlst),) )
sca_wo_res = np.zeros((len(APlst),) )
dc_res = np.zeros((len(APlst),) )
dc_rand_res = np.zeros((len(APlst),) )
Avg = 20

for i, N in enumerate(APlst):
    for fm in range(Avg):
        print(f"AP antennas = {N}, frame = {fm}")
        ### Generate Channel, Method 1
        h_AI = Generate_hAI(N, Nx, Ny, RIS_locate, users_locate, beta_AI, PL_AI)
        h_d = Generate_hd(N, K, BS_locate, users_locate, beta_Au, PL_Au, sigmaK2)
        h_r = Generate_hr(K, Nx, Ny, RIS_locate, users_locate, beta_Iu, PL_Iu, sigmaK2)
        G = np.zeros([N, L, K], dtype = complex) #  (5, 40, 40)
        for k in range(K):
            G[:, :, k] = h_AI @ np.diag(h_r[:,k])

        ## SCA
        # f_sca, theta_sca, MSE_sca = SCA_RIS(N, L, K, h_d, G, threshold, P0, Imax, tau, verbose, RISON = 1)
        # print(f"  SCA MSE = {MSE_sca[-1]}, ")
        # sca_res[i] += MSE_sca[-1]

        ## SCA wo RIS
        # f_sca_wo, theta_sca_wo, MSE_sca_wo = SCA_RIS(N, L, K, h_d, G, threshold, P0, Imax, tau, verbose, RISON = 0)
        # print(f"  SCA wo ris MSE = {MSE_sca_wo[-1]},  ")
        # sca_wo_res[i] += MSE_sca_wo[-1]

        ## DC
        # f_dc, theta_dc, MSE_DC = DC_RIS(N, L, K, h_d, G, epsilon, epsilon_dc, P0, maxiter, iter_num, rho, verbose, )
        # print(f"MSE_DC = {MSE_DC[-1]},  ")
        # dc_res[i] += MSE_DC[-1]

        # DC without RIS
        # f_dc_random, MSE_dc_random = DC_random_theta(N, L, K, h_d, G, epsilon, epsilon_dc, P0, maxiter, iter_num, rho, verbose, )
        # print(f"MSE dc random = {MSE_dc_random[-1]},  ")
        # dc_rand_res[i] += MSE_dc_random[-1]

sca_res =  np.array([0.207355, 0.087434, 0.063046, 0.052393, 0.049518, 0.046928]) # sca_res/Avg #
sca_wo_res = np.array([0.284617, 0.129787, 0.091733, 0.07062 , 0.067346, 0.060522]) # sca_wo_res/Avg # np.array([0.372221, 0.134684, 0.094441, 0.077333, 0.067874, 0.059856])  #  sca_wo_res/Avg
dc_res = np.array([0.3523, 0.216842, 0.1571707428, 0.1206409, 0.09806409, 0.07409]) # dc_res/Avg
# dc_res = np.array([x,x,0.09633 , 0.085182,x,x ])
# dc_rand_res = np.array([ x, x, 0.08873, 0.075455, x, x ]) # dc_rand_res/Avg

# np.savez('./fig1.npz', MSE_sca = MSE_sca, MSE_sca_wo = MSE_sca_wo, MSE_DC = MSE_DC, MSE_wo = MSE_wo )
# data = np.load('./fig1.npz')

# %% 画图
fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
axs.semilogy(APlst, sca_res, color = 'r', lw = 3, linestyle='-',  marker = 'o',ms = 12, label = 'Poposed SCA w/ RIS',  )
axs.semilogy(APlst, sca_wo_res, color = 'purple', lw = 3, linestyle='--',  marker = 'o',ms = 12, label = 'SCA w/o RIS',  )
axs.semilogy(APlst, dc_res, color = 'b', lw = 3,linestyle='--',  marker = 'o',ms = 12, label = 'DC w/ RIS',  )
# axs.semilogy(APlst, dc_rand_res, color = 'green', lw = 3, linestyle='--',  label = 'random phase',  )
# axs.axhline(MSE_wo[-1,], linestyle = (0,(5,5)), lw = 2, color = 'gray')

# font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 30)
axs.set_xlabel( "Number of AP antennas", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
axs.set_ylabel('MSE', fontproperties=font2, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 25}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')                         # 设置图例legend背景透明

# x_major_locator = MultipleLocator(5)               # 把x轴的刻度间隔设置为1，并存在变量里
# axs.xaxis.set_major_locator(x_major_locator)       # 把x轴的主刻度设置为1的倍数
axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 25, width=3,)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
axs.spines['bottom'].set_linewidth(1.5)    ### 设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(1.5)      #### 设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(1.5)     ### 设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(1.5)       #### 设置上部坐标轴的粗细

out_fig = plt.gcf()
out_fig.savefig('fig2_AP.eps' )
out_fig.savefig('fig2_AP.pdf' )
plt.show()










































