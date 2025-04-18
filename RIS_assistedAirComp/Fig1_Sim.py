


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

from Channel import Generate_hd, Generate_hr, Generate_hAI, Point2UPASteerVec


from DC_Solver import DC_RIS
from DC_Solver1 import DC_RIS1
from SCA_solver import SCA_RIS
from DC_wo_RIS import DC_woRIS
from SDR import SDR_RIS
from Utility import set_random_seed, set_printoption
set_random_seed(999)

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


C0 = -30                             # dB
C0 = 10**(C0/10.0)                   # 参考距离的路损
d0 = 1
### K =16,M =30,N =20
N = 20  # Ap antenna
K = 16 # User number
Nx = 8
Ny = 8
L = Nx * Ny  # RIS antenna

## path loss exponents
alpha_Au = 4.8
alpha_AI = 2.2
alpha_Iu = 2.2

## Rician factor
beta_Au = 0   # dB
beta_AI = 3   # dB
beta_Iu = 0   # dB
beta_Au = 10**(beta_Au/10)
beta_AI = 10**(beta_AI/10)
beta_Iu = 10**(beta_Iu/10)

sigmaK2 = -60                        # dBm
sigmaK2 = 10**(sigmaK2/10.0)/1000    # 噪声功率
P0 = 30 # dBm
P0 = 10**(P0/10.0)/1000

# Location, Case I
# BS_locate = np.array([[-50, 0, 10]])
# RIS_locate = np.array([[0, 0, 10]])
# users_locate_x1 = np.random.rand(int(K/2), 1) * (-20)
# users_locate_x2 = np.random.rand(int(K/2), 1) * 20 + 100
# users_locate_x = np.vstack((users_locate_x1, users_locate_x2))
# users_locate_y = np.random.rand(K, 1) * 20 - 10
# users_locate_z = np.zeros((K, 1))
# users_locate = np.hstack((users_locate_x, users_locate_y, users_locate_z))

# Location, Case II
BS_locate = np.array([[-50, 0, 10]])
RIS_locate = np.array([[0, 0, 10]])
users_locate_x = np.random.rand(K, 1) * (-20)
users_locate_y = np.random.rand(K, 1) * 20 - 10
users_locate_z = np.zeros((K, 1))
users_locate = np.hstack((users_locate_x, users_locate_y, users_locate_z))

# Location, Case III
# BS_locate = np.array([[0, 0, 25]])
# RIS_locate = np.array([[50, 50, 40]])
# users_locate_x = np.random.rand(K, 1) * 100 - 50
# users_locate_y = np.random.rand(K, 1) * 100 + 50
# users_locate_z = np.zeros((K, 1))
# users_locate = np.hstack((users_locate_x, users_locate_y, users_locate_z))

## Distance
d_Au = pairwise_distances(users_locate, BS_locate, metric = 'euclidean',)
d_Iu = pairwise_distances(users_locate, RIS_locate, metric = 'euclidean',)
d_AI = pairwise_distances(BS_locate, RIS_locate, metric = 'euclidean',)

## generate path-loss
PL_Au = C0 * (d_Au/d0)**(-alpha_Au)
PL_Iu = C0 * (d_Iu/d0)**(-alpha_Iu)
PL_AI = C0 * (d_AI/d0)**(-alpha_AI)

### Generate Channel, Method 1
## G: RIS to BS channel
# GLos = UPA2ULA_Los(N = N, Nx = Nx, Ny = Ny, azi_AP = 0, azi_RIS = -np.pi, ele_RIS = 0)
# GNLos = np.sqrt(1/2) * ( np.random.randn(N, L) + 1j * np.random.randn(N, L))
# h_AI = np.sqrt(PL_AI) * (np.sqrt(beta_AI/(1+beta_AI)) * GLos + np.sqrt(1/(1+beta_AI)) * GNLos )

# User to AP/RIS channel
# hdLos = Point2ULASteerVec(N, K, BS_locate, users_locate)
# hdNLos = np.sqrt(1/2) * ( np.random.randn(N, K) + 1j * np.random.randn(N, K))
# h_ds = (np.sqrt(beta_Au/(1+beta_Au)) * hdLos + np.sqrt(1/(1+beta_Au)) * hdNLos )
# h_d = h_ds @ np.diag(np.sqrt(PL_Au.flatten()/sigmaK2))

# hrLos = Point2UPASteerVec(K, Nx, Ny, RIS_locate, users_locate)
# hrNLos = np.sqrt(1/2) * ( np.random.randn(L, K) + 1j * np.random.randn(L, K))
# h_rs = (np.sqrt(beta_Iu/(1+beta_Iu)) * hrLos + np.sqrt(1/(1+beta_Iu)) * hrNLos )
# h_r = h_rs @ np.diag(np.sqrt(PL_Iu.flatten()/sigmaK2))

h_AI = Generate_hAI(N, Nx, Ny, RIS_locate, users_locate, beta_AI, PL_AI)
h_d = Generate_hd(N, K, BS_locate, users_locate, beta_Au, PL_Au, sigmaK2)
h_r = Generate_hr(K, Nx, Ny, RIS_locate, users_locate, beta_Iu, PL_Iu, sigmaK2)
G = np.zeros([N, L, K], dtype = complex) #  (5, 40, 40)
for k in range(K):
    G[:, :, k] = h_AI @ np.diag(h_r[:,k])

#%%
rho = 5
epsilon = 1e-3
epsilon_dc = 1e-7
verbose = 2
maxiter = 50
iter_num = 50

## Solver 3
Imax = 100000
tau = 1
threshold = 1e-5
f_sca, theta_sca, MSE_sca = SCA_RIS(N, L, K, h_d, G, threshold, P0, Imax, tau, verbose, RISON = 1)
print(f"MSE = {MSE_sca[-1]}, ||f||_2 = {np.linalg.norm(f_sca)}, |theta| = {np.abs(theta_sca)}")

f_sca_wo, theta_sca_wo, MSE_sca_wo = SCA_RIS(N, L, K, h_d, G, threshold, P0, Imax, tau, verbose, RISON = 0)
print(f"MSE = {MSE_sca_wo[-1]}, ||f||_2 = {np.linalg.norm(f_sca_wo)}, |theta| = {np.abs(theta_sca_wo)}")

# # # # Solver 1
# f_DC, theta_DC, MSE_DC = DC_RIS(N, L, K, h_d, G, epsilon, epsilon_dc, P0, maxiter, iter_num, rho, verbose, )
# # print(f"MSE_DC = {MSE_DC}, ||f_DC||_2 = {np.linalg.norm(f_DC)}, |theta_DC| = f{np.abs(theta_DC)}")

# Solver 4
# DC without RIS
# f_dc_woRIS, MSE_dc_wo = DC_woRIS(N, L, K, h_d, G, epsilon_dc, P0,  iter_num, rho, verbose, )
# print(f"MSE = {MSE_dc_wo[-1]}, ||f||_2 = {np.linalg.norm(f_dc_woRIS)}, ")

# # Solver 5
# f_sdr, theta_sdr, MSE_sdr = SDR_RIS(N, L, K, h_d, G, epsilon, P0, maxiter,  verbose, )


# np.savez('./fig1.npz', MSE_sca = MSE_sca, MSE_sca_wo = MSE_sca_wo, MSE_DC = MSE_DC, MSE_wo = MSE_wo )
# data = np.load('./fig1.npz')


# %% 画图
fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
axs.semilogy(np.arange(len(MSE_sca )), MSE_sca , color = 'r', lw = 3, linestyle='-',  label = 'Poposed w/ RIS',  )
# axs.semilogy(np.arange(len(MSE_DC)), MSE_DC, color = 'b', lw = 3,linestyle=(0,(1,1)),  label = 'DC',  )
# axs.semilogy(np.arange(len(MSE_sdr )), MSE_sdr , color = 'cyan', lw = 3, linestyle='-', label = 'SDR w/ RIS',  )
axs.semilogy(np.arange(len(MSE_sca_wo )), MSE_sca_wo , color = 'purple', lw = 3, linestyle='--',  label = 'SCA w/o RIS',  )
# axs.semilogy(np.arange(len(MSE_dc_wo)), MSE_dc_wo, color = 'olive', lw = 3, linestyle='--',  marker = '*', ms = 14, label = 'DC w/o RIS',  )

# axs.axhline(MSE_DC[-1,], linestyle = (0,(5,5)), lw = 2, color = 'gray')

# font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 30)
axs.set_xlabel( "Iterations", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
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


# axs.set_xlim([0,200])

out_fig = plt.gcf()
out_fig.savefig('./Figures/fig3a.eps' )
out_fig.savefig('./Figures/fig3a.pdf' )
plt.show()










































