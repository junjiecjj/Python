#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 11:44:57 2025

@author: jack
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy
import cvxpy as cp

from WaterFilling import water_filling, plot_waterfilling, waterfilling_manual
from ReverseWaterFilling import reverse_waterfill_D
#%%
# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)

#%%
Ms = 5
Mc = 5
N = 10
T = 100
L = 20

Pt = 1
lamba_s = 1

K = 100
delta_k = 1 # np.random.rand(K)
theta_k_deg = np.random.uniform(-90, 90, K)
theta_k_rad = np.deg2rad(theta_k_deg)
# 初始化 Sigma_s
Sigma_S = np.zeros((N, N), dtype=complex)
for k in range(K):
    a_theta =  (1/np.sqrt(N)) * np.exp(1j * np.pi * np.arange(N) * np.sin(theta_k_rad[k]))
    Sigma_S += delta_k**2 * np.outer(a_theta, a_theta.conj())
# Sigma_S /= np.sqrt(K)
Lambda_s, U_s = np.linalg.eig(Sigma_S)
Lambda_s = np.abs(Lambda_s)

## Sigma_C
Hc = np.random.randn(Mc, N) + 1j * np.random.randn(Mc, N)
Sigma_C = Hc.conj().T @ Hc
Lambda_c, U_c = np.linalg.eig(Sigma_C)
Lambda_c = np.abs(Lambda_c)

SNR_c = 20  # dB
SNR_s = np.arange(0, 21, 5)

epslion = 1e-6

Distor_lst = []
ps_optim_lst = []
pc_optim_lst = []
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Algorithm 1 One-Dimensional Search Algorithm, Special Case: I.I.D. Sensing Subchannels
sigma_c2 = Pt * 10.0**(-SNR_c / 10.0)
for snr_s in SNR_s:
    Dists = []
    sigma_s2 = Pt * 10.0**(-snr_s / 10.0)

    D_old = -10
    D_new = 0
    p_min = 0
    p_max = Pt - p_min
    ps_optim = 0
    pc_optim = Pt - ps_optim
    while np.abs(D_new - D_old) > epslion:
        D_tol = []
        pc_lst_in = np.linspace(p_min, p_max, num = L+1, endpoint = 1)
        for l in range(L):
            pc = pc_lst_in[l]
            ps = Pt - pc
            Pc, _ = water_filling(sigma_c2, Lambda_c, pc)                         # Eq.(35)
            IcPc = np.sum(np.log(Lambda_c * Pc / sigma_c2 + 1))                   # Eq.(33a)
            Ps, _ = water_filling(sigma_s2, Lambda_s, ps)                         # Eq.(36)
            Ds = Ms * np.sum(sigma_s2 * Lambda_s / (sigma_s2 + Lambda_s * Ps))    # Eq.(33b)
            lambaR = Ps * Lambda_s**2 / (sigma_s2 + Lambda_s * Ps)                # Eq.(33c)
            Dc, xi, _, _ ,_ = reverse_waterfill_D(IcPc, lambaR)                            # Eq.(5)
            Dnow = Ds + Mc * Dc
            D_tol.append(Dnow)
        Dmin = np.min(D_tol)
        D_old = D_new
        D_new = Dmin
        l_star = np.array(D_tol).argmin()
        pc_optim = pc_lst_in[l_star]
        ps_optim = Pt - pc_optim
        if l_star == 0:
            l_mins_1 = 0
            l_plus_1 = 1
        else:
            l_mins_1 = l_star - 1
            l_plus_1 = l_star + 1
        p_min = pc_lst_in[l_mins_1]
        p_max = pc_lst_in[l_plus_1]
    Distor_lst.append(Dmin)
    ps_optim_lst.append(ps_optim)
    pc_optim_lst.append(pc_optim)
Distor_lst = np.array(Distor_lst)/N


##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ergodic all Ps and Pc to draw the curve
sigma_c2 = Pt * 10.0**(-SNR_c / 10.0)
K = 100
pc_lst = np.linspace(0, K, num = K+1) * Pt/K   #  np.arange(0, Pt + Pt/K, Pt/K)
Distor_ary = np.zeros((SNR_s.size, pc_lst.size))
for ii, snr_s in enumerate(SNR_s):
    Dists = []
    sigma_s2 = Pt * 10.0**(-snr_s / 10.0)
    for jj, pc in enumerate(pc_lst):
        ps = Pt - pc
        Pc, _ = water_filling(sigma_c2, Lambda_c, pc)                         # Eq.(35)
        IcPc = np.sum(np.log(Lambda_c * Pc / sigma_c2 + 1))                   # Eq.(33a)
        Ps, _ = water_filling(sigma_s2, Lambda_s, ps)                         # Eq.(36)
        Ds = Ms * np.sum(sigma_s2 * Lambda_s / (sigma_s2 + Lambda_s * Ps))    # Eq.(33b)
        lambaR = Ps * Lambda_s**2 / (sigma_s2 + Lambda_s * Ps)                # Eq.(33c)
        Dc, xi, _, _ ,_ = reverse_waterfill_D(IcPc, lambaR)                            # Eq.(5)
        Dnow = Ds + Mc * Dc
        Distor_ary[ii, jj] = Dnow
Distor_ary = Distor_ary/N
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
colors = plt.cm.jet(np.linspace(0, 1, 6))

markers = ['d', 'o', 's', 'v', '^', '+']
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
for jj, snrc in enumerate(SNR_s):
    axs.plot(Pt - pc_lst, Distor_ary[jj,:], ls = '-', lw = 2, color=colors[jj], label = "$SNR_c = $" + f"{snrc} dB")
axs.scatter(ps_optim_lst, Distor_lst, marker = '*', s = 160, color='red', label = "Optimal One-Dimensional Search ")

axs.set_xlabel('Power alocated for the SP (Ps)', )
axs.set_ylabel('Average Distortion', )

axs.set_title("The general case", fontsize = 25)

legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black',  )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

out_fig = plt.gcf()
plt.tight_layout()
plt.show()



#%%





#%%























