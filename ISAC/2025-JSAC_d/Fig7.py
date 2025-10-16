#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:43:45 2025

@author: jack
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy
import cvxpy as cp

from WaterFilling import water_filling, plot_waterfilling
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
# T = 100
# L = 20
Pt = 1
I_N = np.eye(N)
I_Mc = np.eye(Mc)

K = 20
delta_k = 1 # np.random.rand(K)
theta_k_deg = np.random.uniform(-90, 90, K)
theta_k_rad = np.deg2rad(theta_k_deg)
# 初始化 Sigma_s
Sigma_S = np.zeros((N, N), dtype=complex)
for k in range(K):
    a_theta =  (1/np.sqrt(N)) * np.exp(1j * np.pi * np.arange(N) * np.sin(theta_k_rad[k]))
    Sigma_S += delta_k**2 * np.outer(a_theta, a_theta.conj())
Lambda_s, U_s = np.linalg.eig(Sigma_S)
Sigma_S = np.diag(np.abs(Lambda_s))
Sigma_S_inv = np.linalg.inv(Sigma_S)

## Sigma_C
Hc = np.random.randn(Mc, N) + 1j * np.random.randn(Mc, N)
Sigma_C = Hc.conj().T @ Hc
Lambda_c, U_c = np.linalg.eig(Sigma_C)
Lambda_c = np.abs(Lambda_c)

#%%  Algorithm 2 Heuristic MI Maximization Algorithm

def SolverProb45(Hc, Sigma_S, Pt, alpha, sigma_s2, sigma_c2):
    Mc, N = Hc.shape
    # I_N = np.eye(N)
    # I_Mc = np.eye(Mc)
    R = cp.Variable((N, N), hermitian = True)

    obj = cp.Maximize(alpha * Ms * cp.log_det(Sigma_S @ R / sigma_s2 + I_N) + (1 - alpha) * cp.log_det(Hc @ R @ Hc.conj().T / sigma_c2 + I_Mc))
    constraints = [
                    0 << R,
                    cp.trace(R) <= Pt
                    ]
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(verbose=False,)
    except Exception as e:
        print(f"  Solver error : {e}")

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"  Problem status - {prob.status}")

    return R.value

L = 11
alpha_lst = np.linspace(0, 1, L)

SNR_C = [-5, 0, 5]
SNR_S = np.arange(-10, 11, 5)

Distor_ary11 = np.zeros((len(SNR_C), SNR_S.size, L))

for ii, snr_c in enumerate(SNR_C):
    sigma_c2 = Pt * 10.0**(-snr_c / 10.0)
    for jj, snr_s in enumerate(SNR_S):
        sigma_s2 = Pt * 10.0**(-snr_s / 10.0)
        for kk, alpha in enumerate(alpha_lst):
            R = SolverProb45(Hc, Sigma_S, Pt, alpha, sigma_s2, sigma_c2)               # Solve problem (45) with α(l) and obtain R
            Lambda_s2, Psi = np.linalg.eig(R)                                          # Obtain Ψ(l), Λ(l) according to R(l)
            Ds = Ms * np.trace(np.linalg.inv(R/sigma_s2 + Sigma_S_inv))                # Compute Ds(R(l)) according to (17b)
            IcR = np.log(np.linalg.det(Hc @ R @ Hc.conj().T/sigma_c2 + I_Mc ))         # Compute Ic(R(l)) according to (7)
            R_eta = Sigma_S - np.linalg.inv(R/sigma_s2 + Sigma_S_inv)
            lambaR, _ = np.linalg.eig(R_eta)
            Dc, _ = reverse_waterfill_D(IcR.real, np.abs(lambaR))
            D_tol = Ds.real +  Mc * Dc
            Distor_ary11[ii, jj, kk] = D_tol
Distor_ary1 = np.min(Distor_ary11, axis = -1) / N


L = 3
alpha_lst = np.linspace(0, 1, L)
Distor_ary3 = np.zeros((len(SNR_C), SNR_S.size, L))
for ii, snr_c in enumerate(SNR_C):
    sigma_c2 = Pt * 10.0**(-snr_c / 10.0)
    for jj, snr_s in enumerate(SNR_S):
        sigma_s2 = Pt * 10.0**(-snr_s / 10.0)
        for kk, alpha in enumerate(alpha_lst):
            R = SolverProb45(Hc, Sigma_S, Pt, alpha, sigma_s2, sigma_c2)               # Solve problem (45) with α(l) and obtain R
            Lambda_s2, Psi = np.linalg.eig(R)                                          # Obtain Ψ(l), Λ(l) according to R(l)
            Ds = Ms * np.trace(np.linalg.inv(R/sigma_s2 + Sigma_S_inv))                # Compute Ds(R(l)) according to (17b)
            IcR = np.log(np.linalg.det(Hc @ R @ Hc.conj().T/sigma_c2 + I_Mc ))         # Compute Ic(R(l)) according to (7)
            R_eta = Sigma_S - np.linalg.inv(R/sigma_s2 + Sigma_S_inv)
            lambaR, _ = np.linalg.eig(R_eta)
            Dc, _ = reverse_waterfill_D(IcR.real, np.abs(lambaR))
            D_tol = Ds.real +  Mc * Dc
            Distor_ary3[ii, jj, kk] = D_tol
Distor_ary2 = np.min(Distor_ary3, axis = -1) / N

##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
colors = plt.cm.jet(np.linspace(0, 1, 6))

markers = ['d', 'o', 's', 'v', '^', '+']
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
for idx, snrc in enumerate(SNR_C):
    axs.plot(SNR_S, Distor_ary1[idx,:], ls = '-', lw = 2, marker = markers[idx], color=colors[idx], label = "$SNR_c = $" + f"{snrc} dB, L = 11")
    axs.plot(SNR_S, Distor_ary2[idx,:], ls = '--', lw = 2, marker = markers[idx], color=colors[idx], label = "$SNR_c = $" + f"{snrc} dB, L = 3")

axs.set_xlabel('Sensing Channel SNR (dB)', )
axs.set_ylabel('Average CAS Distortion', )

axs.set_title("General case", fontsize = 25)

legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black',  )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

out_fig = plt.gcf()
plt.tight_layout()
plt.show()



#%%









#%%












































































