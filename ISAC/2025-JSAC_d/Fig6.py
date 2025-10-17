#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 11:51:24 2025

@author: Junjie Chen
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


#### Sigma_S
# K = 20
# delta_k = 1 # np.random.rand(K)
# theta_k_deg = np.random.uniform(-90, 90, K)
# theta_k_rad = np.deg2rad(theta_k_deg)
# # 初始化 Sigma_s
# Sigma_S = np.zeros((N, N), dtype=complex)
# for k in range(K):
#     a_theta =  (1/np.sqrt(N)) * np.exp(1j * np.pi * np.arange(N) * np.sin(theta_k_rad[k]))
#     Sigma_S += delta_k**2 * np.outer(a_theta, a_theta.conj())
# Lambda_s, U_s = np.linalg.eig(Sigma_S)
# Sigma_S = np.diag(np.abs(Lambda_s))
# Sigma_S_inv = np.linalg.inv(Sigma_S)

lamba_s = 1 + np.random.rand(N)
Sigma_S = np.eye(N) * lamba_s
Lambda_s, U_s = np.linalg.eig(Sigma_S)
Sigma_S_inv = np.linalg.inv(Sigma_S)

#### Sigma_C
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

SNR_C = [0, 10, 20]
SNR_S = np.arange(-10, 20, 5)

L = 11
alpha_lst = np.linspace(0, 1, L)
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
            Dc, _,  _, _ ,_ = reverse_waterfill_D(IcR.real, np.abs(lambaR))
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
            Dc, _,  _, _ ,_ = reverse_waterfill_D(IcR.real, np.abs(lambaR))
            D_tol = Ds.real +  Mc * Dc
            Distor_ary3[ii, jj, kk] = D_tol
Distor_ary2 = np.min(Distor_ary3, axis = -1) / N

#%% Algorithm 3 Modified Gradient Projection Algorithm
def GradProject(p_bar, P_T, tol=1e-10, max_iter=1000):
    """
    min_{x}: || x - p ||_2,
    s.t. sum_{i} x_i <= Pt
         x_i >=0,
    x = {x_1, x_2, x_N}

    参数:
        p_bar: 当前点, 形状 (n,)
        P_T: 总功率约束
        tol: 容差
        max_iter: 最大迭代次数

    返回:
        x: 投影点
        lambda_val: 使用的 lambda 值
    """
    n = len(p_bar)
    a = p_bar.copy()

    # 步骤1: 计算 sum max(0, a_i)
    S = np.sum(np.maximum(0, a))

    # 步骤2: 如果 S <= P_T, 直接返回
    if S <= P_T:
        x = np.maximum(0, a)
        return x, 0.0

    # 步骤3: 否则，二分搜索 lambda
    low = 0.0
    high = np.max(a)
    lambda_val = 0.0

    for _ in range(max_iter):
        lambda_val = (low + high) / 2.0
        x_temp = np.maximum(0, a - lambda_val)
        S_temp = np.sum(x_temp)

        if abs(S_temp - P_T) < tol:
            break
        elif S_temp > P_T:
            low = lambda_val
        else:
            high = lambda_val

    x = np.maximum(0, a - lambda_val)
    return x, lambda_val


def armijo_search( grad_f, x, d, K, nozero_idx, Lambda_c, Lambda_s, sigma_c2, sigma_s2, alpha0 = 1.0, rho = 0.5, c1 = 1e-3, max_iter = 1000):

    alpha = alpha0
    f_x, _, _ = fun_tildeHp(K, x, nozero_idx, Lambda_c, Lambda_s, sigma_c2, sigma_s2, )
    gradient_direction = np.dot(grad_f, d)  # 梯度与方向d的内积
    for i in range(max_iter):
        x_new = x + alpha * d
        f_new, _, _ = fun_tildeHp(K, x_new, nozero_idx, Lambda_c, Lambda_s, sigma_c2, sigma_s2, )
        if f_new <= f_x + c1 * alpha * gradient_direction:
            return alpha
        else:
            alpha = rho * alpha

    return alpha


def fun_tildeHp(K, p_new, nozero_idx, Lambda_c, Lambda_s, sigma_c2, sigma_s2, ):
    gip = p_new * Lambda_s**2 / (sigma_s2 + Lambda_s * p_new)
    IcP = np.sum(np.log(Lambda_c * p_new / sigma_c2 + 1))
    tilde_fp = (np.sum(np.log(gip[nozero_idx])) - IcP/Ms) / K
    tilde_hp = K * np.exp(tilde_fp) - np.sum( gip[nozero_idx])
    return tilde_hp, gip, tilde_fp

eps = 1e-10
beta = 0.1
alpha = 0.1
Distor_MGP = np.zeros((len(SNR_C), SNR_S.size))

for ii, snr_c in enumerate(SNR_C):
    sigma_c2 = Pt * 10.0**(-snr_c / 10.0)
    for jj, snr_s in enumerate(SNR_S):
        sigma_s2 = Pt * 10.0**(-snr_s / 10.0)
        p_old = np.random.rand(N)
        p_old = Pt * p_old/p_old.sum()

        hp_old = -10
        hp_new = 0
        count = 0
        while np.abs(hp_new - hp_old) > eps:
            count += 1
            IcP = np.sum(np.log(Lambda_c * p_old / sigma_c2 + 1))
            gip = p_old * Lambda_s**2 / (sigma_s2 + Lambda_s * p_old)
            Dc, xi, K, nozero_idx, zero_idx = reverse_waterfill_D(IcP/Ms, gip)   # Calculate K^(l), ξ^(l) through reverse water-filling
            grad_hp = np.zeros(N)
            p_nozero = p_old[nozero_idx]
            lamba_s_nozero = Lambda_s[nozero_idx]
            lamba_c_nozero = Lambda_c[nozero_idx]
            grad_fp_nozero = (sigma_s2/(sigma_s2 * p_nozero + lamba_s_nozero * p_nozero**2) - lamba_c_nozero/Ms * (sigma_c2 + lamba_c_nozero * p_nozero)) / K   # Eq.(50.5)
            tilde_fp = (np.sum(np.log(gip[nozero_idx])) - IcP/Ms) / K                                                                                           # Eq.(48)
            grad_hp[nozero_idx] = K * np.exp(tilde_fp) * grad_fp_nozero - lamba_s_nozero**2 * sigma_s2 / (sigma_s2 + lamba_s_nozero * p_nozero)**2              # Eq.(51)
            # print(grad_hp)
            p_now = p_old - beta * grad_hp
            p_proj, _ = GradProject(p_now, Pt)
            alpha = armijo_search(grad_hp, p_old, p_proj - p_old,  K, nozero_idx, Lambda_c, Lambda_s, sigma_c2, sigma_s2) # Find stepsize alpha based on the Armijo condition;
            p_new = p_old + alpha * (p_proj - p_old)
            ## update h(p)
            hp_now, gip, tilde_fp = fun_tildeHp(K, p_new, nozero_idx, Lambda_c, Lambda_s, sigma_c2, sigma_s2, )
            hp_old = hp_new
            hp_new = hp_now
            p_old = p_new
        print(f"{snr_c} -> {snr_s} -> {count}")
        Distor_MGP[ii, jj] = Ms * ( Lambda_s.sum() - gip[nozero_idx].sum() + K * np.exp(tilde_fp) )
Distor_MGP = Distor_MGP / N

# ##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
colors = plt.cm.jet(np.linspace(0, 1, 6))

markers = ['d', 'o', 's', 'v', '^', '+']
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
for idx, snrc in enumerate(SNR_C):
    # axs.plot(SNR_S, Distor_ary1[idx,:], ls = '-', lw = 2, marker = markers[idx], color=colors[idx], label = "$SNR_c = $" + f"{snrc} dB, L = 11")
    # axs.plot(SNR_S, Distor_ary2[idx,:], ls = '--', lw = 2, marker = markers[idx], color=colors[idx], label = "$SNR_c = $" + f"{snrc} dB, L = 3")
    axs.plot(SNR_S, Distor_MGP[idx,:], ls = '-.', lw = 2, marker = markers[idx], color=colors[idx], label = "$SNR_c = $" + f"{snrc} dB, MGP")

axs.set_xlabel('Sensing Channel SNR (dB)', )
axs.set_ylabel('Average CAS Distortion', )

axs.set_title("Specific case", fontsize = 25)

legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black',  )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

out_fig = plt.gcf()
plt.tight_layout()
plt.show()


#%%









#%%












































































