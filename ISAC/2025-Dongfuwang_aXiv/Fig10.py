#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 15:20:25 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import commpy
import cvxpy as cpy
from scipy.linalg import sqrtm, inv


from Tools import freqDomainView

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

def sensing_distortion(Ms, R_x, sigma_s2, Sigma_s_inv):
    """计算感知失真 D_s(R_x)"""
    inner_matrix = (1/sigma_s2) * R_x + Sigma_s_inv
    return Ms * np.trace(np.linalg.inv(inner_matrix))

def R_s_matrix(Sigma_s, R_x, sigma_s2, Sigma_s_inv):
    """计算 R_s 矩阵"""
    inner_matrix = (1/sigma_s2) * R_x + Sigma_s_inv
    return Sigma_s - np.linalg.inv(inner_matrix)

def channel_capacity(Mc, R_x, H_c, sigma_c2):
    """计算信道容量 C(R_x)"""
    I_mc = np.eye(Mc)
    inner_matrix = (1/sigma_c2) * H_c @ R_x @ H_c.conj().T + I_mc
    return np.log(np.linalg.det(inner_matrix))

def generate_hermitian_matrix(N):
    """
    生成维度为N的随机Hermitian矩阵
    """
    # 生成随机复数矩阵
    A = np.random.randn(N, N) #+ 1j * np.random.randn(N, N)

    # 构造Hermitian矩阵：A + A^H
    H = (A + A.conj().T) / 2

    return H

def generate_hermitian_from_eigenvalues(N):
    """
    通过指定特征值生成Hermitian矩阵
    """
    # 生成随机实数特征值（Hermitian矩阵的特征值都是实数）
    eigenvalues = np.random.randn(N)

    # 生成随机酉矩阵
    U = np.random.randn(N, N) #+ 1j * np.random.randn(N, N)
    U, _ = np.linalg.qr(U)  # QR分解得到酉矩阵

    # 构造Hermitian矩阵：U * diag(eigenvalues) * U^H
    H = U @ np.diag(eigenvalues) @ U.conj().T

    return H
#%%
Nt = 10
Mc = 5
Ms = 2
Iter = 200               # 信道遍历次数
max_iterations = 50     # SCA 迭代次数
I_mc = np.eye(Mc)
I_nt = np.eye(Nt)
Pt = 1

tolerance = 1e-6
SNRcom = [-5, 0, 5]
SNRsen = np.arange(0, 22, 2)
Sigma_s = np.diag(np.random.rand(Nt)*2)          #  generate_hermitian_matrix(Nt),   np.diag(np.random.rand(Nt)*2)
# 预先计算常数矩阵
Sigma_s_inv = np.linalg.inv(Sigma_s)

Distor_optimal = np.zeros((len(SNRcom), len(SNRsen), Iter))
res_d = {}

R0 = Pt/Nt * I_nt

for ii, snrcom in enumerate(SNRcom):
    sigma_c2 = 10**(-snrcom/10.0)
    for jj, snrsen in enumerate(SNRsen):
        sigma_s2 = 10**(-snrsen/10.0)
        for kk, It in enumerate(range(Iter)):
            key = f"{snrcom}-{snrsen}-{It}"
            print(f"snrcom:{snrcom} -> snrsen:{snrsen} -> It:{It}")
            Hc = np.random.randn(Mc, Nt) + 1j * np.random.randn(Mc, Nt)

            R_x_prev = R0
            obj_values = []
            for it in range(max_iterations):
                # 计算常数矩阵 P
                P = inv((1/sigma_s2) * R_x_prev + inv(Sigma_s))
                # 定义优化变量
                Rx = cpy.Variable((Nt, Nt), symmetric=True)
                D = cpy.Variable((Nt, Nt), symmetric=True)
                # 计算线性化近似 ~R_s
                R_s_tilde = Sigma_s - P + (1/sigma_s2) * P @ (Rx - R_x_prev) @ P

                # 计算 f(R_x) 的线性化近似
                Sigma_minus_P = Sigma_s - P
                Sigma_minus_P_inv = np.linalg.inv(Sigma_minus_P)
                f_Rx = np.log(np.linalg.det(Sigma_minus_P)) + (1/sigma_s2) * cpy.trace(Sigma_minus_P_inv @ P @ (Rx - R_x_prev) @ P)

                # 目标函数 - 使用矩阵分式方法避免直接求逆,对于 Tr((A*R_x + B)^{-1})，我们使用辅助变量和矩阵不等式
                A = (1/sigma_s2) * I_nt
                B = Sigma_s_inv

                # 方法1：使用Schur补引理, 引入辅助变量 Z，约束 [A*R_x + B, I; I, Z] >= 0,然后最小化 trace(Z)
                Z = cpy.Variable((Nt, Nt), symmetric=True)
                schur_constraint = cpy.bmat([
                                            [A @ Rx + B, I_nt],
                                            [I_nt, Z]
                                           ]) >> 0

                objective = Ms * cpy.trace(Z) + cpy.trace(D)
                #### 约束条件
                inner_matrix_capacity = (1/sigma_c2) * Hc @ Rx @ Hc.conj().T + I_mc
                capacity_constraint = cpy.log_det((1/sigma_c2) * Hc @ Rx @ Hc.conj().T + I_mc) - Ms * (f_Rx - cpy.log_det(D)) >= 0

                constraints = [
                                schur_constraint,
                                capacity_constraint,
                                R_s_tilde - D >> 0,
                                Rx >> 0,
                                cpy.trace(Rx) <= Pt
                                ]

                # 求解问题
                prob = cpy.Problem(cpy.Minimize(objective), constraints)
                try:
                    prob.solve(solver=cpy.MOSEK, verbose=False,)
                except Exception as e:
                    print(f"  Solver error at iteration {it}: {e}")
                    break

                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    print(f"  Iteration {it}: Problem status - {prob.status}")
                    if it == 0:
                        R_x_opt =  R_x_prev
                    break

                R_x_opt = Rx.value
                if R_x_opt is None:
                    print(f"  Iteration {it}: No solution found")
                    break

                obj_value = sensing_distortion(Ms, R_x_opt, sigma_s2, Sigma_s_inv) + np.trace(D.value)
                obj_values.append(obj_value)

                print(f"  Iteration {it}: Objective = {obj_value:.6f}")

                if it > 0:
                    relative_change = abs(obj_value - obj_values[-2]) / (abs(obj_values[-2]) + 1e-8)
                    if relative_change < tolerance:
                        print(f"Converged after {it+1} iterations")
                        break

                R_x_prev = R_x_opt
            Distor_optimal[ii, jj, kk] = obj_values[-1]
            res_d[key] = obj_values

Distor_optim_avg = np.mean(Distor_optimal, axis = (2)) / (Ms * Nt)

colors = plt.cm.jet(np.linspace(0, 1, 5))

fig, axs = plt.subplots(1, 1, figsize=(10, 6))
axs.plot(SNRsen, Distor_optim_avg[0], '--', lw = 2, marker = '*', color=colors[0], label = "Comm SNR = -5 dB")
axs.plot(SNRsen, Distor_optim_avg[1], '--', lw = 2, marker = 'd', color=colors[1], label = "Comm SNR = 0 dB")
axs.plot(SNRsen, Distor_optim_avg[2], '--', lw = 2, marker = 'o', color=colors[2], label = "Comm SNR = 5 dB")

axs.set_xlabel('Sensing SNR (dB)', )
axs.set_ylabel('Average Distortion', )

legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black',  )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

out_fig = plt.gcf()
plt.tight_layout()
plt.show()




#%%






#%%






#%%






#%%












