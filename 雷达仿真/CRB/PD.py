#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 14:53:53 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
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

#%% https://deepinout.com/numpy/numpy-questions/109_numpy_how_can_i_compute_the_null_spacekernel_x_mx_0_of_a_sparse_matrix_in_python.html#google_vignette
import numpy as np
# 稀疏矩阵的空间/核
# 创建一个随机稀疏矩阵
M = scipy.sparse.random(6, 6, density=0.5, format='lil', random_state=0)

# 计算随机稀疏矩阵的SVD
U, s, Vt = scipy.sparse.linalg.svds(M, k = 5)

# 计算空间/核
null_space = Vt.T[:, -1]

# 输出结果
print("稀疏矩阵M：\n", M.todense())
print("空间/核：\n", null_space)

#%% https://blog.51cto.com/u_16213370/11892718
import numpy as np
# from scipy.linalg import null_space

# 定义一个矩阵 A
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
# A = np.random.randn(2, 8) + 1j*np.random.randn(2, 8)
# 计算零空间
null_space_A = scipy.linalg.null_space(A)

print("矩阵 A 的零空间:")
print(null_space_A)
print(f"A@null_space_A = {A@null_space_A}")

# 计算矩阵的秩
rank_A = np.linalg.matrix_rank(A)
print("矩阵 A 的秩:", rank_A)

#%% https://www.zhihu.com/question/294214797
# 定义一个矩阵 A
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
def null(A, eps = 1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s < eps)
    null_space = np.compress(null_mask, vh, axis = 0)
    return np.transpose(null_space)

# 计算零空间
null_space_A = null(A)
print("矩阵 A 的零空间:")
print(null_space_A)
print(f"A@null_space_A = {A@null_space_A}")

#%% Define Parameters
from scipy.linalg import null_space
from scipy.stats import ncx2


pi = np.pi
# Speed of light
c = 3*10**8
#  Nr Comm Receivers
Nr = 2
#  Mt Radar Transmiters
Mt = 8
#  Mr Radar Receivers
Mr = Mt
#  Radial velocity of 2000 m/s
v_r = 2000
#  Radar reference point
r_0 = 500*10**3
#  Carrier frequency 3.5GHz
f_c = 3.5*1e9  #  Angular carrier frequency
omega_c = 2*pi*f_c
lamba = (2*pi*c)/omega_c
theta = 30 * pi / 180.0

# Steering vector and Transmit Signal Correlation Matrix
# Transmit/Receive Steering vector (Mt x 1)
a = np.exp(-1j * pi * np.sin(theta) * np.arange(Mt))[:,None]
# Transmit Correlation Matrix (Mt x Mt) for Orthonormal Waveforms
Rs = np.eye(Mt)
# Define SNR for ROC (Reciever Operating Characteristics)
SNR_db = np.arange(-8, 11)
SNR_mag = 10**(SNR_db/10)
# Probability of false alarm values
P_FA = np.array([1e-1, 1e-2])
# Monte-Carlo iterations
MC_iter = 10
BS = 5

Pd_orthog_cat = np.zeros((MC_iter, BS, SNR_mag.size, P_FA.size), dtype = complex)
Pd_NSP_cat = np.zeros((MC_iter, BS, SNR_mag.size, P_FA.size), dtype = complex)

for i in range(MC_iter):
    rho_orthog = np.zeros(BS)
    rho_NSP = np.zeros(BS)
    BS_channels = np.zeros((BS, Nr, Mt), dtype = complex)
    Proj_matrix = np.zeros((BS, Mt, Mt), dtype = complex)
    Rs_null = np.zeros((BS, Mt, Mt), dtype = complex)

    Pd_orthog_cell = np.zeros((BS, SNR_mag.size, P_FA.size))
    Pd_NSP_cell = np.zeros((BS, SNR_mag.size, P_FA.size))

    for b in range(BS):
        BS_channels[b,:,:] = (np.random.randn(Nr, Mt) + 1j*np.random.randn(Nr, Mt)) / np.sqrt(2)
        Proj_matrix[b,:,:] = null_space(BS_channels[b,:,:]) @ null_space(BS_channels[b,:,:]).conjugate().T
        Rs_null[b,:,:]     = Proj_matrix[b,:,:] @ Rs @ (Proj_matrix[b,:,:].conjugate().T)

        Pd_orthog = np.zeros((SNR_mag.size, P_FA.size))
        Pd_NSP = np.zeros((SNR_mag.size, P_FA.size))
        for z, snr in enumerate(SNR_mag):
            rho_orthog[b] = SNR_mag[z] * (np.abs(a.conjugate().T @ Rs @ a)[0,0])**2
            rho_NSP[b] = SNR_mag[z] * (np.abs(a.conjugate().T @ Rs_null[b] @ a)[0,0])**2
            # Creates threshold values for a desired Pfa for an inverse central-chi-square w/2 degrees of freedom
            # delta = ncx2.ppf(1 - P_FA, df = 2, nc = 0)
            ## or
            delta = scipy.stats.chi2.ppf(1 - P_FA, df = 2)
            # rows = SNR, cols = P_FA, ncx2cdf = Noncentral chi -square cumulative distribution function
            Pd_orthog[z,:] = 1 - ncx2.cdf(delta, 2, rho_orthog[b])
            Pd_NSP[z,:] = 1 -  ncx2.cdf(delta, 2, rho_NSP[b])
        Pd_orthog_cell[b] = Pd_orthog
        Pd_NSP_cell[b] = Pd_NSP
    Pd_orthog_cat[i,...] = Pd_orthog_cell
    Pd_NSP_cat[i,...] = Pd_NSP_cell

Pd_orthog_cat_mean = np.mean(Pd_orthog_cat, axis = 0)
Pd_NSP_cat_mean = np.mean(Pd_NSP_cell, axis = 0)

colors = plt.cm.jet(np.linspace(0, 1, BS))
for i, pfa in enumerate(P_FA):
    # print(i, pfa)
    fig, axs = plt.subplots(1, 1, figsize = (10, 8))
    for k in range(BS):
        axs.plot(SNR_db, Pd_orthog_cat_mean[k,:,i] , color = colors[k], linestyle='-', lw = 2, label = f"P_D for NSP Waveforms to BS {k}", )

    axs.set_xlabel( "SNR/(dB)",)
    axs.set_ylabel('P_D',)
    axs.legend()

    plt.show()
    plt.close('all')






























































