#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 10 12:56:18 2025

@author: jack
"""
import scipy
import numpy as np
# import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# import commpy

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300      # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6    # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22
pi = np.pi

#%%
# def f1(y, a, sigma):
#     return y*(0.5/np.sqrt(2*pi)/sigma*np.exp(-(y - a)**2/2/sigma**2) * (1 - np.log2((1 + np.exp(-2*a/sigma**2*y)))))

# def f2(y, a, sigma):
#     return y*(0.5/np.sqrt(2*pi)/sigma*np.exp(-(y + a)**2/2/sigma**2) * (1 - np.log2((1 + np.exp(2*a*y/sigma**2)))))

# def get_AWGN_capacity(a, sigma):
#     # This function is employed here to verify the correctness of the program that is being writen
#     # f1 = lambda y: y*(0.5/np.sqrt(2*pi)/sigma*np.exp(-(y - a)**2/2/sigma**2) * (1 - np.log2((1 + np.exp(-2*a/sigma**2*y)))))

#     C0 = scipy.integrate.quad(f1, -30, 1000, args = (a, sigma))[0]

#     # f2 = lambda y: y*(0.5/np.sqrt(2*pi)/sigma*np.exp(-(y + a)**2/2/sigma**2) * (1 - np.log2((1 + np.exp(2*a*y/sigma**2)))))
#     C1 = scipy.integrate.quad(f2, -1000, 30, args = (a, sigma))[0]

#     C = C0 + C1;
#     return C

# snr = np.arange(-10, 22,1)
# R = 1/2
# sigma = 1/np.sqrt(2 * R) * 10**(-snr/20)
# n = 8
# N = 2**n
# C_AWGN = np.zeros(snr.size)
# for i in range(snr.size):
#     C_AWGN[i] = get_AWGN_capacity(1, sigma[i])

# ##### plot
# fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

# axs.plot(snr,C_AWGN, color = 'b', label = 'capacity boundary')
# axs.set_xlabel(r'$\mathrm{E_b}/\mathrm{N_0}$(dB)',)
# axs.set_ylabel('Spectral Efficiency (Bit/s/Hz)',)

# plt.show()
# plt.close()


#%%
## 发射端未知CSI时信道容量
def mimo_capacity_noCIS(Nr, Nt, SNR, trail = 3000):
    SNR_D = 10**(SNR/10.0) # SNR in decimal
    C = np.zeros(trail)

    for i in range(trail):
        H = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt))/np.sqrt(2)
        U, s, VH = np.linalg.svd(H)

        C_temp = np.zeros(s.size)
        for j in range(s.size):
            C_temp[j] = np.log2(1 + s[j]**2*SNR_D/Nt);

        C[i] = np.sum(C_temp)

    cap = np.mean(C)
    return cap

## 发射端已知CSI时信道容量
def mimo_capacity_wCIS(Nr, Nt, SNR, trail = 3000):
    SNR_D = 10**(SNR/10.0) # SNR in decimal
    C = np.zeros(trail)

    for i in range(trail):
        H = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt))/np.sqrt(2)
        U, s, VH = np.linalg.svd(H)

        C_temp = np.zeros(s.size)
        for j in range(s.size):
            C_temp[j] = np.log2(1 + s[j]**2*SNR_D/Nt);

        C[i] = np.sum(C_temp)

    cap = np.mean(C)
    return cap

def awgn_capacity(SNRdB):
    SNR_D = 10**(0.1*SNRdB)
    cap = np.log2(1 + SNR_D)
    return cap

def ralychannel(SNRdB):
    # snrdB = np.arange(-10, 30, 1/2)
    h = (np.random.randn(1, 10000) + 1j * np.random.randn(1, 10000))/np.sqrt(2)
    sigma_z = 1
    snr = 10**(SNRdB/10)
    P = (sigma_z**2) * snr / np.mean(np.abs(h)**2)

    # C_awgn = np.log2(1 + np.mean(np.abs(h)**2) * P / (sigma_z**2))
    C_fading = np.mean(np.log2(1 + (np.abs(h)**2).T @ P.reshape(1, -1) / sigma_z**2 ), axis = 0)
    return C_fading

#%% 接收天线变化
Nt = 4












#%%










#%%

























































































































