#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:28:29 2025

@author: jack

https://github.com/LiZhuoRan0/CRLB-demo


"""


import numpy as np
import matplotlib.pyplot as plt
import scipy

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
def genSteerVector(theta_deg, N, d, lamda):
    n = np.arange(N)[:,None]
    at = np.exp(1j * 2 * np.pi * d * np.sin(theta) * n / lamda)
    return at

def genPartialSteerVector(theta, N, d, lamda, flag):
    n = np.arange(N)[:,None]
    if flag == 1:
        at = (1j * 2 * np.pi * d * n * np.cos(theta) / lamda) * np.exp(1j * 2 * np.pi * d * np.sin(theta) * n / lamda)
    else:
        at = - (2 * np.pi * d * n * np.cos(theta) / lamda)**2 * np.exp(1j * 2 * np.pi * d * np.sin(theta) * n / lamda) - (1j * 2 * np.pi * d * n * np.sin(theta) / lamda) * np.exp(1j * 2 * np.pi * d * np.sin(theta) * n / lamda)
    return at

def MUSIC(y, K = 1):
    # y: receive signal;
    # K: num of target;
    N, T = y.shape
    Range = 1
    thre = 1e-12
    center = 0
    Nit = 20
    Rxx = y @ y.T.conjugate() / T
    # 特征值分解
    eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
    idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
    eigvector = eigvector[:, idx]
    eigvector = eigvector[:,::-1]                         # 对应特征矢量排序
    Un = eigvector[:, K:N]

    return

def TLS_ESPRIT(y):

    return

#%% 参数设置
N               =           64                 # 基站处天线
fc              =           100e9              #100GHz
lambda_c        =           3e8/fc
d               =           lambda_c/2
T               =           100
Nit             =           100
SNRdBs          =           np.arange(-10, 22, 2)
#%%
theta   = np.deg2rad(30)  #  np.pi/6

H   = genSteerVector(theta, N, d, lambda_c)
Y   = np.zeros((N, T))
A   = np.eye(N)
a1  = genPartialSteerVector(theta, N, d, lambda_c, 1)
a2  = genPartialSteerVector(theta, N, d, lambda_c, 2);
MseMUSIC  = np.zeros(SNRdBs.size)
MseESPRIT = np.zeros(SNRdBs.size)
CRLB      = np.zeros(SNRdBs.size)
D = a1
Cst = D.conjugate().T  @ (np.eye(N) - H @ scipy.linalg.inv(H.conj().T@H)@H.conj().T) @ D

for i, snr in enumerate(SNRdBs):
    print('==============================================')
    print(f'SNR        = {snr}')
    for it in range(Nit):
        if it % 10 == 0:
            print(f"{it+1}/{Nit}")
        X = np.sqrt(1./2) * (np.random.randn(1, T) + 1j * np.random.randn(1, T))
        HX = H @ X
        sig_power = np.mean(np.abs(HX)**2)
        noise_var = sig_power * 10**(-snr/10)
        y = HX + np.sqrt(noise_var/2) * (np.random.randn(*HX.shape) + 1j * np.random.randn(*HX.shape))

        theta_MUISC     = MUSIC(Y);
        psi             = TLS_ESPRIT(Y, 1);
        theta__ESPRIT   = np.log(psi)/(1j * np.pi);
        MseMUSIC[i]     += np.abs(theta_MUISC - theta)**2
        MseESPRIT[i]    += np.abs(theta_MUISC - theta)**2

        sigma2  = 10**(-SNRdBs[i]/10);
        # 这个sigma2和上面的值是渐进一致的
        # sigma2  = (norm(Y, 'fro')^2-norm(H*X, 'fro')^2)/ (size(Y, 1)*size(Y, 2));
        # X_bar   = kron(X.', eye(N));
        # y       = reshape(Y,[],1);
        # CRLB(i_SNR)     = CRLB(i_SNR) + sigma2/2./ real(-y'*X_bar*a2 + a2'*(X_bar'*X_bar)*H + a1'*(X_bar'*X_bar)*a1);
        CRLB[i] = CRLB[i] + sigma2/2/np.real((Cst*(X@X.conj().T)))

# MseMUSIC     = MseMUSIC/Nit;
# MseESPRIT    = MseESPRIT/Nit;
CRLB         = CRLB/Nit;


colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)

axs.semilogy(SNRdBs, MseMUSIC , linestyle='-', lw = 2, marker = 'o', color = colors[0], markersize = 12,  label = "MUSIC", )
axs.semilogy(SNRdBs, MseESPRIT, linestyle='-', lw = 2, marker = '*', color=colors[1], markersize = 12,  label = "ESPIRT",)
axs.semilogy(SNRdBs, CRLB,      linestyle='--', lw = 2, color = 'k', markersize = 12, label = "CRLB", )

axs.set_xlabel( "SNR/(dB)",)
axs.set_ylabel('MMSE',)
axs.legend()

plt.show()
plt.close('all')










