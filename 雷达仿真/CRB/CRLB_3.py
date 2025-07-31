
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
    at = np.zeros_like(n, dtype = np.complex128)
    at = np.exp(1j * 2 * np.pi * d * theta * n / lamda)
    return at

def genPartialSteerVector(theta, N, d, lamda, flag):
    n = np.arange(N)[:,None]
    at = np.zeros_like(n, dtype = np.complex128)
    if flag == 1:
        at = (1j * 2 * np.pi * d * n * np.cos(np.arcsin(theta)) / lamda) * np.exp(1j * 2 * np.pi * d * theta * n / lamda)
    else:
        at = - (2 * np.pi * d * n * np.cos(np.arcsin(theta)) / lamda)**2 * np.exp(1j * 2 * np.pi * d * theta * n / lamda) - (1j * 2 * np.pi * d * n * theta / lamda) * np.exp(1j * 2 * np.pi * d * theta * n / lamda)
    return at

def MUSIC(y, K = 1):
    # y: receive signal;
    # K: num of target;
    N, T = y.shape
    Range = 1
    thre = 1e-12
    center = 0
    nit = 20
    Rxx = y @ y.T.conjugate()
    # 特征值分解
    eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
    idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
    eigvector = eigvector[:, idx]
    eigvector = eigvector[:,::-1]                         # 对应特征矢量排序
    Un = eigvector[:, K:N]
    UnUnH = Un @ Un.T.conjugate()
    while Range > thre:
        # print(f"center = {center}, range = {Range}")
        theta_1 = np.linspace(center - Range, center + Range, nit, dtype = np.float128)
        P = np.zeros(nit, dtype = np.float128)
        for i, ang in enumerate(theta_1):
            a = np.zeros(N, dtype = np.complex128)
            a = np.exp(1j * np.pi * np.arange(N) * ang).reshape(-1, 1)
            P[i] = 1/np.abs(a.T.conjugate() @ UnUnH @ a)[0,0]
        Idx = np.argmax(np.abs(P))
        center = center - Range + 2*Range * Idx/(nit-1)
        Range /= 10
        # print(f"Idx = {Idx}")
    return center

def TLS_ESPRIT(y, L = 1):
    K_sub = y.shape[0]
    Y_sub1 = Y_ll_bar[:-1,:]
    Y_sub2 = Y_ll_bar[1:-1,:]
    Z_mtx = [Y_sub1; Y_sub2]
    R_ZZ = Z_mtx @ Z_mtx.conj().T
    return

def ESPRIT(Rxx, K, N):
    # 特征值分解
    D, U = np.linalg.eigh(Rxx)             # 特征值分解
    idx = np.argsort(D)                    # 将特征值排序 从小到大
    U = U[:, idx]
    U = U[:,::-1]                          # 对应特征矢量排序
    Us = U[:, 0:K]

    ## 角度估计
    Ux = Us[0:K, :]
    Uy = Us[1:K+1, :]

    # ## 方法一:最小二乘法
    # Psi = np.linalg.inv(Ux)@Uy
    # Psi = np.linalg.solve(Ux,Uy)    # or Ux\Uy

    ## 方法二：完全最小二乘法
    Uxy = np.hstack([Ux,Uy])
    Uxy = Uxy.T.conjugate() @ Uxy
    eigenvalues, eigvector = np.linalg.eigh(Uxy)          # 特征值分解
    idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
    eigvector = eigvector[:, idx]
    eigvector = eigvector[:,::-1]                          # 对应特征矢量排序
    F0 = eigvector[0:K, K:2*K]
    F1 = eigvector[K:2*K, K:2*K]
    Psi = -F0 @ np.linalg.inv(F1)

    # 特征值分解
    D, U = np.linalg.eig(Psi)          # 特征值分解
    Theta = np.arcsin(np.angle(D)/np.pi)
    Theta = np.sort(Theta)
    return Theta

#%% 参数设置
N               =           64                 # 基站处天线
fc              =           100e9              # 100GHz
lambda_c        =           3e8/fc
d               =           lambda_c/2
T               =           100
Nit             =           200
SNRdBs          =           np.arange(-10, 22, 5)
#%%
theta           = 0.5     # sin(pi/6)

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
        if (it+1) % 10 == 0:
            print(f"{it+1}/{Nit}")
        X = (np.sqrt(1./2) * (np.random.randn(1, T) + 1j * np.random.randn(1, T))).astype(np.complex128)
        HX = H @ X
        sig_power = np.mean(np.abs(HX)**2)
        noise_var = sig_power * 10**(-snr/10)
        y = HX + (np.sqrt(noise_var/2) * (np.random.randn(*HX.shape) + 1j * np.random.randn(*HX.shape))).astype(np.complex128)

        theta_MUISC     = MUSIC(y)
        Rxx = y @ y.T.conjugate() / T
        theta_ESPIRT = ESPRIT(Rxx, 1, N)[0]
        # psi             = TLS_ESPRIT(y, 1);
        # theta__ESPRIT   = np.log(psi)/(1j * np.pi);
        MseMUSIC[i]     += np.abs(np.arcsin(theta_MUISC) - np.arcsin(theta))**2
        MseESPRIT[i]    += np.abs(theta_ESPIRT - np.arcsin(theta))**2

        sigma2  = 10**(-SNRdBs[i]/10);
        # 这个sigma2和上面的值是渐进一致的
        # sigma2  = (norm(Y, 'fro')^2-norm(H*X, 'fro')^2)/ (size(Y, 1)*size(Y, 2));
        # X_bar   = kron(X.', eye(N));
        # y       = reshape(Y,[],1);
        # CRLB(i_SNR)     = CRLB(i_SNR) + sigma2/2./ real(-y'*X_bar*a2 + a2'*(X_bar'*X_bar)*H + a1'*(X_bar'*X_bar)*a1);
        CRLB[i] += sigma2/2/np.real((Cst*(X@X.conj().T)))[0,0]

MseMUSIC     = MseMUSIC/Nit;
MseESPRIT    = MseESPRIT/Nit;
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










