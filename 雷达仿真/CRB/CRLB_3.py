
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
    eigenvalues, eigvector = np.linalg.eig(Rxx)          # 特征值分解
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
    Y_sub1 = y[:-1,:]
    Y_sub2 = y[1:,:]
    Z_mtx = np.vstack((Y_sub1, Y_sub2))
    R_ZZ = Z_mtx @ Z_mtx.conj().T

    U, s, VH =  np.linalg.svd(R_ZZ)
    Es = U[:, 0][:,None]

    Exy = np.hstack(((Es[:K_sub-1, :], Es[K_sub-1:, :])))
    Exy_conj = Exy.conj().T
    EE = Exy_conj @ Exy
    U, s, VH =  np.linalg.svd(EE)
    EE = VH.conj().T
    E12 = EE[:L, L:]
    E22 = EE[L:,L:]
    Psi = - E12 @ scipy.linalg.inv(E22)
    value, vecs = np.linalg.eig(Psi)
    return value[0]

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

# https://github.com/highskyno1/MIMO_DOA
def DOA_ESPRIT(X, K, N, lamda = 2, d = 1):
    # DOA_ESPRIT 基于旋转不变子空间法实现DOA
    #   x_sig       每个阵元接收到的信号矩阵，阵元数*快拍数
    #   target_len  目标数量
    #   lamda       载波波长
    #   d           阵元间隔
    #   DOA_esp_ml  基于最大似然估计准则得到的估计结果
    #   DOA_esp_tls 基于最小二乘准则得到的估计结果
    N = X.shape[0]
    # 回波子阵列合并
    x_esp = np.vstack((X[:N-1,:], X[1:N, :]))
    #  计算协方差
    R_esp = np.cov(x_esp.conj())
    # 特征分解
    D, W = np.linalg.eig(R_esp.T)
    D1, W1 = np.linalg.eig(R_esp)
    # 获取信号子空间
    # W = np.fliplr(W)
    U_s = W[:,:K]
    # 拆分
    U_s1 = U_s[:N-1,:]
    U_s2 = U_s[N-1:,:]

    ## LS-ESPRIT法
    mat_esp_ml = scipy.linalg.pinv(U_s1) @ U_s2;
    # 获取对角线元素并解算来向角
    DOA_esp_ml = np.angle(np.linalg.eig(mat_esp_ml)[0])
    DOA_esp_ml = np.arcsin(DOA_esp_ml * lamda / 2 / np.pi / d)
    # DOA_esp_ml = np.rad2deg(DOA_esp_ml)

    ## TLS-ESPRIT
    Us12 = np.hstack((U_s1, U_s2))
    U, s, VH = np.linalg.svd(Us12)
    V = VH.conj().T
    ## 提取E12和E22
    E12 = V[:K, K:]
    E22 = V[K:,K:]
    mat_esp_tls = - E12 @ scipy.linalg.inv(E22)
    # 获取对角线元素并解算来向角
    DOA_esp_tls = np.angle(np.linalg.eig(mat_esp_tls)[0])
    DOA_esp_tls = np.arcsin(DOA_esp_tls * lamda / 2 / np.pi / d)
    # DOA_esp_tls = np.rad2deg(DOA_esp_tls);

    return DOA_esp_ml, DOA_esp_tls

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
MseESPRIT_tls = np.zeros(SNRdBs.size)
MseESPRIT_tls1 = np.zeros(SNRdBs.size)
MseESPRIT_ml = np.zeros(SNRdBs.size)
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
        # HX = H @ X
        # sig_power = np.mean(np.abs(HX)**2)
        # noise_var = sig_power * 10**(-snr/10)
        # y = HX + (np.sqrt(noise_var/2) * (np.random.randn(*HX.shape) + 1j * np.random.randn(*HX.shape))).astype(np.complex128)
        y = np.zeros((N, T), dtype = np.complex128)
        for t in range(T):
            tmp = H * X[0, t]
            sig_power = np.mean(np.abs(tmp)**2)
            noise_var = sig_power * 10**(-snr/10)
            y[:,t] = (H * X[0, t] + (np.sqrt(noise_var/2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))).astype(np.complex128)).flatten()

        theta_MUISC     = MUSIC(y)
        MseMUSIC[i]     += np.abs(np.arcsin(theta_MUISC) - np.arcsin(theta))**2

        Rxx = y @ y.T.conjugate() / T
        theta_ESPIRT = ESPRIT(Rxx, 1, N)[0]
        MseESPRIT[i]    += np.abs(theta_ESPIRT - np.arcsin(theta))**2

        psi             = TLS_ESPRIT(y, 1);
        theta_ESPRIT_tls   = np.log(psi)/(1j * np.pi)
        MseESPRIT_tls[i]   += np.abs(np.arcsin(theta_ESPRIT_tls) - np.arcsin(theta))**2

        DOA_esp_ml, DOA_esp_tls = DOA_ESPRIT(y, 1, N, lamda = 2, d = 1)
        MseESPRIT_tls1[i]   += np.abs(DOA_esp_tls[0] - np.arcsin(theta))**2
        MseESPRIT_ml[i]   += np.abs(DOA_esp_ml[0] - np.arcsin(theta))**2
        sigma2  = 10**(-snr/10);
        # 这个sigma2和上面的值是渐进一致的
        # sigma2  = (norm(Y, 'fro')^2-norm(H*X, 'fro')^2)/ (size(Y, 1)*size(Y, 2));
        # X_bar   = kron(X.', eye(N));
        # y       = reshape(Y,[],1);
        # CRLB(i_SNR)     = CRLB(i_SNR) + sigma2/2./ real(-y'*X_bar*a2 + a2'*(X_bar'*X_bar)*H + a1'*(X_bar'*X_bar)*a1);
        CRLB[i] += sigma2/2/np.real((Cst*(X@X.conj().T)))[0,0]

MseMUSIC        = MseMUSIC/Nit;
MseESPRIT       = MseESPRIT/Nit;
MseESPRIT_tls   = MseESPRIT_tls/Nit;
MseESPRIT_tls1  = MseESPRIT_tls1/Nit;
MseESPRIT_ml    = MseESPRIT_ml/Nit;
CRLB            = CRLB/Nit;

colors = plt.cm.jet(np.linspace(0, 1, 6))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)

axs.semilogy(SNRdBs, MseMUSIC , linestyle='-', lw = 2, marker = 's', color = colors[0], markersize = 12,  label = "MUSIC", )
axs.semilogy(SNRdBs, MseESPRIT, linestyle='-', lw = 2, marker = '*', color = colors[1], markersize = 12,  label = "ESPIRT",)
axs.semilogy(SNRdBs, MseESPRIT_tls, linestyle='-', lw = 2, marker = 'd', color = colors[2], markersize = 12,  label = "ESPIRT tls",)
axs.semilogy(SNRdBs, MseESPRIT_ml, linestyle='-', lw = 2, marker = '^', color = colors[3], markersize = 12,  label = "ESPIRT ml",)
axs.semilogy(SNRdBs, MseESPRIT_tls1, linestyle='-', lw = 2, marker = 'v', color = colors[4], markersize = 12,  label = "ESPIRT tls1",)
axs.semilogy(SNRdBs, CRLB,      linestyle='--', lw = 2, marker = 'o', color = colors[-1], markersize = 12, label = "CRLB", )

axs.set_xlabel( "SNR/(dB)",)
axs.set_ylabel('MMSE',)
axs.legend()

plt.show()
plt.close('all')


































