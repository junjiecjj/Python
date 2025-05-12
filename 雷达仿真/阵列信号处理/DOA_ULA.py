#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:55:49 2025

@author: jack

https://github.com/chenhui07c8/Radio_Localization?tab=readme-ov-file
"""


import numpy as np
import scipy
import matplotlib.pyplot as plt


# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16         # 设置 y 轴刻度字体大小
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
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.labelspacing'] = 0.2
filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

#%%%%%%%% Uniform Linear Array %%%%%%%%
# create a manifold vector
def steering_vector(k, N):
    n = np.arange(N)
    return np.exp(-1j * np.pi * np.sin(k) * n)

def MUSIC(Rxx, K, N):
    # 特征值分解
    eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
    idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
    eigvector = eigvector[:, idx]
    eigvector = eigvector[:,::-1]                         # 对应特征矢量排序
    Un = eigvector[:, K:N]

    # Un = eigvector
    UnUnH = Un @ Un.T.conjugate()
    Thetalst = np.arange(-90, 90.1, 0.5)
    angle = np.deg2rad(Thetalst)
    Pmusic = np.zeros(angle.size)
    for i, ang in enumerate(angle):
        a = np.exp(-1j * np.pi * np.arange(N) * np.sin(ang)).reshape(-1, 1)
        Pmusic[i] = 1/np.abs(a.T.conjugate() @ UnUnH @ a)[0,0]

    Pmusic = np.abs(Pmusic) / np.abs(Pmusic).max()
    Pmusic = 10 * np.log10(Pmusic)
    peaks, _ =  scipy.signal.find_peaks(Pmusic, height = -10, distance = 10)

    angle_est = Thetalst[peaks]

    return Thetalst, Pmusic, angle_est, peaks

def MUSIC1(Rxx, K, N):
    # Eigenvalue Decomposition
    eigvals, eigvecs = np.linalg.eigh(Rxx)
    U_n = eigvecs[:, :-K]  # noise sub-space
    UnUnH = U_n @ U_n.conj().T
    # MUSIC pseudo-spectrum
    Thetalst = np.arange(-90, 90.1, 0.5)
    k_scan = np.deg2rad(Thetalst)
    P_music = np.zeros_like(k_scan, dtype = float)

    for i, k in enumerate(k_scan):
        a_k = steering_vector(k, N)
        P_music[i] = 1 / np.abs(a_k.conj().T @ UnUnH @ a_k)

    # normalize
    P_music = np.abs(P_music) / np.abs(P_music).max()
    P_music = 10 * np.log10(P_music)
    peaks, _ =  scipy.signal.find_peaks(P_music, height=-10, distance = 10)
    angle_est = Thetalst[peaks]

    return Thetalst, P_music, angle_est, peaks

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
    Theta = np.rad2deg(np.arcsin(-np.angle(D)/np.pi ))

    Theta = np.sort(Theta)
    return Theta

def CBF(Rxx, K, N):
    Thetalst = np.arange(-90, 90.1, 0.5)
    angle = np.deg2rad(Thetalst)
    Pcbf = np.zeros(angle.size)
    d = np.arange(0, N).reshape(-1, 1)
    for i, ang in enumerate(angle):
        a = np.exp(-1j * np.pi * d * np.sin(ang))
        Pcbf[i] = np.real(a.T.conjugate() @ Rxx @ a)[0,0]

    Pcbf = np.abs(Pcbf) / np.abs(Pcbf).max()
    Pcbf = 10 * np.log10(Pcbf)
    peaks, _ =  scipy.signal.find_peaks(Pcbf, height=-2,  distance = 10)
    angle_est = Thetalst[peaks]

    return Thetalst, Pcbf, angle_est, peaks

def Capon(Rxx, K, N):
    Thetalst = np.arange(-90, 90.1, 0.5)
    angle = np.deg2rad(Thetalst)
    Pcapon = np.zeros(angle.size)
    # for i, ang in enumerate(angle):
    #     a = np.exp(-1j * np.pi * d * np.sin(ang))
    #     Pcbf[i] = np.real(a.T.conjugate() @ Rxx @ a)[0,0]
    d = np.arange(0, N).reshape(-1, 1)
    for i, ang in enumerate(angle):
        a = np.exp(-1j * np.pi * d * np.sin(ang))
        Pcapon[i] = 1/np.real(a.T.conjugate() @ scipy.linalg.inv(Rxx) @ a)[0,0]

    Pcapon = np.abs(Pcapon) / np.abs(Pcapon).max()
    Pcapon = 10 * np.log10(Pcapon)
    peaks, _ =  scipy.signal.find_peaks(Pcapon, height=-2,  distance = 10)
    angle_est = Thetalst[peaks]

    return Thetalst, Pcapon, angle_est, peaks

# https://blog.csdn.net/weixin_44705592/article/details/131500890
# https://zhuanlan.zhihu.com/p/22897428966
# https://blog.csdn.net/qq_42233059/article/details/126524639
# 对求根MUSIC算法， 作如下说明。
# （1）求根MUSIC算法与谱搜索方式的MUSIC算法原理是一样的，只不过是用一个关于z的矢量来代替导向矢量，从而用求根过程代替搜索过程；
# （2）由于噪声的存在，求出的根不可能在单位圆上，可选择接近单位圆上的根为真实信号的根，这就存在一定的误差；
# （3）求根MUSIC算法与谱搜索的MUSIC算法相似，同样存在两种表达方式，一个是利用噪声子空间，另一个是利用信号子空间。

def ROOT_MUSIC(Rxx, K, d = 0.5, wavelength = 1.0):
    """
    Root-MUSIC 算法进行 DOA 估计（适用于 ULA）。

    参数:
        R: 接收信号的样本协方差矩阵 (num_sensors x num_sensors)
        num_sources: 信号数（需要估计的 DOA 数量）
        d: 传感器间距（以波长为单位，默认 0.5）
        wavelength: 信号波长（默认 1.0）

    返回:
        doa_estimates_deg: 估计的 DOA（单位：度，按从小到大排序）
    """
    N = Rxx.shape[0]
    eigvals, eigvecs = np.linalg.eigh(Rxx)  # # 对协方差矩阵进行特征值分解
    En = eigvecs[:, :N - K]      # 选取噪声子空间：使用最小的 (num_sensors - num_sources) 个特征向量
    Pn = En @ En.conj().T        # 构造噪声子空间投影矩阵

    # 利用 Toeplitz 结构提取多项式系数: 对于 ULA, Pn 的每条对角线理论上应相等, 这里对每条对角线求和, 得到系数 c[k] (k 从 -M+1 到 M-1)
    c = np.array([np.sum(np.diag(Pn, k)) for k in range(-N+1, N)])
    c = c / c[N - 1] # # 归一化：令 k=0（主对角线）的系数为 1，这不会改变根的位置

    poly_coeffs = c[::-1] # 构造多项式系数，注意 np.roots 要求系数按降幂排列
    roots_all = np.roots(poly_coeffs) # 求解多项式的所有根
    roots_inside = roots_all[np.abs(roots_all) < 1] # 只考虑位于单位圆内部的根（理论上信号相关根应落在单位圆附近）

    # 根据距离单位圆的距离排序，选择最接近单位圆的 num_sources 个根
    distances = np.abs(np.abs(roots_inside) - 1)
    sorted_indices = np.argsort(distances)
    selected_roots = roots_inside[sorted_indices][:K]

    # 由理论，根的相位与 DOA 满足: angle(z) = -2π*d*sin(θ)/wavelength
    # beta = 2π*d/wavelength
    beta = 2 * np.pi * d / wavelength
    phi = np.angle(selected_roots)

    doa_estimates_rad = np.arcsin(-phi / beta)
    doa_estimates_deg = np.rad2deg(doa_estimates_rad)

    return np.sort(doa_estimates_deg), roots_all

derad = np.pi/180             # 角度->弧度
N = 8                         # 阵元个数
K = 4                         # 信源数目
doa_deg = [-30, 0, 30, 60]    # 待估计角度
doa_rad = np.deg2rad(doa_deg) # beam angles
f0 = 1e6
f = np.array([0.1, 0.2, 0.3, 0.5 ]) * f0  # 为了保持各个用户的信号正交，需要满足各个用户的频率不等或者是各个用户的信号为随机噪声
snr = 20                                  # 信噪比
Ns = 1000                                 # 快拍数
fs = 1e8                                  # 满足采样定理，fs >> f0
Ts = 1/fs
t = np.arange(Ns) * Ts
SNR = 10                                  # 信噪比(dB)

# generate signals
X = np.zeros((N, Ns), dtype = complex)
for i in range(K):
    a_k = steering_vector(doa_rad[i], N)
    # s = np.exp(1j * 2 * np.pi * np.random.rand(Ns))  # 信源信号，入射信号，不相干即可，也可以用正弦替代
    s = np.exp(1j * 2 * np.pi * f[i] * t)  # 正弦 signals
    X += np.outer(a_k, s)

# add noise
noisevar = 10 ** (-SNR / 20)
noise = np.sqrt(noisevar/2) * (np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape))
X += noise
Rxx = X @ X.T.conjugate() / Ns

Thetalst, Pmusic, angle_music, peak_music = MUSIC(Rxx, K, N)
Thetalst, Pcbf, angle_cbf, perak_cbf = CBF(Rxx, K, N)
Thetalst, Pcapon, angle_capon, peak_capon = Capon(Rxx, K, N)
Theta_esprit = ESPRIT(Rxx, K, N)
Theta_root, roots_all = ROOT_MUSIC(Rxx, K )

print(f"True = {doa_deg}")
print(f"MUSIC = {angle_music}")
print(f"Root MUSIC = {Theta_root}")
print(f"CBF = {angle_cbf}")
print(f"Capon = {angle_capon}")
print(f"ESPRIT = {Theta_esprit}")

###>>>>>>>>>>
colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8))

axs.plot(Thetalst, Pmusic , color = colors[0], linestyle='-', lw = 2, label = "MUSIC", )
axs.plot(angle_music, Pmusic[peak_music], linestyle='', marker = 'o', color=colors[0], markersize = 12)

axs.plot(Thetalst, Pcbf , color = colors[1], linestyle='--', lw = 2, label = "CBF", )
axs.plot(angle_cbf, Pcbf[perak_cbf], linestyle='', marker = 'd', color=colors[1], markersize = 12)

axs.plot(Thetalst, Pcapon , color = colors[2], linestyle='-.', lw = 2, label = "CAPON", )
axs.plot(angle_capon, Pcapon[peak_capon], linestyle='', marker = 's', color=colors[2], markersize = 12)

axs.plot(Theta_esprit, np.zeros(K), linestyle='', marker = '*', color=colors[3], markersize = 12, label = "ESPRIT", )
axs.plot(Theta_root, np.zeros(K)-5, linestyle='', marker = 'v', color='r', markersize = 12, label = "ROOT MUSIC", )

axs.set_xlabel( "DOA/(degree)",)
axs.set_ylabel('Normalized Spectrum/(dB)',)
axs.legend()

plt.show()
plt.close('all')

##>>>>>>>>>>>>>
fig, axs = plt.subplots(1, 1, figsize = (6, 6))

theta = np.linspace(0, 2*np.pi, 400)
axs.plot(np.cos(theta), np.sin(theta), 'k--', lw = 1, label='unit circle')
axs.scatter(np.real(roots_all), np.imag(roots_all), marker='o', color='b', label='roots of polynomial')
axs.set_xlabel('Real', fontsize = 12, )
axs.set_ylabel('Imaginary', fontsize = 12,)
axs.set_title('Root-MUSIC Root Distribution', fontsize = 12,)
axs.axis('equal')
axs.legend( fontsize = 12,)
axs.grid(True)
plt.show()
plt.close('all')


















