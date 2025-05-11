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
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
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
    return

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
    return C_fading[0]

#%% AWGN SISO信道
SNRs = np.arange(0, 30)
cap_awgn = np.zeros(SNRs.size)
for i, SNR in enumerate(SNRs):
    cap_awgn[i] = awgn_capacity(SNR)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.plot(SNRs, cap_awgn, color = 'r', marker = 'o', linestyle = '--', lw = 1, label = f"SNR = {SNR}")

# ax.set_ylim(1e-6, 1)
ax.set_xlabel('SNR in dB')
ax.set_ylabel('信道容量 bits/s/Hz')
ax.set_title('awgn信道容量')
# ax.legend()
plt.show()
plt.close()

#%% Raly SISO信道
SNRs = np.arange(0, 30)
cap_raly = np.zeros(SNRs.size)
for i, SNR in enumerate(SNRs):
    cap_raly[i] = ralychannel(SNR)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.plot(SNRs, cap_raly, color = 'r', marker = 'o', linestyle = '--', lw = 1, label = f"SNR = {SNR}")

# ax.set_ylim(1e-6, 1)
ax.set_xlabel('SNR in dB')
ax.set_ylabel('信道容量 bits/s/Hz')
ax.set_title('raly 信道容量')
# ax.legend()
plt.show()
plt.close()

#%% 接收天线变化
Nt = 4
Nr = np.arange(1, 20)
SNRs = [5, 10, 15, 20]
CAPs = np.zeros((len(SNRs), Nr.size))

colors = plt.cm.jet(np.linspace(0, 1, len(CAPs))) # colormap
fig, ax = plt.subplots(nrows = 1, ncols = 1)

for i, SNR in enumerate(SNRs):
    CAPs[i,:] = np.array([mimo_capacity_noCIS(nr, Nt, SNR) for nr in Nr])
    ax.plot(Nr, CAPs[i,:], color = colors[i], marker = 'o', linestyle = '--', lw = 1, label = f"SNR = {SNR}")

# ax.set_ylim(1e-6, 1)
ax.set_xlabel('接收天线数目')
ax.set_ylabel('信道容量 bits/s/Hz')
ax.set_title('发射天线数目为4的情况')
ax.legend()
plt.show()
plt.close()

#%% 发射天线变化
from matplotlib import cm
Nr = 4
Nt = np.arange(1, 20)
SNRs = [5, 10, 15, 20]
CAPs = np.zeros((len(SNRs), Nt.size))

colors = plt.cm.jet(np.linspace(0, 1, len(CAPs))) # colormap
fig, ax = plt.subplots(nrows = 1, ncols = 1)

for i, SNR in enumerate(SNRs):
    CAPs[i,:] = np.array([mimo_capacity_noCIS(Nr, nt, SNR) for nt in Nt])
    ax.plot(Nt, CAPs[i,:], color = colors[i], marker = 'o', linestyle = '--', lw = 1, label = f"SNR = {SNR}")

# ax.set_ylim(1e-6, 1)
ax.set_xlabel('发射天线数目')
ax.set_ylabel('信道容量 bits/s/Hz')
ax.set_title('接收天线数目为4的情况')
ax.legend()
plt.show()
plt.close()

#%% 三维图展示MIMO
Nr = np.arange(1, 20)
Nt = np.arange(1, 20)
SNR = 15

CAPs = np.zeros((Nr.size, Nt.size))
for i, SNR in enumerate(Nr):
    CAPs[i,:] = np.array([mimo_capacity_noCIS(Nr[i], nt, SNR) for nt in Nt])

xx,yy = np.meshgrid(Nt, Nr)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.set_proj_type('ortho')
norm_plt = plt.Normalize(CAPs.min(), CAPs.max())
colors = cm.RdYlBu_r(norm_plt(CAPs))
# colors = cm.Blues_r(norm_plt(ff))

surf = ax.plot_surface(xx, yy, CAPs, facecolors = colors,
                       rstride = 1,
                       cstride = 1,
                       linewidth = 1, # 线宽
                       shade = False ) # 删除阴影
surf.set_facecolor((0, 0, 0, 0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

ax.set_proj_type('ortho')
ax.set_xlabel('# Nt')
ax.set_ylabel('# Nr')
ax.set_zlabel('Capacity')

# 设置X、Y、Z面的背景是白色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

# ax.set_xlim(x_array.min(), x_array.max())
# ax.set_ylim(y_array.min(), y_array.max())

ax.view_init(azim = -135, elev = 30)
ax.grid(False)

plt.show()


#%%  MIMO-OFDM Wireless Communications with MATLAB
#%% Program 9.1 “Ergodic_Capacity_CDF.m” for ergodic capacity of MIMO channel, 发射端未知CSI
SNR_dB = 10
SNR = 10**(SNR_dB/10.0)
N_iter = 50000
sq2 = np.sqrt(0.5)
CASEs = [[2,2], [4,4]]
N = 100
CDF = np.zeros((len(CASEs), N))
Bins = np.zeros((len(CASEs), N))
fig, axs = plt.subplots(nrows = 1, ncols = 1)
colors = plt.cm.jet(np.linspace(0, 1, len(CASEs))) # colormap
for i, (Nr, Nt) in enumerate(CASEs):
    n = min(Nr, Nt)
    I = np.eye(n)
    C = np.zeros(N_iter)
    for it in range(N_iter):
        H = sq2 * (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt))
        C[it] = np.log2(np.abs(np.linalg.det(I + SNR/Nt * H.conjugate().T @ H)))
    cdf = axs.hist(C, bins = N, density = True, histtype = 'step',color = colors[i], alpha = 0.75, cumulative = True, rwidth = 0.8, label = f'Nt = {Nt}, Nr = {Nr}')
    CDF[i] = cdf[0]
    Bins[i] = cdf[1][:-1]
axs.set_xlabel('Rate[bps/Hz]')
axs.set_ylabel('CDF')
# ax.set_title('raly 信道容量')
# ax.legend()
plt.show()
plt.close()

fig, axs = plt.subplots(nrows = 1, ncols = 1)
colors = plt.cm.jet(np.linspace(0, 1, len(CASEs))) # colormap
for i, (Nr, Nt) in enumerate(CASEs):
    axs.plot(Bins[i], CDF[i], color = colors[i], label = f'Nt = {Nt}, Nr = {Nr}' )
axs.set_xlabel('CDF')
axs.set_ylabel('Rate[bps/Hz]')
# ax.set_title('raly 信道容量')
axs.grid()
axs.legend()
plt.show()
plt.close()

#%% Program 9.2 “Ergodic_Capacity_vs_SNR.m” for ergodic channel capacity vs. SNR in Figure 9.6.
SNR_dB = np.arange(0, 22, 2)
SNR = 10**(SNR_dB/10.0)
N_iter = 5000
sq2 = np.sqrt(0.5)
CASEs = [[1,1], [1,2], [2,1], [2,2], [4,4]]
C = np.zeros((len(CASEs), SNR_dB.size))
for i, (Nt, Nr) in enumerate(CASEs):
    n = min(Nr, Nt)
    I = np.eye(n)
    for j, snr in enumerate(SNR):
        for it in range(N_iter):
            H = sq2 * (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt))
            if Nr > Nt:
                HH = H.conjugate().T @ H
            else:
                HH = H @ H.conjugate().T
            C[i, j] += np.log2(np.abs(np.linalg.det(I + snr/Nt * HH)))
C /= N_iter

fig, axs = plt.subplots(nrows = 1, ncols = 1)
colors = plt.cm.jet(np.linspace(0, 1, len(C))) # colormap
for i, (Nt, Nr) in enumerate(CASEs):
    axs.plot(SNR_dB, C[i,:], color = colors[i], marker = 'o', linestyle = '--', lw = 1, label = f'Nt = {Nt}, Nr = {Nr}')
axs.set_xlabel('SNR[dB]')
axs.set_ylabel('Rate[bps/Hz]')
axs.set_title('信道容量')
axs.legend()
plt.show()
plt.close()

#%% Program 9.3 “OL_CL_Comparison.m” for Ergodic channel capacity: open-loop vs. closed- loop
import cvxpy as cp

def Water_Filling(snr, s, Nt):
    n = s.size
    x = cp.Variable(shape=n)
    alpha = cp.Parameter(n, nonneg = True)
    alpha.value = snr*s/Nt
    obj = cp.Maximize(cp.sum(cp.log(1 + cp.multiply(alpha, x))))
    constraints = [x >= 0, cp.sum(x) - Nt == 0]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    if(prob.status=='optimal'):
        return prob.status, prob.value, x.value
    else:
        return prob.status, np.nan, np.nan

SNR_dB = np.arange(0, 22, 2)
SNR = 10**(SNR_dB/10.0)
N_iter = 1000
sq2 = np.sqrt(0.5)
nT = 4
nR = 4

n = min(nT,nR);
I = np.eye(n)
rho = 0.2
Rtx = np.array([[1, rho, rho**2, rho**3],
               [ rho, 1, rho, rho**2],
               [ rho**2, rho, 1, rho],
               [ rho**3, rho**2, rho, 1]])
Rrx = np.array([[1, rho, rho**2, rho**3],
               [ rho, 1, rho, rho**2],
               [ rho**2, rho, 1, rho],
               [ rho**3, rho**2, rho, 1]])
C_OL = np.zeros(SNR.size)
C_CL = np.zeros(SNR.size)
C_OL1 = np.zeros(SNR.size)
C_CL1 = np.zeros(SNR.size)

for it in range(N_iter):
    # print(f"{i+1}/{SNR.size}")
    Hw = sq2 * (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt))
    H = Rrx**(1/2) @ Hw @ Rtx**(1/2)
    # H = Hw
    tmp = H.conjugate().T @ H / Nt
    U, s, VH = np.linalg.svd(H.conjugate().T @ H)
    for i, snr in enumerate(SNR):
        C_OL[i] += np.log2(np.real(np.linalg.det(I + snr * tmp)))
        C_OL1[i] += np.sum([np.log2(1 + snr/Nt * lamba) for lamba in list(s)]) # equivalent with C_OL
        status, value, sol = Water_Filling(snr, s, Nt)
        C_CL[i] += value
        C_CL1[i] += np.log2(np.abs(np.linalg.det(I + np.diag(sol)@np.diag(s) * snr/Nt)))
C_OL /= N_iter
C_CL /= N_iter
C_OL1 /= N_iter
C_CL1 /= N_iter

fig, axs = plt.subplots(nrows = 1, ncols = 1)
colors = plt.cm.jet(np.linspace(0, 1, 2)) # colormap
# axs.plot(SNR_dB, C_OL, color = 'r', marker = 'o', linestyle = '--', lw = 1, label = "Channel Unknown")
# axs.plot(SNR_dB, C_CL, color = 'b', marker = 'o', linestyle = '--', lw = 1, label = "Channel Known")
axs.plot(SNR_dB, C_OL1, color = 'r', marker = '*', linestyle = '-', lw = 1, label = "Channel Unknown1")
axs.plot(SNR_dB, C_CL1, color = 'b', marker = '*', linestyle = '-', lw = 1, label = "Channel Known1")

axs.set_xlabel('SNR[dB]')
axs.set_ylabel('Rate[bps/Hz]')
axs.set_title('开环和闭环MIMO信道容量')
axs.legend()
plt.show()
plt.close()



#%% Program 9.5 “Ergodic_Capacity_Correlation.m:” Channel capacity reduction due to correlation

SNR_dB = np.arange(0, 22, 2)
SNR = 10**(SNR_dB/10.0)
N_iter = 1000
sq2 = np.sqrt(0.5)
nT = 4
nR = 4

n = min(nT,nR);
I = np.eye(n)
rho = 0.2
pi = np.pi
RRx = np.eye(n)
# RRx = np.array([[1, 0.76*np.exp(0.17j*pi), 0.43*np.exp(0.35j*pi), 0.25*np.exp(0.53j*pi)],
#               [0.76*np.exp(-0.17j*pi), 1,  0.76*np.exp(0.17j*pi), 0.43*np.exp(0.35j*pi)],
#               [0.43*np.exp(-0.35j*pi), 0.76*np.exp(-0.17j*pi), 1, 0.76*np.exp(0.17j*pi)],
#               [0.25*np.exp(-0.53j*pi), 0.43*np.exp(-0.35j*pi), 0.76*np.exp(-0.17j*pi), 1]])

RTx = np.array([[1, 0.76*np.exp(0.17j*pi), 0.43*np.exp(0.35j*pi), 0.25*np.exp(0.53j*pi)],
              [0.76*np.exp(-0.17j*pi), 1,  0.76*np.exp(0.17j*pi), 0.43*np.exp(0.35j*pi)],
              [0.43*np.exp(-0.35j*pi), 0.76*np.exp(-0.17j*pi), 1, 0.76*np.exp(0.17j*pi)],
              [0.25*np.exp(-0.53j*pi), 0.43*np.exp(-0.35j*pi), 0.76*np.exp(-0.17j*pi), 1]])
C_44_iid = np.zeros(SNR.size)
C_44_corr = np.zeros(SNR.size)

for it in range(N_iter):
   H_iid = sq2 * (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt))
   H_corr = RRx**(1/2) @ H_iid @ RTx**(1/2)
   tmp1 = H_iid.conjugate().T @ H_iid/Nt
   tmp2 = H_corr.conjugate().T @ H_corr/Nt
   for i, snr in enumerate(SNR):
      C_44_iid[i] = C_44_iid[i] + np.log2(np.abs(np.linalg.det(I + snr*tmp1)))
      C_44_corr[i] = C_44_corr[i] + np.log2(np.abs(np.linalg.det(I + snr*tmp2)))

C_44_iid /= N_iter
C_44_corr /= N_iter

fig, axs = plt.subplots(nrows = 1, ncols = 1)
colors = plt.cm.jet(np.linspace(0, 1, 2)) # colormap
axs.plot(SNR_dB, C_44_iid, color = 'r', marker = 'o', linestyle = '--', lw = 1, label = "iid 4x4 channels")
axs.plot(SNR_dB, C_44_corr, color = 'b', marker = 'o', linestyle = '--', lw = 1, label = "correlated 4x4 channels")
axs.set_xlabel('SNR[dB]')
axs.set_ylabel('Rate[bps/Hz]')
axs.set_title('信道相关信道容量')
axs.legend()
plt.show()
plt.close()





















































































































