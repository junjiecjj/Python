#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 17:46:01 2025

@author: Junjie Chen

需要安装commpy, 安装命令:  pip install scikit-commpy,
主要是使用了commpy里面的调制解调函数，也可以用自己写的
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import commpy
from Modulations import modulator

from matplotlib.font_manager import FontProperties
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22                     # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22                # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22                # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18               # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18               # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False         # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]            # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300                 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2                # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6               # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22
np.random.seed(42)

#%%
# 产生傅里叶矩阵
def FFTmatrix(L, ):
     mat = np.zeros((L, L), dtype = complex)
     ll = np.arange(L)
     for i in range(L):
         mat[i,:] = 1.0*np.exp(-1j*2.0*np.pi*i*ll/L) / (np.sqrt(L)*1.0)
     return mat

def srrcFunction(beta, L, span, Tsym = 1):
    # Function for generating rectangular pulse for the given inputs
    # L - oversampling factor (number of samples per symbol)
    # span - filter span in symbol durations
    # Returns the output pulse p(t) that spans the discrete-time base -span:1/L:span. Also returns the filter delay.
    t = np.arange(-span*Tsym/2, span*Tsym/2 + 0.5/L, Tsym/L)
    A = np.sin(np.pi*t*(1-beta)/Tsym) + 4*beta*t/Tsym * np.cos(np.pi*t*(1+beta)/Tsym)
    B = np.pi*t/Tsym * (1-(4*beta*t/Tsym)**2)
    p = 1/np.sqrt(Tsym) * A/B
    p[np.argwhere(np.isnan(p))] = 1
    p[np.argwhere(np.isinf(p))] = beta/(np.sqrt(2*Tsym)) * ((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
    filtDelay = (len(p)-1)/2
    p = p / np.sqrt(np.sum(np.power(p, 2))) # power normaize.
    return p, t, filtDelay

## CDMA, U
def hadamard_matrix_sylvester(n):
    """Sylvester 构造法生成 Hadamard 矩阵（n 必须是 2 的幂）"""
    if n == 1:
        return np.array([[1]])
    else:
        H_prev = hadamard_matrix_sylvester(n // 2)
        H = np.kron(H_prev, np.array([[1, 1], [1, -1]]))
        return H

# AFDM, U
def IDAFT(c1, c2, N):
    """
    AFDM调制函数
    参数:
        x : 输入信号 (Nx1 列向量)
        c1 : 第一个调频参数
        c2 : 第二个调频参数
    返回:
        out : 调制输出信号
    """
    # N = x.shape[0]

    # 创建DFT矩阵并归一化
    F = np.fft.fft(np.eye(N))
    F = F / np.linalg.norm(F, ord = 2)

    # 创建L1和L2对角矩阵
    n = np.arange(N)
    L1 = np.diag(np.exp(-1j * 2 * np.pi * c1 * (n**2)))
    L2 = np.diag(np.exp(-1j * 2 * np.pi * c2 * (n**2)))

    # 构建AFDM矩阵
    A = L2 @ F @ L1
    # 计算调制输出 (注意MATLAB的'是共轭转置，Python用.conj().T)
    return A.conj().T

#%%
# 参数设置
Tsym = 1
pi = np.pi
N = 128       # 符号数
L = 10        # 过采样率
alpha = 0.3   # 滚降因子
span = 6    # 滤波器跨度（根据旁瓣要求调整）

p, t, filtDelay = srrcFunction(alpha, L, span, Tsym = Tsym)
p = np.pad(p, (0, L*N - p.size))

# t, p = commpy.filters.rrcosfilter(L*N , alpha, Tsym, L/Tsym)
# p = p / np.sqrt(np.sum(np.power(p, 2)))

norm2p = np.linalg.norm(p)
FLN = FFTmatrix(L*N )
FN = FFTmatrix(N )

###>>>>> OFDM, Eq.(36), 化简后的表达式

kappa = 1.32
U = FN.conj().T
V = np.eye(N)  # U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
g = (N * (FLN@p) * (FLN.conj() @ p.conj()))

TheoAveACF_Iceberg = np.zeros(L*N)
TheoAveACF_OFDM_M1 = np.zeros(L*N)

for k in range(L*N):

    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    fk = np.exp(-1j * 2*pi * k * np.arange(N)/(L*N))

    r1 = np.abs(gk @ fk.conj())**2
    TheoAveACF_Iceberg[k] = r1 #+ r2

    M = 1
    r2 = (kappa - 1) / M * (N - 2 *(1-np.cos(2*pi*k/L))*(g[:N] * (1- g[:N])).sum())
    TheoAveACF_OFDM_M1[k] = r1 + r2

TheoAveACF_Iceberg = TheoAveACF_Iceberg/TheoAveACF_Iceberg.max() + 1e-10
TheoAveACF_Iceberg = np.fft.fftshift(TheoAveACF_Iceberg)

TheoAveACF_OFDM_M1 = TheoAveACF_OFDM_M1/TheoAveACF_OFDM_M1.max() + 1e-10
TheoAveACF_OFDM_M1 = np.fft.fftshift(TheoAveACF_OFDM_M1)

###>>>>> SC, Eq.(27, 34), 化简前的general表达式

# kappa = 1.32
# U = np.eye(N)
# V = U.conj().T @ FN.conj().T
# tilde_V = V * V.conj()
# g = (N * (FLN@p) * (FLN.conj() @ p.conj()))

# TheoAveACF_SC_M1 = np.zeros(L*N)

# for k in range(L*N):
#     fk = FLN[:,k]
#     fk_tilde = fk[:N]
#     gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
#     r1 = L * N * np.abs(fk_tilde.conj().T @ gk)**2
#     r2 = np.linalg.norm(gk)**2
#     r3 = (kappa - 2) * L * N * np.linalg.norm(tilde_V @ (gk * fk_tilde.conj()))**2

#     TheoAveACF_SC_M1[k] = r1 + (r2 + r3)/1

# TheoAveACF_SC_M1 = TheoAveACF_SC_M1/TheoAveACF_SC_M1.max() + 1e-10
# TheoAveACF_SC_M1 = np.fft.fftshift(TheoAveACF_SC_M1)

###>>>>> SC, Eq.(37)
kappa = 1.32
g = (N * (FLN@p) * (FLN.conj() @ p.conj()))
TheoAveACF_SC_M1 = np.zeros(L*N)
for k in range(L*N):
    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    fk = np.exp(-1j * 2*pi * k * np.arange(N)/(L*N))

    M = 1
    r1 = (1+ (kappa-2)/(M*N)) * np.abs(gk @ fk.conj())**2
    r2 = 1 / M * (N - 2 *(1-np.cos(2*pi*k/L))*(g[:N] * (1- g[:N])).sum())
    TheoAveACF_SC_M1[k] = r1 + r2

TheoAveACF_SC_M1 = TheoAveACF_SC_M1/TheoAveACF_SC_M1.max() + 1e-10
TheoAveACF_SC_M1 = np.fft.fftshift(TheoAveACF_SC_M1)

#%% CDMA, Eq.(27, 34), 化简前的general表达式
kappa = 1.32
U = hadamard_matrix_sylvester(N)/np.sqrt(N)
V = U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
g = (N * (FLN@p) * (FLN.conj() @ p.conj()))

TheoAveACF_CDMA_M1 = np.zeros(L*N)

for k in range(L*N):
    fk = FLN[:,k]
    fk_tilde = fk[:N]
    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    r1 = L * N * np.abs(fk_tilde.conj().T @ gk)**2
    r2 = np.linalg.norm(gk)**2
    r3 = (kappa - 2) * L * N * np.linalg.norm(tilde_V @ (gk * fk_tilde.conj()))**2

    TheoAveACF_CDMA_M1[k] = r1 + (r2 + r3)/1
TheoAveACF_CDMA_M1 = np.abs(TheoAveACF_CDMA_M1)
TheoAveACF_CDMA_M1 = TheoAveACF_CDMA_M1/TheoAveACF_CDMA_M1.max() + 1e-10
TheoAveACF_CDMA_M1 = np.fft.fftshift(TheoAveACF_CDMA_M1)

#%% OTFS, (27, 34), 化简前的general表达式

kappa = 1.32
FFTN = 32
Neye = int(N/FFTN)
FFTM = FFTmatrix(FFTN)
eyeM = np.eye(Neye)
U = np.kron(FFTM, eyeM)
V = U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
g = (N * (FLN@p) * (FLN.conj() @ p.conj()))

TheoAveACF_OTFS_M1 = np.zeros(L*N)

for k in range(L*N):
    fk = FLN[:,k]
    fk_tilde = fk[:N]
    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    r1 = L * N * np.abs(fk_tilde.conj().T @ gk)**2
    r2 = np.linalg.norm(gk)**2
    r3 = (kappa - 2) * L * N * np.linalg.norm(tilde_V @ (gk * fk_tilde.conj()))**2

    TheoAveACF_OTFS_M1[k] = r1 + (r2 + r3)/1
TheoAveACF_OTFS_M1 = np.abs(TheoAveACF_OTFS_M1)
TheoAveACF_OTFS_M1 = TheoAveACF_OTFS_M1/TheoAveACF_OTFS_M1.max() + 1e-10
TheoAveACF_OTFS_M1 = np.fft.fftshift(TheoAveACF_OTFS_M1)

#%% AFDM, (27, 34), 化简前的general表达式

c1 = 1/128
c2 = 4/(3*pi)
U = IDAFT(c1, c2, N)
V = U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
g = (N * (FLN@p) * (FLN.conj() @ p.conj()))

TheoAveACF_AFDM_M1 = np.zeros(L*N)

for k in range(L*N):
    fk = FLN[:,k]
    fk_tilde = fk[:N]
    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    r1 = L * N * np.abs(fk_tilde.conj().T @ gk)**2
    r2 = np.linalg.norm(gk)**2
    r3 = (kappa - 2) * L * N * np.linalg.norm(tilde_V @ (gk * fk_tilde.conj()))**2

    TheoAveACF_AFDM_M1[k] = r1 + (r2 + r3)/1
TheoAveACF_AFDM_M1 = np.abs(TheoAveACF_AFDM_M1)
TheoAveACF_AFDM_M1 = TheoAveACF_AFDM_M1/TheoAveACF_AFDM_M1.max() + 1e-10
TheoAveACF_AFDM_M1 = np.fft.fftshift(TheoAveACF_AFDM_M1)

#%% plot together

x = np.arange(-N*L//2, N*L//2,)
fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

axs.plot(x, 10 * np.log10(TheoAveACF_SC_M1), color='#00A1F1', linestyle='-', label='SC',)

axs.plot(x, 10 * np.log10(TheoAveACF_AFDM_M1), color='#00A8BB', linestyle='--', lw = 2, label='AFDM',)

axs.plot(x, 10 * np.log10(TheoAveACF_OTFS_M1), color='#8A2BE2', linestyle='--', lw = 2, label='OTFS',)

axs.plot(x, 10 * np.log10(TheoAveACF_CDMA_M1), color='#77AC30', linestyle='--', lw = 2, label='CDMA',)

axs.plot(x, 10 * np.log10(TheoAveACF_OFDM_M1 ), color='r', linestyle='-', lw = 2, label='OFDM',)

axs.plot(x, 10 * np.log10(TheoAveACF_Iceberg), color='k', linestyle='--', lw = 2, label="Iceberg",)

font1 = {'family':'Times New Roman','style':'normal','size':12, }
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', fontsize = 12, labelspacing = 0.2, prop = font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

bw = 2
axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 16, width = bw)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(22) for label in labels] #刻度值字号

axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )


axs.set_xlabel(r'Delay Index', )
axs.set_ylabel(r'Ambiguity Level (dB)', )
axs.set_title(r'16-QAM With CP')
axs.set_xlim([-100, 100])
axs.set_ylim([-50, 5])
out_fig = plt.gcf()
out_fig.savefig('SPM_Fig3.pdf', )
plt.show()
plt.close()






















































































































































