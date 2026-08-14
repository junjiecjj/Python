#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 17:29:50 2025

@author: jack
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
     for i in range(L):
          for j in range(L):
               mat[i,j] = 1.0*np.exp(-1j*2.0*np.pi*i*j/L) / (np.sqrt(L)*1.0)
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

#%%
# 参数设置
Tsym = 1
pi = np.pi
N = 128       # 符号数
L = 10        # 过采样率
alpha = 0.3  # 滚降因子
# span = 6      # 滤波器跨度（根据旁瓣要求调整）

# p, t, filtDelay = srrcFunction(alpha, L, span, Tsym = Tsym)
# p = np.pad(p, (0, L*N - p.size))

t, p = commpy.filters.rrcosfilter(L*N , alpha, Tsym, L/Tsym)
p = p / np.sqrt(np.sum(np.power(p, 2)))

norm2p = np.linalg.norm(p)
FLN = FFTmatrix(L*N )
FN = FFTmatrix(N )

###>>>>> OFDM, 16QAM, Eq.(36), 化简后的表达式
kappa = 1.32
U = FN.conj().T
V = np.eye(N)  # U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
g = (N * (FLN@p) * (FLN.conj() @ p.conj()))

TheoAveACF_OFDM_16QAM_M1 = np.zeros(L*N)
for k in range(L*N):
    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    fk = np.exp(-1j * 2*pi * k * np.arange(N)/(L*N))

    r1 = np.abs(gk @ fk.conj())**2
    # TheoAveACF_Iceberg[k] = r1 #+ r2

    M = 1
    r2 = (kappa - 1) / M * (N - 2 *(1-np.cos(2*pi*k/L))*(g[:N] * (1- g[:N])).sum())
    TheoAveACF_OFDM_16QAM_M1[k] = r1 + r2

TheoAveACF_OFDM_16QAM_M1 = TheoAveACF_OFDM_16QAM_M1/TheoAveACF_OFDM_16QAM_M1.max() + 1e-10
TheoAveACF_OFDM_16QAM_M1 = np.fft.fftshift(TheoAveACF_OFDM_16QAM_M1)

###>>>>> OFDM, 1024 QAM, Eq.(36), 化简后的表达式
kappa = 1.3988
U = FN.conj().T
V = np.eye(N)  # U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
g = (N * (FLN@p) * (FLN.conj() @ p.conj()))

TheoAveACF_OFDM_1024QAM_M1 = np.zeros(L*N)
for k in range(L*N):
    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    fk = np.exp(-1j * 2*pi * k * np.arange(N)/(L*N))

    r1 = np.abs(gk @ fk.conj())**2
    # TheoAveACF_Iceberg[k] = r1 #+ r2

    M = 1
    r2 = (kappa - 1) / M * (N - 2 *(1-np.cos(2*pi*k/L))*(g[:N] * (1- g[:N])).sum())
    TheoAveACF_OFDM_1024QAM_M1[k] = r1 + r2

TheoAveACF_OFDM_1024QAM_M1 = TheoAveACF_OFDM_1024QAM_M1/TheoAveACF_OFDM_1024QAM_M1.max() + 1e-10
TheoAveACF_OFDM_1024QAM_M1 = np.fft.fftshift(TheoAveACF_OFDM_1024QAM_M1)

###>>>>> OFDM, PSK, Eq.(36), 化简后的表达式
kappa = 1
U = FN.conj().T
V = np.eye(N)  # U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
g = (N * (FLN@p) * (FLN.conj() @ p.conj()))
TheoAveACF_Iceberg = np.zeros(L*N)
TheoAveACF_OFDM_PSK_M1 = np.zeros(L*N)

for k in range(L*N):

    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    fk = np.exp(-1j * 2*pi * k * np.arange(N)/(L*N))

    r1 = np.abs(gk @ fk.conj())**2
    TheoAveACF_Iceberg[k] = r1 #+ r2

    M = 1
    r2 = (kappa - 1) / M * (N - 2 *(1-np.cos(2*pi*k/L))*(g[:N] * (1- g[:N])).sum())
    TheoAveACF_OFDM_PSK_M1[k] = r1 + r2


TheoAveACF_Iceberg = TheoAveACF_Iceberg/TheoAveACF_Iceberg.max() + 1e-10
TheoAveACF_Iceberg = np.fft.fftshift(TheoAveACF_Iceberg)


TheoAveACF_OFDM_PSK_M1 = TheoAveACF_OFDM_PSK_M1/TheoAveACF_OFDM_PSK_M1.max() + 1e-10
TheoAveACF_OFDM_PSK_M1 = np.fft.fftshift(TheoAveACF_OFDM_PSK_M1)

###>>>>> OFDM, Gaussian, Eq.(36), 化简后的表达式
kappa = 2
U = FN.conj().T
V = np.eye(N)  # U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
g = (N * (FLN@p) * (FLN.conj() @ p.conj()))

TheoAveACF_OFDM_Gaussian_M1 = np.zeros(L*N)

for k in range(L*N):

    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    fk = np.exp(-1j * 2*pi * k * np.arange(N)/(L*N))

    r1 = np.abs(gk @ fk.conj())**2
    # TheoAveACF_Iceberg[k] = r1 #+ r2

    M = 1
    r2 = (kappa - 1) / M * (N - 2 *(1-np.cos(2*pi*k/L))*(g[:N] * (1- g[:N])).sum())
    TheoAveACF_OFDM_Gaussian_M1[k] = r1 + r2

TheoAveACF_OFDM_Gaussian_M1 = TheoAveACF_OFDM_Gaussian_M1/TheoAveACF_OFDM_Gaussian_M1.max() + 1e-10
TheoAveACF_OFDM_Gaussian_M1 = np.fft.fftshift(TheoAveACF_OFDM_Gaussian_M1)

#%% plot together
x = np.arange(-N*L//2, N*L//2,)
fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

axs.plot(x, 10 * np.log10(TheoAveACF_OFDM_Gaussian_M1), color='g', linestyle='--', label='Gaussian',)
axs.plot(x, 10 * np.log10(TheoAveACF_OFDM_1024QAM_M1 ), color='r', linestyle='-', label='1024QAM',)

axs.plot(x, 10 * np.log10(TheoAveACF_OFDM_16QAM_M1 ), color='b', linestyle='--', label='16QAM',)
axs.plot(x, 10 * np.log10(TheoAveACF_OFDM_PSK_M1), color='k', linestyle='--', label='PSK',)

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
axs.set_xlim([-100, 100])
axs.set_ylim([-50, 5])
out_fig = plt.gcf()
out_fig.savefig('SPM_Fig5.pdf', )
plt.show()
plt.close()



























































































































































































