#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 00:56:12 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import commpy

from Tools import freqDomainView

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


#%%
def generateJk(L, N, k):
    if k < 0:
        k = L*N+k
    if k == 0:
        Jk = np.eye(L*N)
    elif k > 0:
        tmp1 = np.zeros((k, L*N-k))
        tmp2 = np.eye(k)
        tmp3 = np.eye(L*N-k)
        tmp4 = np.zeros((L*N - k, k))
        Jk = np.block([[tmp1, tmp2], [tmp3, tmp4]])
    return Jk

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
    p = p / np.sqrt(np.sum(np.power(p, 2))) # both Add and Delete this line is OK.
    return p, t, filtDelay

# 产生傅里叶矩阵
def FFTmatrix(row, col):
     mat = np.zeros((row, col), dtype = complex)
     for i in range(row):
          for j in range(col):
               mat[i,j] = 1.0*np.exp(-1j*2.0*np.pi*i*j/row) / (np.sqrt(row)*1.0)
     return mat

# 生成 循环矩阵
def CirculantMatric(gen, row):
     if type(gen) == list:
         col = len(gen)
     elif type(gen) == np.ndarray:
         col = gen.size
     row = col
     mat = np.zeros((row, col), np.complex128)
     mat[0, :] = gen
     for i in range(1, row):
         mat[i, :] = np.roll(gen, i)
     return mat

#%% Eq.(23)
beta = 0.35
Tsym = 1
N = 64
L = 6
span = 6
Fs = L/Tsym
B0  = 1/(2*Tsym)                  # Hz
B = (1 + beta) * B0
# f_max = 2*np.pi*B               # 角频率rad/s,
f_max = B                         # 画图用的时间频率 Hz

# p, t, filtDelay = srrcFunction(beta, L, span, Tsym = Tsym)
# p = np.pad(p, (0, L*N - p.size))

t, p = commpy.filters.rrcosfilter(L*N , beta, Tsym, L/Tsym)
p = p / np.sqrt(np.sum(np.power(p, 2)))
# t = t[:-1]

# 对时域采样信号, 执行快速傅里叶变换 FFT
FFTN =  p.size       ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
## IFFT
IX = scipy.fftpack.ifft(scipy.fftpack.fft(p))

f, X, A, Pha, R, I = freqDomainView(p, Fs, FFTN = FFTN, type = 'single')
A[0] = 2*A[0]

f1, X1, A1, Pha1, R1, I1 = freqDomainView(p, Fs, FFTN = FFTN, type = 'double')

##==================================================
#     频率刻度错位
#==================================================
X2 = scipy.fftpack.fft(p, n = FFTN)
# 消除相位混乱
threshold = np.max(np.abs(X2)) / 10000
X2[np.abs(X2) < threshold] = 0

# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
X2 = X2/p.size             # 将频域序列 X 除以序列的长度 N

# 计算频域序列 Y 的幅值和相角
A2 = np.abs(X2);                       # 计算频域序列 Y 的幅值
Pha2 = np.angle(X2,deg=True)       # 计算频域序列 Y 的相角 (弧度制)
R2 = np.real(X2)                   # 计算频域序列 Y 的实部
I2 = np.imag(X2)                   # 计算频域序列 Y 的虚部

df = Fs/FFTN                           # 频率间隔
# f2 = scipy.fftpack.fftfreq(FFTN, 1/Fs)
f2 = np.arange(-FFTN/2, FFTN/2)*df

###============================ 开始画图 ============================
width = 4
high = 3
horvizen = 5
vertical = 3
fig, axs = plt.subplots(vertical, horvizen, figsize=(horvizen*width, vertical*high), constrained_layout=True)
labelsize = 20

##半谱图
#======================================= 0,0 =========================================
axs[0,0].plot(t, p, color='b', linestyle='-', label='原始信号值',)

axs[0,0].set_xlabel(r'时间(s)', )
axs[0,0].set_ylabel(r'原始信号值', )

legend1 = axs[0,0].legend(loc='best', borderaxespad=0,  edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0,0].set_xlim(-Tsym*4, Tsym*4)  #拉开坐标轴范围显示投影

#======================================= 0,1 =========================================
axs[0,1].plot(f, A, color='r', linestyle='-', label='幅度',)

axs[0,1].set_xlabel(r'频率(Hz)', )
axs[0,1].set_ylabel(r'幅度', )
#axs[0,0].set_title('信号值', fontproperties=font3)

legend1 = axs[0,1].legend(loc='best', borderaxespad=0,  edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')                  # 设置图例legend背景透明

axs[0,1].set_xlim(-f_max - 0.2, f_max + 0.2)  # 拉开坐标轴范围显示投影
#======================================= 0,2 =========================================
axs[0,2].plot(f, Pha, color='g', linestyle='-', label='相位',)

axs[0,2].set_xlabel(r'频率(Hz)', )
axs[0,2].set_ylabel(r'相位', )

legend1 = axs[0,2].legend(loc='best', borderaxespad=0,  edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 0,3 =========================================
axs[0,3].plot(f, R, color='cyan', linestyle='-', label='实部',)

axs[0,3].set_xlabel(r'频率(Hz)', )
axs[0,3].set_ylabel(r'实部', )

legend1 = axs[0,3].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 0,4 =========================================
axs[0,4].plot(f, I, color='#FF8C00', linestyle='-', label='虚部',)

axs[0,4].set_xlabel(r'频率(Hz)', )
axs[0,4].set_ylabel(r'虚部', )

legend1 = axs[0,4].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#### 半谱图
#======================================= 1,0 =========================================
axs[1,0].plot(t, IX, color='b', linestyle='-', label='恢复的信号值',)

axs[1,0].set_xlabel(r'时间(s)', )
axs[1,0].set_ylabel(r'逆傅里叶变换信号值', )

legend1 = axs[1,0].legend(loc='best', borderaxespad=0,  edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,0].set_xlim(-Tsym*4, Tsym*4)  #拉开坐标轴范围显示投影

#======================================= 1,1 =========================================
axs[1,1].plot(f1, A1, color='r', linestyle='-', label='幅度',)

axs[1,1].set_xlabel(r'频率(Hz)', )
axs[1,1].set_ylabel(r'幅度', )

legend1 = axs[1,1].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# axs[1,1].set_xlim(-f_max - 0.2, f_max + 0.2)  # 拉开坐标轴范围显示投影
#======================================= 1,2 =========================================
axs[1,2].plot(f1, Pha1, color='g', linestyle='-', label='相位',)

axs[1,2].set_xlabel(r'频率(Hz)', )
axs[1,2].set_ylabel(r'相位', )

legend1 = axs[1,2].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 1,3 =========================================
axs[1,3].plot(f1, R1, color='cyan', linestyle='-', label='实部',)

axs[1,3].set_xlabel(r'频率(Hz)', )
axs[1,3].set_ylabel(r'实部', )

legend1 = axs[1,3].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 1,4 =========================================
axs[1,4].plot(f1, I1, color='#FF8C00', linestyle='-', label='虚部',)

axs[1,4].set_xlabel(r'频率(Hz)', )
axs[1,4].set_ylabel(r'虚部', )

legend1 = axs[1,4].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#### 频率刻度错位
#======================================= 2,0 =========================================
#======================================= 2,1 =========================================
axs[2,1].plot(f2, A2, color='r', linestyle='-', label='幅度',)

axs[2,1].set_xlabel(r'频率(Hz)', )
axs[2,1].set_ylabel(r'幅度', )

legend1 = axs[2,1].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 2,2 =========================================
axs[2,2].plot(f2, Pha2, color='g', linestyle='-', label='相位',)

axs[2,2].set_xlabel(r'频率(Hz)', )
axs[2,2].set_ylabel(r'相位', )

legend1 = axs[2,2].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 2,3 =========================================
axs[2,3].plot(f2, R2, color='cyan', linestyle='-', label='实部',)

axs[2,3].set_xlabel(r'频率(Hz)',  )
axs[2,3].set_ylabel(r'实部', )

legend1 = axs[2,3].legend(loc='best', borderaxespad=0,  edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 2,4 =========================================
axs[2,4].plot(f2, I2, color='#FF8C00', linestyle='-', label='虚部',)

axs[2,4].set_xlabel(r'频率(Hz)', )
axs[2,4].set_ylabel(r'虚部', )

legend1 = axs[2,4].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#================================= super ===============================================
out_fig = plt.gcf()
#out_fig.savefig(filepath2+'hh.eps',  bbox_inches='tight')
plt.show()
plt.close()

#%% Check Eq(23)(25)
# p, t, filtDelay = srrcFunction(beta, L, N, Tsym = Tsym)
# p = np.delete(p, p.size//2)

t, p = commpy.filters.rrcosfilter(L*N , beta, Tsym, L/Tsym)
p = p / np.sqrt(np.sum(np.power(p, 2)))

norm2p = np.linalg.norm(p)
FLN = FFTmatrix(L*N, L*N)

g = (N * (FLN@p) * (FLN.conj() @ p.conj())).real

g_1N = 1 - g[:N]
g_lastN = g[-N:]

np.abs(g_lastN - g_1N) ## ~= 0







































































































