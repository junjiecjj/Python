#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:28:05 2025

@author: jack
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
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

#%%
def freqDomainView(x, Fs, FFTN = None, type = 'double'): # N为偶数
    if FFTN == None:
        FFTN = 2**int(np.ceil(np.log2(x.size)))
    X = scipy.fftpack.fft(x, n = FFTN)
    # 消除相位混乱
    threshold = np.max(np.abs(X)) / 10000
    X[np.abs(X) < threshold] = 0
    # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
    X = X/x.size               # 将频域序列 X 除以序列的长度 N
    if type == 'single':
        Y = X[0 : int(FFTN/2)+1].copy()       # 提取 X 里正频率的部分,N为偶数
        Y[1 : int(FFTN/2)] = 2*Y[1 : int(FFTN/2)].copy()
        f = np.arange(0, int(FFTN/2)+1) * (Fs/FFTN)
        # 计算频域序列 Y 的幅值和相角
        A = abs(Y)                        # 计算频域序列 Y 的幅值
        Pha = np.angle(Y, deg=1)          # 计算频域序列 Y 的相角 (弧度制)
        R = np.real(Y)                    # 计算频域序列 Y 的实部
        I = np.imag(Y)                    # 计算频域序列 Y 的虚部
    elif type == 'double':
        f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/Fs))
        Y = scipy.fftpack.fftshift(X, )
        # 计算频域序列 Y 的幅值和相角
        A = abs(Y)                        # 计算频域序列 Y 的幅值
        Pha = np.angle(Y, deg=1)          # 计算频域序列 Y 的相角 (弧度制)
        R = np.real(Y)                    # 计算频域序列 Y 的实部
        I = np.imag(Y)                    # 计算频域序列 Y 的虚部

    return f, Y, A, Pha, R, I

def srrcFunction(beta, L, span):
    # Function for generating rectangular pulse for the given inputs
    # L - oversampling factor (number of samples per symbol)
    # span - filter span in symbol durations
    # Returns the output pulse p(t) that spans the discrete-time base -span:1/L:span. Also returns the filter delay.
    Tsym = 1
    t = np.arange(-span/2, span/2 + 0.5/L, 1/L)
    A = np.sin(np.pi*t*(1-beta)/Tsym) + 4*beta*t/Tsym * np.cos(np.pi*t*(1+beta)/Tsym)
    B = np.pi*t/Tsym * (1-(4*beta*t/Tsym)**2)
    p = 1/np.sqrt(Tsym) * A/B
    p[np.argwhere(np.isnan(p))] = 1
    p[np.argwhere(np.isinf(p))] = beta/(np.sqrt(2*Tsym)) * ((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
    filtDelay = (len(p)-1)/2
    p = p / np.sqrt(np.sum(np.power(p, 2))) # both Add and Delete this line is OK.
    return p, t, filtDelay

#%% ======================================================
## ===========  定义时域采样信号
## ======================================================
Tsym = 1                          #
B0  = 1/(2*Tsym)                  # Hz
beta = 0.01
B = (1 + beta) * B0
# f_max = 2*np.pi*B               # 角频率rad/s,
f_max = B                         # 画图用的时间频率 Hz

# Fs = 100                          # 信号采样频率
# Ts = 1/Fs                         # 采样时间间隔
# N = 100                         # 采样信号的长度
# m = 30
# t = np.arange(-m*Tsym, m*Tsym, Ts)    # 定义信号采样的时间点 t
span = 8
L = 4
x, t, filtDelay = srrcFunction(beta, L, span)
# N = x.size
#=====================================================


#%%==================================================
# 半谱图
#==================================================
# 对时域采样信号, 执行快速傅里叶变换 FFT
FFTN = 1024        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
## IFFT
IX = scipy.fftpack.ifft(scipy.fftpack.fft(x ))

Fs = L/Tsym
f, Y, A, Pha, R, I = freqDomainView(x, Fs, FFTN = FFTN, type = 'single')
#%%==================================================
# 全谱图
#==================================================
f1, Y1, A1, Pha1, R1, I1 = freqDomainView(x, Fs, FFTN = FFTN, type = 'double')

#%%==================================================
#     频率刻度错位
#==================================================
X2 = scipy.fftpack.fft(x, n = FFTN)

# 消除相位混乱
X2[np.abs(X2)<1e-8] = 0        # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
X2 = X2/x.size             # 将频域序列 X 除以序列的长度 N

# 计算频域序列 Y 的幅值和相角
A2 = np.abs(X2);                       # 计算频域序列 Y 的幅值
Pha2 = np.angle(X2,deg=True)       # 计算频域序列 Y 的相角 (弧度制)
R2 = np.real(X2)                   # 计算频域序列 Y 的实部
I2 = np.imag(X2)                   # 计算频域序列 Y 的虚部

df = Fs/FFTN                           # 频率间隔
f2 = scipy.fftpack.fftfreq(FFTN, 1/Fs)

#%%====================================== 开始画图 ===============================================
width = 4
high = 3
horvizen = 5
vertical = 3
fig, axs = plt.subplots(vertical, horvizen, figsize=(horvizen*width, vertical*high), constrained_layout=True)
labelsize = 20

#%% 半谱图
#======================================= 0,0 =========================================
axs[0,0].plot(t, x, color='b', linestyle='-', label='原始信号值',)

axs[0,0].set_xlabel(r'时间(s)', )
axs[0,0].set_ylabel(r'原始信号值', )

legend1 = axs[0,0].legend(loc='best', borderaxespad=0,  edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,0].get_xticklabels() + axs[0,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

# axs[0,0].set_xlim(-Tsym*4, Tsym*4)  #拉开坐标轴范围显示投影

#======================================= 0,1 =========================================
axs[0,1].plot(f, A, color='r', linestyle='-', label='幅度',)

axs[0,1].set_xlabel(r'频率(Hz)', )
axs[0,1].set_ylabel(r'幅度', )
#axs[0,0].set_title('信号值', fontproperties=font3)

legend1 = axs[0,1].legend(loc='best', borderaxespad=0,  edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')                  # 设置图例legend背景透明

axs[0,1].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,1].get_xticklabels() + axs[0,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

# axs[0,1].set_xlim(-f_max - 0.2, f_max + 0.2)  # 拉开坐标轴范围显示投影
#======================================= 0,2 =========================================
axs[0,2].plot(f, Pha, color='g', linestyle='-', label='相位',)

axs[0,2].set_xlabel(r'频率(Hz)', )
axs[0,2].set_ylabel(r'相位', )

legend1 = axs[0,2].legend(loc='best', borderaxespad=0,  edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0,2].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,2].get_xticklabels() + axs[0,2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 0,3 =========================================
axs[0,3].plot(f, R, color='cyan', linestyle='-', label='实部',)

axs[0,3].set_xlabel(r'频率(Hz)', )
axs[0,3].set_ylabel(r'实部', )

legend1 = axs[0,3].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0,3].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,3].get_xticklabels() + axs[0,3].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 0,4 =========================================
axs[0,4].plot(f, I, color='#FF8C00', linestyle='-', label='虚部',)

axs[0,4].set_xlabel(r'频率(Hz)', )
axs[0,4].set_ylabel(r'虚部', )

legend1 = axs[0,4].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[0,4].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,4].get_xticklabels() + axs[0,4].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#%% 半谱图
#======================================= 1,0 =========================================
axs[1,0].plot(t, IX, color='b', linestyle='-', label='恢复的信号值',)

axs[1,0].set_xlabel(r'时间(s)', )
axs[1,0].set_ylabel(r'逆傅里叶变换信号值', )

legend1 = axs[1,0].legend(loc='best', borderaxespad=0,  edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[1,0].get_xticklabels() + axs[1,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 1,1 =========================================
axs[1,1].plot(f1, A1, color='r', linestyle='-', label='幅度',)

axs[1,1].set_xlabel(r'频率(Hz)', )
axs[1,1].set_ylabel(r'幅度', )

legend1 = axs[1,1].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,1].tick_params(direction='in', axis='both', top=True, right=True, labelsize=labelsize, width=3,)
labels = axs[1,1].get_xticklabels() + axs[1,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号
axs[1,1].set_xlim(-f_max - 0.2, f_max + 0.2)  # 拉开坐标轴范围显示投影
#======================================= 1,2 =========================================
axs[1,2].plot(f1, Pha1, color='g', linestyle='-', label='相位',)

axs[1,2].set_xlabel(r'频率(Hz)', )
axs[1,2].set_ylabel(r'相位', )

legend1 = axs[1,2].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,2].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[1,2].get_xticklabels() + axs[1,2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 1,3 =========================================
axs[1,3].plot(f1, R1, color='cyan', linestyle='-', label='实部',)

axs[1,3].set_xlabel(r'频率(Hz)', )
axs[1,3].set_ylabel(r'实部', )

legend1 = axs[1,3].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,3].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[1,3].get_xticklabels() + axs[1,3].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 1,4 =========================================
axs[1,4].plot(f1, I1, color='#FF8C00', linestyle='-', label='虚部',)

axs[1,4].set_xlabel(r'频率(Hz)', )
axs[1,4].set_ylabel(r'虚部', )

legend1 = axs[1,4].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[1,4].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[1,4].get_xticklabels() + axs[1,4].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


#%% 频率刻度错位
#======================================= 2,0 =========================================

#======================================= 2,1 =========================================
axs[2,1].plot(f2, A2, color='r', linestyle='-', label='幅度',)

axs[2,1].set_xlabel(r'频率(Hz)', )
axs[2,1].set_ylabel(r'幅度', )

legend1 = axs[2,1].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[1,1].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[2,1].get_xticklabels() + axs[2,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 2,2 =========================================
axs[2,2].plot(f2, Pha2, color='g', linestyle='-', label='相位',)

axs[2,2].set_xlabel(r'频率(Hz)', )
axs[2,2].set_ylabel(r'相位', )

legend1 = axs[2,2].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[2,2].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[2,2].get_xticklabels() + axs[2,2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 2,3 =========================================
axs[2,3].plot(f2, R2, color='cyan', linestyle='-', label='实部',)

axs[2,3].set_xlabel(r'频率(Hz)',  )
axs[2,3].set_ylabel(r'实部', )

legend1 = axs[2,3].legend(loc='best', borderaxespad=0,  edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[2,3].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[2,3].get_xticklabels() + axs[2,3].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 2,4 =========================================
axs[2,4].plot(f2, I2, color='#FF8C00', linestyle='-', label='虚部',)

axs[2,4].set_xlabel(r'频率(Hz)', )
axs[2,4].set_ylabel(r'虚部', )

legend1 = axs[2,4].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[2,4].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[2,4].get_xticklabels() + axs[2,4].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


#================================= super ===============================================
out_fig = plt.gcf()
#out_fig.savefig(filepath2+'hh.eps',  bbox_inches='tight')
plt.show()
plt.close()





