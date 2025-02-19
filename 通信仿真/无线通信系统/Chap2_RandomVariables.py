#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:25:56 2025

@author: jack
<Wireless communication systems in Matlab> Chap2
"""
#%%
import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import commpy

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22  # 设置坐标轴标题字体大小

plt.rcParams['axes.labelsize'] = 22  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22


#%% 2.5 Generating correlated random variables

# 2.5.2 Generating multiple sequences of correlated random variables using Cholesky decomposition

C = np.array([[1,0.5,0.3],[0.5,1,0.3],[0.3,0.3,1]])

L = np.linalg.cholesky(C)
U = L.T

R = np.random.randn(100000, 3)
Rc = R@U

X = Rc[:,0]
Y = Rc[:,1]
Z = Rc[:,2]

C_hat = np.cov(Rc.T)
print("相关系数矩阵=\n", C_hat)
corr = np.corrcoef(Rc.T,Rc.T)
print("相关系数矩阵=\n", corr)


##### plot
fig, axs = plt.subplots(1, 3, figsize = (12, 4), constrained_layout = True)

# x
axs[0].scatter(X, X, color = 'b',  label = '原始波形')
axs[0].set_xlabel('X',)
axs[0].set_ylabel('X',)
lb = 'X and X, ' + r'$\rho = {:.2f}$'.format(C_hat[0,0])
axs[0].set_title(lb )
# axs[0].legend()

axs[1].scatter(X, Y, color = 'r', label = '载波信号')
axs[1].set_xlabel('X',)
axs[1].set_ylabel('Y',)
lb = 'X and Y, ' + r'$\rho = {:.2f}$'.format(C_hat[0,1])
axs[1].set_title(lb)
# axs[1].legend()

axs[2].scatter(X, Z, color = 'gray', label = '幅度调制信号')
axs[2].set_xlabel('X',)
axs[2].set_ylabel('Z',)
lb = 'X and Z, ' + r'$\rho = {:.2f}$'.format(C_hat[0,2])
axs[2].set_title(lb )
# axs[2].legend()

plt.show()
plt.close()

#%% 2.6 Generating correlated Gaussian sequences
#%% 2.6.1 Spectral factorization method
from Tools import freqDomainView

def Jakes_filter(fd, Ts, N):
    #  FIR channel shaping filter with Jakes doppler spectrum %S(f) = 1/ [sqrt(1-(f/f_max)ˆ2)*pi*f_max]
    # Use impulse response obtained from spectral factorization
    # Input parameters are  fd = maximum doppler frequency in Hz
    # Ts = sampling interval in seconds
    # N = desired filter order
    # Returns the windowed impulse response of the FIR filter %Example: fd=10; Ts=0.01; N=512; Jakes_filter(fd,Ts,N)
    L = N/2
    n = np.arange(1, L+1)
    J_pos = (scipy.special.jv(0.25, 2 * np.pi * fd * n * Ts)/(n**0.25))[None,:]
    Jneg = np.fliplr(J_pos)
    J0 = np.array([[1.468813 * (fd*Ts)**(0.25)]])
    J = np.hstack((Jneg, J0, J_pos))

    hamm = scipy.signal.windows.hamming(N+1)
    # hamm = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0, N+1)/N)
    hw = J * hamm
    hw = hw / np.sqrt(np.sum(np.abs(hw)**2))

    f, Y, A, Pha, R, I = freqDomainView(hw, 1/Ts, 'double' )
    Hw = (Ts / hw.size) * np.abs(Y)**2 * (hw.size**2)

    ##### plot
    fig, axs = plt.subplots(1, 2, figsize = (12, 5), constrained_layout = True)

    axs[0].plot(np.arange(hw.size), hw.flatten(), lw = 1, color = 'b',  label = ' ')
    axs[0].set_xlabel('samples(n)',)
    axs[0].set_ylabel(r'$h_w[n]$',)
    axs[0].set_title('Windowed impulse response' )
    # axs[0].legend()

    axs[1].plot(f, Hw.flatten(), color = 'r', lw = 1, label = ' ')
    axs[1].set_xlabel('frequency(Hz)',)
    axs[1].set_ylabel(r'$|H_w(f)|^2$',)
    axs[1].set_title('Jakes Spectrum')
    axs[1].set_xlim(-20,20)
    plt.suptitle(f"Impulse response & spectrum of windowed Jakes filter ( fmax = {fd}Hz, Ts = {Ts}s, N = {N})", fontsize = 22)
    plt.show()
    plt.close()

    return hw

fd = 10
Fs = 100
N = 512
Ts = 1/Fs

h = Jakes_filter(fd, Ts, N)
x = np.random.randn( 1, 10000)
y = scipy.signal.convolve(x, h, mode = 'valid')
f, Y, A, Pha, R, I = freqDomainView(y, 1/Ts, 'double' )
Syy = (Ts / y.size) * np.abs(Y)**2 * (y.size)**2
##### plot
fig, axs = plt.subplots(1, 2, figsize = (12, 5), constrained_layout = True)

axs[0].plot(np.arange(y.size), np.log10(np.abs(y.flatten())), lw = 1, color = 'b',  label = ' ')
axs[0].set_xlabel('samples(n)',)
axs[0].set_ylabel(r'$log|y(n)|$',)
axs[0].set_title('Envelop function' )
# axs[0].legend()

axs[1].plot(f, Syy.flatten(), color = 'r', lw = 1, label = ' ')
axs[1].set_xlabel('frequency(Hz)',)
axs[1].set_ylabel(r'$|S_{yy}(f)|^2$',)
axs[1].set_title('Power Spectral Density')
axs[1].set_xlim(-20,20)
plt.suptitle(f"Simulated colored noise samples and its power spectral density (fmax = {fd} Hz)", fontsize = 22)

plt.show()
plt.close()


#%% 2.6.2 Auto-Regressive (AR) model









































































































































































































































































































































































































































































