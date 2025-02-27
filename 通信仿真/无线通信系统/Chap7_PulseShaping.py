#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 01:16:46 2025

@author: jack
"""

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


#%% Program 7.1: rectFunction.m: Function for generating a rectangular pulse
# Program 7.2: test rectPulse.m: Rectangular pulse and its manifestation in frequency domain
from Tools import freqDomainView

def rectFunction(L, span):
    # Function for generating rectangular pulse for the given inputs
    # L - oversampling factor (number of samples per symbol)
    # span - filter span in symbol durations
    # Returns the output pulse p(t) that spans the discrete-time base -span:1/L:span. Also returns the filter delay.

    Tsym = 1
    t = np.arange(-span/2, span/2 + 0.5/L, 1/L)
    p = np.logical_and(t > -Tsym/2, t <= Tsym/2).astype(int)
    filtDelay = (len(p)-1)/2
    return p, t, filtDelay

Tsym = 1
L = 16
span = 80
Fs = L/Tsym

p, t, filtDelay = rectFunction(L, span)
t = t*Tsym

f, Y, A, Pha, R, I = freqDomainView(p, Fs, type = 'double')

##### plot
fig, axs = plt.subplots(1, 2, figsize = (12, 4), constrained_layout = True)

# x
axs[0].plot(t, p, color = 'b',  label = '原始波形')
axs[0].set_xlabel('Time(s)',)
axs[0].set_ylabel('Amplitude',)
axs[0].set_title("Rectangular pulse" )
axs[0].set_xlim(-1.5 , 1.5  )  #拉开坐标轴范围显示投影

axs[1].plot(f, abs(A)/np.abs(A[int(len(A)/2+1)]), color = 'r', label = '载波信号')
axs[1].set_xlabel('Frequency(Hz)',)
axs[1].set_ylabel('Magnitude',)
axs[1].set_title("Frequency response")
# axs[1].legend()

plt.show()
plt.close()


#%% Program 7.3: sincFunction.m: Function for generating sinc pulse
# Program 7.4: test sincPulse.m: Sinc pulse and its manifestation in frequency domain

from Tools import freqDomainView

def sincFunction(L, span):
    # Function for generating rectangular pulse for the given inputs
    # L - oversampling factor (number of samples per symbol)
    # span - filter span in symbol durations
    # Returns the output pulse p(t) that spans the discrete-time base -span:1/L:span. Also returns the filter delay.

    Tsym = 1
    t = np.arange(-span/2, span/2 + 0.5/L, 1/L)
    p = np.sin(np.pi*t/Tsym) / (np.pi*t/Tsym)
    p[np.argwhere(np.isnan(p))] = 1
    filtDelay = (len(p)-1)/2
    return p, t, filtDelay


Tsym = 1
L = 16
span = 80
Fs = L/Tsym

p, t, filtDelay = sincFunction(L, span)
t = t*Tsym

f, Y, A, Pha, R, I = freqDomainView(p, Fs, type = 'double')

##### plot
fig, axs = plt.subplots(1, 2, figsize = (12, 4), constrained_layout = True)

# x
axs[0].plot(t, p, color = 'b',  label = '原始波形')
axs[0].set_xlabel('Time(s)',)
axs[0].set_ylabel('Amplitude',)
axs[0].set_title("Sinc pulse" )
axs[0].set_xlim(-10 , 10)  #拉开坐标轴范围显示投影

axs[1].plot(f, abs(A)/np.abs(A[int(len(A)/2+1)]), color = 'r', label = '载波信号')
axs[1].set_xlabel('Frequency(Hz)',)
axs[1].set_ylabel('Magnitude',)
axs[1].set_title("Frequency response")
axs[1].set_xlim(-2 , 2)  #拉开坐标轴范围显示投影

plt.show()
plt.close()

# Fig. 7.5: Fourier transform of truncated sinc pulse for various lengths kTsym
colors = ['b', 'r', 'k']
K = [2, 6, 25]
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

for i,k in enumerate(K):
    tmp1 = t - k*Tsym
    h = np.where(tmp1 == np.abs(tmp1).min())[0][0]
    tmp2 = t + k*Tsym
    l = np.where(tmp2 == np.abs(tmp2).min())[0][0]
    ptmp = p[l:h]
    f, Y, A, Pha, R, I = freqDomainView(ptmp, Fs, FFTN = 2048, type = 'double')

    axs.plot(f, 20*np.log10(abs(A)/np.abs(A[int(len(A)/2+1)])), color = colors[i], label = f'k = {k}')


axs.set_xlabel('Frequency(Hz)',)
axs.set_ylabel('20*log10(|S(t)|)',)
axs.set_title("Frequency response")
axs.set_xlim(-2 , 2)  #拉开坐标轴范围显示投影
axs.legend()


plt.show()
plt.close()


#%% Program 7.5: raisedCosineFunction.m: Function for generating raised-cosine pulse

def raisedCosineFunction(alpha, L, span):
    # Function for generating rectangular pulse for the given inputs
    # L - oversampling factor (number of samples per symbol)
    # span - filter span in symbol durations
    # Returns the output pulse p(t) that spans the discrete-time base -span:1/L:span. Also returns the filter delay.

    Tsym = 1
    t = np.arange(-span/2, span/2 + 0.5/L, 1/L)
    A = np.sin(np.pi*t/Tsym)/(np.pi*t/Tsym)
    B = np.cos(np.pi*alpha*t/Tsym)
    p = A*B/(1 - (2*alpha*t/Tsym)**2)
    p[np.argwhere(np.isnan(p))] = 1
    p[np.argwhere(np.isinf(p))] = (alpha/2)*np.sin(np.pi/(2*alpha))
    filtDelay = (len(p)-1)/2
    return p, t, filtDelay


# Program 7.6: test RCPulse.m: raised-cosine pulses and their manifestation in frequency domain
Tsym = 1
L = 10
span = 80
Fs = L/Tsym

colors = ['b', 'r', 'g', 'k', 'c']
alphas = [0.01, 0.3, 0.5, 1]
fig, axs = plt.subplots(1, 2, figsize = (12, 6), constrained_layout = True)

for i, alpha in enumerate(alphas):
    p, t, filtDelay = raisedCosineFunction(alpha, L, span)
    t = t*Tsym
    f, Y, A, Pha, R, I = freqDomainView(p, Fs, FFTN = 2048, type = 'double')
    # x
    axs[0].plot(t, p, color = colors[i],  label = f'alpha = {alpha:.2f}')
    axs[1].plot(f, abs(A)/np.abs(A[int(len(A)/2+1)]), color = colors[i], label = f'alpha = {alpha:.2f}')

axs[0].set_xlabel('Time(s)',)
axs[0].set_ylabel('Amplitude',)
axs[0].set_title("Sinc pulse" )
axs[0].set_xlim(-4 , 4)  #拉开坐标轴范围显示投影
axs[0].legend()

axs[1].set_xlabel('Frequency(Hz)',)
axs[1].set_ylabel('Magnitude',)
axs[1].set_title("Frequency response")
axs[1].set_xlim(-1.5 , 1.5)  #拉开坐标轴范围显示投影
axs[1].legend()

plt.show()
plt.close()


#%% Program 7.7: srrcFunction.m: Function for generating square-root raised-cosine pulse
# Program 7.8: test SRRCPulse.m: Square-root raised-cosine pulse characteristics
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
    return p, t, filtDelay

Tsym = 1
L = 10
span = 80
Fs = L/Tsym

colors = ['b', 'r', 'g', 'k', 'c']
betas = [0.01, 0.22, 0.5, 1]
fig, axs = plt.subplots(1, 2, figsize = (12, 6), constrained_layout = True)

for i, beta in enumerate(betas):
    ptx, t, filtDelay = srrcFunction(beta, L, span)
    prx = ptx
    comb_response = scipy.signal.convolve(ptx, prx, 'same')
    t = t*Tsym
    f, Y, A, Pha, R, I = freqDomainView(ptx, Fs, FFTN = 2048, type = 'double')
    # x
    axs[0].plot(t,  comb_response/np.max(np.abs(comb_response)), color = colors[i],  label = f'beta = {beta:.2f}')
    axs[1].plot(f, abs(A)/np.abs(A[int(len(A)/2+1)]), color = colors[i], label = f'beta = {beta:.2f}')

axs[0].set_xlabel('Time(s)',)
axs[0].set_ylabel('Amplitude',)
axs[0].set_title("Combined response of SRRC filters" )
axs[0].set_xlim(-4 , 4)  #拉开坐标轴范围显示投影
axs[0].legend()

axs[1].set_xlabel('Frequency(Hz)',)
axs[1].set_ylabel('Magnitude',)
axs[1].set_title("Frequency response (at Tx/Rx only)")
axs[1].set_xlim(-1.5 , 1.5)  #拉开坐标轴范围显示投影
axs[1].legend()

plt.show()
plt.close()



# Program 7.9: plotEyeDiagram.m: Function for plotting eye diagram
def plotEyeDiagram(x, L, nSamples, offset, nTraces):
    M = 4
    tnSamp = nSamples * M * nTraces
    t = np.arange(x.size)
    t1 = np.arange(M * x.size) / M
    y = np.interp(t1, t, x)
    eyeVals = y[M*offset:(M*offset + tnSamp)].reshape(nSamples * M, nTraces)
    t = np.arange(0, M*nSamples, 1) / (M*L)

    ##### plot
    fig, axs = plt.subplots(1, 1, figsize = (6, 4), constrained_layout = True)

    # x
    axs[0].plot(t, eyeVals, color = 'b',  label = '原始波形')
    axs[0].set_xlabel('t/$T_{sym}$',)
    axs[0].set_ylabel('Amplitude',)
    axs[0].set_title("Eye Plot" )
    axs[0].set_xlim(-1.5 , 1.5  )  #拉开坐标轴范围显示投影

    plt.show()
    plt.close()

    return eyeVals

#%% 7.5 Implementing a matched ﬁlter system with SRRC ﬁltering
from Modulations import modulator






#%% Program 7.17: mpam srrc matched filtering.m: Performance simulation of an MPAM modulation based communication system with SRRC transmit and matched ﬁlters








































































































































































































































































































































































































































































































































































































































































































































































































