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
axs[0].set_title("Raised Cosine pulse" )
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

    axs.plot(t, eyeVals, color = 'b',  label = '原始波形')
    axs.set_xlabel('t/$T_{sym}$',)
    axs.set_ylabel('Amplitude',)
    axs.set_title("Eye Plot" )
    plt.show()
    plt.close()
    return t, eyeVals

#%% Program 7.10: MPAM modulation
from Modulations import modulator
from ChannelModels import add_awgn_noise
# MPAM modulation
N = 100000
MOD_TYPE = "pam"
M = 4
modem, Es, bps = modulator(MOD_TYPE, M)
d = np.random.randint(0, M, N)
u = modem.modulate(d, inputtype = 'int')

##### plot
fig, axs = plt.subplots(1, 1, figsize = (6, 4), constrained_layout = True)
axs.stem(np.real(u), linefmt = 'r-', markerfmt = 'D', )
axs.set_title("PAM modulated symbols u(k)" )
axs.set_xlim(0, 20)  #拉开坐标轴范围显示投影
plt.show()
plt.close()

# Program 7.11: Upsampling
L = 4
v = np.vstack((u, np.zeros((L-1, u.size))))
v = v.T.flatten()

##### plot
fig, axs = plt.subplots(1, 1, figsize = (6, 4), constrained_layout = True)
axs.stem(np.real(v), linefmt = 'r-', markerfmt = 'D', )
axs.set_title("Oversampled symbols v(n)" )
axs.set_xlim(0, 20*L)  #拉开坐标轴范围显示投影
plt.show()
plt.close()

# Program 7.12: SRRC pulse shaping
beta = 0.3
span = 8
L = 4
p, t, filtDelay = srrcFunction(beta, L, span)
s = scipy.signal.convolve(v, p, 'full')
##### plot
fig, axs = plt.subplots(1, 1, figsize = (6, 4), constrained_layout = True)
axs.plot(np.real(s), 'r-', )
axs.set_title("Pulse shaped symbols s(n)" )
axs.set_xlim(0, 150)  #拉开坐标轴范围显示投影
plt.show()
plt.close()

# Program 7.13: Adding AWGN noise for given SNR value
EbN0dB = 100
snr = 10*np.log10(np.log2(M)) + EbN0dB
r, _ = add_awgn_noise(s, snr, L)
##### plot
fig, axs = plt.subplots(1, 1, figsize = (6, 4), constrained_layout = True)
axs.plot(np.real(r), 'r-', )
axs.set_title("Received signal r(n)" )
axs.set_xlim(0, 150)  #拉开坐标轴范围显示投影
plt.show()
plt.close()

# Program 7.14: Matched ﬁltering with SRRC pulse shape
vCap = scipy.signal.convolve(r, p, 'full')
##### plot
fig, axs = plt.subplots(1, 1, figsize = (6, 4), constrained_layout = True)
axs.plot(np.real(vCap), 'r-', )
axs.set_title("After matched filtering $\hat{v}(n)$" )
axs.set_xlim(0, 150)  #拉开坐标轴范围显示投影
plt.show()
plt.close()

t, eyeVals = plotEyeDiagram(vCap, L, 3*L, int(2*filtDelay), 100)

# Program 7.15: Symbol rate sampler and demodulation
uCap = vCap[int(2 * filtDelay) : int(vCap.size - 2*filtDelay) : L ] /L
##### plot
fig, axs = plt.subplots(1, 1, figsize = (6, 4), constrained_layout = True)
axs.stem(np.real(uCap), linefmt = 'r-', markerfmt = 'D', )
axs.set_title("After symbol rate sampler $\hat{u}$(n)" )
axs.set_xlim(0, 20)  #拉开坐标轴范围显示投影
plt.show()
plt.close()


dCap = modem.demodulate(uCap, outputtype = 'int',)


#%% Program 7.17: mpam srrc matched filtering.m: Performance simulation of an MPAM modulation based communication system with SRRC transmit and matched ﬁlters
from Chap6_PerformanceofDigitalModulations import ser_awgn

N = 100000
MOD_TYPE = 'pam'
M = 4
modem, Es, bps = modulator(MOD_TYPE, M)

EbN0dB = np.arange(-4, 26, 2 )
beta = 0.3
span = 8
L = 4
p, t, filtDelay = srrcFunction(beta, L, span)

SER_sim = np.zeros(EbN0dB.size)
snr = 10*np.log10(np.log2(M)) + EbN0dB

for i, snrdB in enumerate(snr):
    # transmiter
    d = np.random.randint(0, M, N)
    u = modem.modulate(d, inputtype = 'int')

    v = np.vstack((u, np.zeros((L-1, u.size))))
    v = v.T.flatten()
    s = scipy.signal.convolve(v, p, 'full')

    # channel
    r, _ = add_awgn_noise(s, snrdB, L)

    # receiver
    vCap = scipy.signal.convolve(r, p, 'full')
    uCap = vCap[int(2 * filtDelay) : int(vCap.size - 2*filtDelay) : L ] /L

    dCap = modem.demodulate(uCap, outputtype = 'int',)

    SER_sim[i] = np.sum(dCap != d)/N
SER_theory = ser_awgn(EbN0dB, MOD_TYPE, M)
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
axs.semilogy(EbN0dB, SER_sim, color = 'r', ls = 'none', marker = "o", ms = 12, )
axs.semilogy(EbN0dB, SER_theory, color = 'b', ls = '-', label = f'{M}-{MOD_TYPE.upper()}' )

axs.set_ylim(1e-6, 1)
axs.set_xlabel( 'Eb/N0(dB)',)
axs.set_ylabel('SER (Ps)',)
axs.set_title(f"Symbol Error Rate for M-{MOD_TYPE.upper()} over AWGN")
axs.legend(fontsize = 20)

plt.show()
plt.close()


#%% Program 7.18: PRSignaling.m: Function to generate PR signals at symbol sampling instants

def PRSignaling(Q, L, span):
    qn = scipy.signal.lfilter(Q, 1.0, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q = np.vstack((qn, np.zeros((L-1, qn.size))))
    q = q.T.flatten()
    Tsym = 1
    t = np.arange(-span/2, span/2, 1/L)
    g = np.sin(np.pi*t/Tsym)/(np.pi*t/Tsym)
    g[np.argwhere(np.isnan(g))] = 1
    b = scipy.signal.convolve(g, q, 'same')
    return b, t


#%% Program 7.19: test PRSignaling.m: PR Signaling schemes and their frequency domain responses
L = 50
span = 8
QD_arr = {}
QD_arr[0] = [1.0, 1.0]
QD_arr[1] = [1.0, -1.0]
QD_arr[2] = [1.0, 2.0, 1.0]
QD_arr[3] = [2.0, 1.0, -1.0]
QD_arr[4] = [1.0, 0.0, -1.0 ]
QD_arr[5] = [1.0, 1.0, -1.0, -1.0]
QD_arr[6] = [1.0, 2.0, 0.0, -2.0, -1.0]
QD_arr[7] = [1.0, 0.0, -2.0, 0.0, 1.0]

A = 1
titles = ['PR1 Duobinary', 'PR1 Dicode','PR Class II','PR Class III','PR4 Modified Duobinary','EPR4','E2PR4','PR Class V']

for i in range(8):
    Q = QD_arr[i]
    b, t = PRSignaling(Q, L, span)

    W, H = scipy.signal.freqz(Q, A, worN = 1024, whole = True)
    H = np.hstack((H[int(H.size/2):], H[:int(H.size/2)]))
    response = np.abs(H)
    norm_response = response/np.max(response)
    norm_freq = W/np.max(W) - 1/2

    fig, axs = plt.subplots(1, 2, figsize = (12, 4), constrained_layout = True)
    axs[0].stem(t[0:t.size:L], b[0:t.size:L], linefmt = 'r-', markerfmt = 'D', )
    axs[0].plot(t, b, 'b-', )
    axs[0].set_xlabel('t/$T_{sym}$',)
    axs[0].set_ylabel('b(t)',)
    axs[0].set_title(f"{titles[i]}-b(t)" )
    # axs[0].set_xlim(-4 , 4)  #拉开坐标轴范围显示投影
    # axs[0].legend()

    axs[1].plot(norm_freq, norm_response, 'b-', )
    axs[1].set_xlabel('f/$F_{sym}$',)
    axs[1].set_ylabel('|Q(D)|',)
    axs[1].set_title(f"{titles[i]}-Frequency response Q(D)" )
    # axs[1].set_xlim(-1.5 , 1.5)  #拉开坐标轴范围显示投影
    # axs[1].legend()

    plt.show()
    plt.close()


#%% Program 7.20: PR1 precoded system.m: Discrete-time equivalent partial response class 1 signaling model








































































































































































































































































































































































































































































































































































































































































































































































