#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 01:16:48 2025

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

#%% Program 8.1: Channel model


nSamp = 5
Fs = 100
Ts = 1/Fs
Tsym = nSamp * Ts
k = 6
N0 = 0.001

t = np.arange(-k*Tsym, k*Tsym+Ts/2, Ts)
h_t = 1/(1 + (t/Tsym)**2 )
h_t = h_t + N0 * np.random.randn(h_t.size)
h_k = h_t[0:h_t.size:nSamp]
t_inst = t[0:h_t.size:nSamp]

##### plot
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

# x
axs.plot(t, h_t, color = 'b',  label = 'continuous-time model')
axs.stem(t_inst, h_k, linefmt = 'r-', markerfmt = 'D', label = 'discrete-time model')
axs.set_xlabel('Time(s)',)
axs.set_ylabel('Amplitude',)
axs.set_title("Channel impulse response" )
# axs[0].set_xlim(-10 , 10)  #拉开坐标轴范围显示投影

plt.show()
plt.close()

#%% Program 8.3: zf equalizer.m: Function to design ZF equalizer and to equalize a signal
def convMatrix(h, p):
    # Construct the convolution matrix of size (N+p-1)x p from the input matrix h of size N.
    h = h.flatten()
    col = np.hstack((h, np.zeros(p-1)))
    row = np.hstack((h[0], np.zeros(p-1)))
    H = scipy.linalg.toeplitz(col, row)
    return H

def zf_equalizer(h, N, delay = None):
    h = h.flatten()
    L = h.size
    H = convMatrix(h, N)
    Hp = scipy.linalg.inv(H.T@H)@H.T
    optDelay = (np.diag(H@Hp)).argmax()
    k0 = optDelay
    if delay != None:
        if delay >= (L+N-1):
            print("Too large delay")
        k0 = delay
    k0 = int(k0)
    d = np.zeros((N+L-1,1))
    d[k0] = 1
    w = Hp @ d
    err = 1-H[k0,:] @ w
    MSE = (1 - d.T @ H @ Hp @ d)

    return w, err, optDelay, MSE

# Program 8.4: Design and test the zero-forcing equalizer
N = 14
delay = 11
w, error, k0, _ = zf_equalizer(h_k, N, delay)
w = w.flatten()
r_k = h_k
d_k = scipy.signal.convolve(w, r_k,)
h_sys = scipy.signal.convolve(w, h_k)
print(f"ZF Equalizer Design: N= {N}, Delay = {delay}, Error = {error}, ZF equalizer weights:{w}")

# Program 8.5: Frequency responses of channel equalizer and overall system
Omega_1, H_F = scipy.signal.freqz(h_k)
Omega_2, W = scipy.signal.freqz(w)
Omega_3, H_sys = scipy.signal.freqz(h_sys)

##### plot
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

# x
axs.plot(Omega_1/np.pi, 20*np.log10(np.abs(H_F)/np.max(np.abs(H_F))), color = 'g',  label = 'channel')
axs.plot(Omega_2/np.pi, 20*np.log10(np.abs(W)/np.max(np.abs(W))), color = 'r',  label = 'ZF equalizer')
axs.plot(Omega_3/np.pi, 20*np.log10(np.abs(H_sys)/np.max(np.abs(H_sys))), color = 'k',  label = 'overall system')
axs.set_xlabel('Normalized frequency(x $\pi$ rad/sample)',)
axs.set_ylabel('Magnitude(dB)',)
axs.set_title("Frequency response" )
axs.legend(fontsize = 20)
plt.show()
plt.close()

# Program 8.6: Design and test the zero-forcing equalizer
fig, axs = plt.subplots(1, 2, figsize = (12, 4), constrained_layout = True)
axs[0].stem(np.arange(r_k.size), r_k, linefmt = 'r-', markerfmt = 'D', )

axs[0].set_xlabel('Samples',)
axs[0].set_ylabel('Amplitude',)
axs[0].set_title("Equalizer input" )
# axs[0].set_xlim(-4 , 4)  #拉开坐标轴范围显示投影
# axs[0].legend()

axs[1].stem(np.arange(d_k.size), d_k, linefmt = 'r-', markerfmt = 'D', )
axs[1].set_xlabel('Samples',)
axs[1].set_ylabel('Amplitude',)
axs[1].set_title(f"Equalizer output- N={N}, Delay = {delay}, Error = {error}" )
# axs[1].set_xlim(-1.5 , 1.5)  #拉开坐标轴范围显示投影
# axs[1].legend()

plt.show()
plt.close()


#%% Program 8.8: mmse equalizer.m: Function to design a delay-optimized MMSE equalizer
### Program 8.9: mmse equalizer test.m: Simulation of MMSE equalizer
def mmse_equalizer(h, snr, N, delay = None):
    h = h.flatten()
    L = h.size
    H = convMatrix(h, N)
    gamma = 10**(-snr/10)
    optDelay = (np.diag(H @ scipy.linalg.inv(H.T@H + gamma * np.eye(N)) @ H.T)).argmax()

    k0 = optDelay
    if delay != None:
        if delay >= (L+N-1):
            print("Too large delay")
        k0 = delay
    k0 = int(k0)
    d = np.zeros((N+L-1,1))
    d[k0] = 1
    w = scipy.linalg.inv(H.T@H + gamma * np.eye(N) ) @ H.T @ d

    MSE = 1 - d.T @ H @ scipy.linalg.inv(H.T@H + gamma * np.eye(N) ) @ H.T @ d
    return w, MSE, optDelay

nSamp = 5
Fs = 100
Ts = 1/Fs
Tsym = nSamp * Ts
k = 6
N0 = 0.1

t = np.arange(-k*Tsym, k*Tsym+Ts/2, Ts)
h_t = 1/(1 + (t/Tsym)**2 )
h_t = h_t + N0 * np.random.randn(h_t.size)
h_k = h_t[0:h_t.size:nSamp]
t_k = t[0:h_t.size:nSamp]

##### plot
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
axs.plot(t, h_t, color = 'b',  label = 'continuous-time model')
axs.stem(t_k, h_k, linefmt = 'r-', markerfmt = 'D', label = 'discrete-time model')
axs.set_xlabel('Time(s)',)
axs.set_ylabel('Amplitude',)
axs.set_title("Channel impulse response" )
plt.show()
plt.close()


nTaps = 14
noisevar = N0**2
snr = 10*np.log10(1/N0)
w, error, optDelay = mmse_equalizer(h_k, snr, nTaps)
w = w.flatten()
r_k = h_k
d_k = scipy.signal.convolve(w, r_k,)
h_sys = scipy.signal.convolve(w, h_k)
print(f"MMSE Equalizer Design: N= {nTaps}, Delay = {delay}, Error = {error}, MMSE equalizer weights:{w}")


##### plot
fig, axs = plt.subplots(1, 2, figsize = (12, 4), constrained_layout = True)
axs[0].stem(np.arange(r_k.size), r_k, linefmt = 'r-', markerfmt = 'D', )

axs[0].set_xlabel('Samples',)
axs[0].set_ylabel('Amplitude',)
axs[0].set_title("Equalizer input" )
# axs[0].set_xlim(-4 , 4)  #拉开坐标轴范围显示投影
# axs[0].legend()

axs[1].stem(np.arange(d_k.size), d_k, linefmt = 'r-', markerfmt = 'D', )
axs[1].set_xlabel('Samples',)
axs[1].set_ylabel('Amplitude',)
axs[1].set_title(f"Equalizer output- N={N}, Delay = {optDelay}, Error = {error}" )
# axs[1].set_xlim(-1.5 , 1.5)  #拉开坐标轴范围显示投影
# axs[1].legend()
plt.show()
plt.close()


Omega_1, H_F = scipy.signal.freqz(h_k)
Omega_2, W = scipy.signal.freqz(w)
Omega_3, H_sys = scipy.signal.freqz(h_sys)

##### plot
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

# x
axs.plot(Omega_1/np.pi, 20*np.log10(np.abs(H_F)/np.max(np.abs(H_F))), color = 'g',  label = 'channel')
axs.plot(Omega_2/np.pi, 20*np.log10(np.abs(W)/np.max(np.abs(W))), color = 'r',  label = 'ZF equalizer')
axs.plot(Omega_3/np.pi, 20*np.log10(np.abs(H_sys)/np.max(np.abs(H_sys))), color = 'k',  label = 'overall system')
axs.set_xlabel('Normalized frequency(x $\pi$ rad/sample)',)
axs.set_ylabel('Magnitude(dB)',)
axs.set_title("Frequency response" )
axs.legend(fontsize = 20)
plt.show()
plt.close()

#%% Program 8.11: bpsk isi channels equalizers.m: Performance of FIR linear equalizers over ISI channels

# from Chap6_PerformanceofDigitalModulations import ser_awgn
from Modulations import modulator
from ChannelModels import add_awgn_noise

def Qfun(x):
    return 0.5 * scipy.special.erfc(x / np.sqrt(2))

def ser_awgn(EbN0dB, MOD_TYPE, M, COHERENCE = None):
    EbN0 = 10**(EbN0dB/10)
    EsN0 = np.log2(M) * EbN0
    SER = np.zeros(EbN0dB.size)
    if MOD_TYPE.lower() == "bpsk":
        SER = Qfun(np.sqrt(2 * EbN0))
    elif MOD_TYPE == "psk":
        if M == 2:
            SER = Qfun(np.sqrt(2 * EbN0))
        else:
            if M == 4:
                SER = 2 * Qfun(np.sqrt(2* EbN0)) - Qfun(np.sqrt(2 * EbN0))**2
            else:
                SER = 2 * Qfun(np.sin(np.pi/M) * np.sqrt(2 * EsN0))
    elif MOD_TYPE.lower() == "qam":
        SER = 1 - (1 - 2*(1 - 1/np.sqrt(M)) * Qfun(np.sqrt(3 * EsN0/(M - 1))))**2
    elif MOD_TYPE.lower() == "pam":
        SER = 2*(1-1/M) * Qfun(np.sqrt(6*EsN0/(M**2-1)))
    return SER

h_cA = np.array([0.04, -0.05, 0.07, -0.21, -0.5, 0.72, 0.36, 0.21, 0.03, 0.07]) #  Channel A
h_cB = np.array([0.407, 0.815, 0.407])               #  Channel B
h_cC = np.array([0.227, 0.460, 0.688, 0.460, 0.227]) #  Channel C

N = 100000
EbN0dB = np.arange(0, 32, 2)

ntaps = 31
MOD_TYPE = "psk"
M = 2
k = int(np.log2(M))
modem, Es, bps = modulator(MOD_TYPE, M)
map_table, demap_table = modem.getMappTable()

uu = np.random.randint(0, 2, size = N * bps).astype(np.int8)
s = modem.modulate(uu)
d = np.array([demap_table[sym] for sym in s])

channelTypes = ["Channel A", "Channel B", "Channel C" ]
H_C = {}
H_C[0] = h_cA
H_C[1] = h_cB
H_C[2] = h_cC
markers = ['none', "o", 'v', ]

colors = ['k', 'r', 'b']
for idx, channeltype in enumerate(channelTypes):
    h_c = H_C[idx]
    F, H = scipy.signal.freqz(h_c)
    ##### plot
    fig, axs = plt.subplots(1, 2, figsize = (12, 4), constrained_layout = True)
    axs[0].stem(h_c, linefmt = f'{colors[idx]}-', markerfmt = 'D', )
    axs[0].set_xlabel('Time(s)',)
    axs[0].set_ylabel('h(t)',)
    axs[0].set_title(f"Channel impulse response, {channeltype}" )
    axs[1].plot(F , 20*np.log10(np.abs(H)/np.max(np.abs(H))), color = colors[idx],  )
    axs[1].set_xlabel('Samples',)
    axs[1].set_ylabel('Amplitude',)
    axs[1].set_title( "Frequency response" )
    plt.show()
    plt.close()

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
SER_theory = ser_awgn(EbN0dB, MOD_TYPE, M)
for idx, channeltype in enumerate(channelTypes):
    SER_zf = np.zeros(EbN0dB.size)
    SER_mmse = np.zeros(EbN0dB.size)
    h_c = H_C[idx]
    x = scipy.signal.convolve(s, h_c)
    for i, EbN0db in enumerate(EbN0dB):
        ## channel
        r, _ = add_awgn_noise(x, EbN0db)

        ## Receiver
        ## MMSE equalizer
        h_mmse, mse, optDelay = mmse_equalizer(h_c, EbN0db, ntaps)
        y_mmse = scipy.signal.convolve(h_mmse.flatten(), r)
        y_mmse = y_mmse[optDelay:optDelay+N]

        sCap = modem.demodulate(y_mmse, 'hard')
        dCap_mmse = []
        for j in range(N):
            dCap_mmse.append( int(''.join([str(num) for num in sCap[j*k:(j+1)*k]]), base = 2) )
        dCap_mmse = np.array(dCap_mmse)

        ## ZF equalizer
        h_zf, error, optDelay, _ = zf_equalizer(h_c, ntaps)
        y_zf = scipy.signal.convolve(h_zf.flatten(), r)
        y_zf = y_zf[optDelay:optDelay+N]

        sCap = modem.demodulate(y_zf, 'hard')
        dCap_zf = []
        for j in range(N):
            dCap_zf.append( int(''.join([str(num) for num in sCap[j*k:(j+1)*k]]), base = 2) )
        dCap_zf = np.array(dCap_zf)

        SER_mmse[i] = np.sum(d != dCap_mmse)/d.size
        SER_zf[i] = np.sum(d != dCap_zf)/d.size
    axs.semilogy(EbN0dB, SER_zf, color = 'g', ls = '-', marker = markers[idx], ms = 12, label = f'{channeltype}, ZF eq.')
    axs.semilogy(EbN0dB, SER_mmse, color = 'r', ls = '-', marker = markers[idx], ms = 12, label = f'{channeltype}, MMSE eq.')
axs.semilogy(EbN0dB, SER_theory, color = 'k', ls = '-', label = f'{M}-{MOD_TYPE.upper()}' )

axs.set_ylim(1e-4, 1)
axs.set_xlabel( 'Eb/N0(dB)',)
axs.set_ylabel('SER',)
axs.set_title( "Probability of Symbol Error for BPSK signals")
axs.legend(fontsize = 20)

plt.show()
plt.close()


#%%










#%%





































