#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:25:57 2025

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
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22

#%%
# 生成 循环矩阵
def CirculantMatric(gen, row):
     if type(gen) == list:
          col = len(gen)
     elif type(gen) == np.ndarray:
          col = gen.size
     row = col

     mat = np.zeros((row, col), dtype = gen.dtype)
     mat[:, 0] = gen
     for i in range(1, row):
          mat[:,i] = np.roll(gen, i)
     return mat

def circularConvolve(h, s, N):
    if h.size < N:
        h = np.hstack((h, np.zeros(N-h.size)))
    col = N
    row = s.size
    H = np.zeros((row, col), dtype = s.dtype)
    H[:, 0] = h
    for i in range(1, row):
          H[:,i] = np.roll(h, i)
    res = H @ s
    return res

# # generateVec =  [1+1j, 2+2j, 3+3j, 4+1j ]
# generateVec =  [1 , 2  , 3 , 4  ]
# X = np.array(generateVec)
# L = len(generateVec)
# A = CirculantMatric(X, L)
N = 8
h = np.array([-0.4878, -1.5351, 0.2355])
s = np.array([-0.0155, 2.5770, 1.9238, -0.0629, -0.8105, 0.6727, -1.5924, -0.8007])

lin_s_h = scipy.signal.convolve(h, s)
cir_s_h = circularConvolve(h, s, N)

Ncp = 2
s_cp = np.hstack((s[-Ncp:], s))
lin_scp = scipy.signal.convolve(h, s_cp)
r = lin_scp[Ncp:Ncp+N]
print(f"lin_scp = \n{lin_scp[Ncp:Ncp+N]}\ncir_s_h = \n{cir_s_h}")


R = scipy.fft.fft(r, N)
H = scipy.fft.fft(h, N)
S = scipy.fft.fft(s, N)

r1 = scipy.fft.ifft(S*H)

print(f"r1 = \n{r1}\ncir_s_h = \n{cir_s_h}")


#%% Program 14.2: add cyclic prefix.m: Function to add cyclic prefix of length Ncp symbols
def add_cyclic_prefix(x, Ncp):
    s = np.hstack((x[-Ncp:], x))
    return s

#%% Program 14.3: remove cyclic prefix.m: Function to remove the cyclic prefix from the OFDM symbol
def remove_cyclic_prefix(r, Ncp, N):
    y = r[Ncp : Ncp+N]
    return y

#%% Program 14.4: ofdm on awgn.m: OFDM transmission and reception on AWGN channel

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


nSym = 10000
EbN0dB = np.arange(0, 22, 2)
MOD_TYPE = "psk"

M = 4
N = 64
Ncp = 16
colors = ['b', 'g', 'r', 'c', 'm', 'k']
k = int(np.log2(M))
EsN0dB = 10*np.log10(k) + EbN0dB
errors= np.zeros(EsN0dB.size)

modem, Es, bps = modulator(MOD_TYPE, M)
map_table, demap_table = modem.getMappTable()
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

for i, esnodB in enumerate(EsN0dB):
    for j, sym in enumerate(range(nSym)):
        print(f"{i}/{EsN0dB.size}, {j}/{nSym}")
        ## Transmitter
        uu = np.random.randint(0, 2, size = N * bps).astype(np.int8)
        X = modem.modulate(uu)
        d = np.array([demap_table[sym] for sym in X])
        x = scipy.fft.ifft(X, N)
        s = add_cyclic_prefix(x, Ncp)

        ## Channel
        r, _ = add_awgn_noise(s, esnodB)

        ## Receiver
        y = remove_cyclic_prefix(r, Ncp, N)
        Y = scipy.fft.fft(y, N)
        uCap = modem.demodulate(Y, 'hard')
        dCap = []
        for l in range(N):
            dCap.append( int(''.join([str(num) for num in uCap[l*bps:(l+1)*bps]]), base = 2) )
        dCap = np.array(dCap)

        ## Error Counter
        numErrors = np.sum(d != dCap)
        errors[i] += numErrors
SER_sim = errors/(nSym * N)
SER_theory = ser_awgn(EbN0dB, MOD_TYPE, M)

axs.semilogy(EbN0dB, SER_sim, color = colors[0], ls = 'none', marker = "o", ms = 12, )
axs.semilogy(EbN0dB, SER_theory, color = colors[0], ls = '-', label = f'{M}-{MOD_TYPE.upper()}' )

axs.set_ylim(1e-6, 1)
axs.set_xlabel( 'Eb/N0(dB)',)
axs.set_ylabel('SER (Ps)',)
axs.set_title(f"M{MOD_TYPE.upper()}-CP-OFDM over AWGN")
axs.legend(fontsize = 20)

plt.show()
plt.close()

#%%
from tqdm import tqdm

nSym = 10000
EbN0dB = np.arange(-2, 26, 2)
# MOD_TYPE = "psk"    ## "pam"
# arrayOfM = [2, 4, 8, 16, 32]

MOD_TYPE = "qam"
arrayOfM = [4, 16, 64, 256]

N = 64
Ncp = 16
colors = ['b', 'g', 'r', 'c', 'm', 'k']

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

for m, M in  enumerate(arrayOfM) :
    print(f"{m}/{len(arrayOfM)}")
    k = int(np.log2(M))
    EsN0dB = 10*np.log10(k) + EbN0dB
    errors = np.zeros(EsN0dB.size)
    modem, Es, bps = modulator(MOD_TYPE, M)
    map_table, demap_table = modem.getMappTable()
    for i, esnodB in tqdm(enumerate(EsN0dB)):
        for j, sym in enumerate(range(nSym)):
            ## Transmitter
            uu = np.random.randint(0, 2, size = N * bps).astype(np.int8)
            X = modem.modulate(uu)
            d = np.array([demap_table[sym] for sym in X])
            x = scipy.fft.ifft(X, N)
            s = add_cyclic_prefix(x, Ncp)

            ## Channel
            r, _ = add_awgn_noise(s, esnodB)

            ## Receiver
            y = remove_cyclic_prefix(r, Ncp, N)
            Y = scipy.fft.fft(y, N)
            uCap = modem.demodulate(Y, 'hard')
            dCap = []
            for l in range(N):
                dCap.append( int(''.join([str(num) for num in uCap[l*bps:(l+1)*bps]]), base = 2) )
            dCap = np.array(dCap)

            ## Error Counter
            numErrors = np.sum(d != dCap)
            errors[i] += numErrors
    SER_sim = errors/(nSym * N)
    SER_theory = ser_awgn(EbN0dB, MOD_TYPE, M)

    axs.semilogy(EbN0dB, SER_sim, color = colors[m], ls = 'none', marker = "o", ms = 12, )
    axs.semilogy(EbN0dB, SER_theory, color = colors[m], ls = '-', label = f'{M}{MOD_TYPE.upper()} OFDM' )

axs.set_ylim(1e-6, 1)
axs.set_xlabel( 'Eb/N0(dB)',)
axs.set_ylabel('SER (Ps)',)
axs.set_title(f"M{MOD_TYPE.upper()}-CP-OFDM over AWGN")
axs.legend(fontsize = 20)

plt.show()
plt.close()


#%% Program 14.5: ofdm on freq sel chan.m: OFDM on frequency selective Rayleigh fading channel

from Modulations import modulator
from ChannelModels import add_awgn_noise
def ser_rayleigh(EbN0dB, MOD_TYPE, M):
    EbN0 = 10**(EbN0dB/10)
    EsN0 = np.log2(M) * EbN0
    SER = np.zeros(EbN0dB.size)
    if MOD_TYPE.lower() == "bpsk":
        SER = 1/2 * (1 - np.sqrt(EsN0/(1 + EsN0)))
    elif MOD_TYPE.lower() == "psk":
        SER = np.zeros(EsN0.size)
        for i in range(len(EsN0)):
            g = np.sin(np.pi/M)**2
            fun = lambda x: 1.0 / (1.0 + g * EsN0[i]/(np.sin(x)**2))
            SER[i] = 1/np.pi * scipy.integrate.quad(fun, 0, np.pi*(M-1)/M)[0]
    elif MOD_TYPE.lower() == "qam":
        SER = np.zeros(EsN0.size)
        for i in range(len(EsN0)):
            g = 1.5 / (M-1)
            fun = lambda x: 1.0 / (1.0 + g * EsN0[i]/np.sin(x)**2)
            SER[i] = 4/np.pi * (1-1/np.sqrt(M)) * scipy.integrate.quad(fun, 0, np.pi/2)[0] - 4/np.pi * (1-1/np.sqrt(M))**2 * scipy.integrate.quad(fun, 0, np.pi/4)[0]
    elif MOD_TYPE.lower() == "pam":
        SER = np.zeros(EsN0.size)
        for i in range(len(EsN0)):
            g = 3/(M**2 - 1)
            fun = lambda x: 1.0 / (1.0 + g * EsN0[i]/np.sin(x)**2)
            SER[i] = 2*(M-1)/(M*np.pi) * scipy.integrate.quad(fun, 0, np.pi/2)[0]
    return SER

nSym = 10000
EbN0dB = np.arange(-2, 24, 2)
# MOD_TYPE = "psk"
# arrayOfM = [2, 4, 8, 16, 32, 64]

MOD_TYPE = "qam"
arrayOfM = [4, 16, 64, 256]

N = 64
Ncp = 16
L = 10 # Number of taps for the frequency selective channel model
colors = ['yellow', 'g', 'pink', 'b', 'c', 'r', 'k']

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

for m, M in  enumerate(arrayOfM):
    print(f"{m}/{len(arrayOfM)}")
    k = int(np.log2(M))
    EsN0dB = 10*np.log10(k*N/(N+Ncp)) + EbN0dB
    errors= np.zeros(EsN0dB.size)

    modem, Es, bps = modulator(MOD_TYPE, M)
    map_table, demap_table = modem.getMappTable()

    for i, esnodB in tqdm(enumerate(EsN0dB)):
        for j, sym in enumerate(range(nSym)):
            # print(f"{i}/{EsN0dB.size}, {j}/{nSym}")
            ## Transmitter
            uu = np.random.randint(0, 2, size = N * bps).astype(np.int8)
            X = modem.modulate(uu)
            d = np.array([demap_table[sym] for sym in X])
            x = scipy.fft.ifft(X, N)
            s = add_cyclic_prefix(x, Ncp)

            ## Channel
            h = (np.random.randn(L) + 1j * np.random.randn(L))/np.sqrt(2)
            H = scipy.fft.fft(h, N)
            hs = scipy.signal.convolve(h, s)
            r, _ = add_awgn_noise(hs, esnodB)

            ## Receiver
            y = remove_cyclic_prefix(r, Ncp, N)
            Y = scipy.fft.fft(y, N)
            V = Y/H
            uCap = modem.demodulate(V, 'hard')
            dCap = []
            for l in range(N):
                dCap.append( int(''.join([str(num) for num in uCap[l*bps:(l+1)*bps]]), base = 2) )
            dCap = np.array(dCap)

            ## Error Counter
            numErrors = np.sum(d != dCap)
            errors[i] += numErrors
    SER_sim = errors/(nSym * N)
    SER_theory = ser_rayleigh(EbN0dB, MOD_TYPE, M)

    axs.semilogy(EbN0dB, SER_sim, color = colors[m], ls = 'none', marker = "o", ms = 12, )
    axs.semilogy(EbN0dB, SER_theory, color = colors[m], ls = '-', label = f'{M}-{MOD_TYPE.upper()}' )

axs.set_ylim(1e-3, 1)
axs.set_xlabel( 'Eb/N0(dB)',)
axs.set_ylabel('SER (Ps)',)
axs.set_title(f"M{MOD_TYPE.upper()}-CP-OFDM over Freq Selective Rayleigh")
axs.legend(fontsize = 20)

plt.show()
plt.close()


#%%











































































































































