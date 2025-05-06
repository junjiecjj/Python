#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 13:06:03 2025

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

#%% Program 12.1: receive diversity.m: Receive diversity error rate performance simulation
import numpy as np # for numerical computing
import matplotlib.pyplot as plt # for plotting functions
from matplotlib import cm # colormap for color palette
# from scipy.special import erfc
from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
from DigiCommPy.channels import awgn
from DigiCommPy.errorRates import ser_awgn
from DigiCommPy.errorRates import ser_rayleigh

nSym = 200000
N = [1, 2, 20]
EbN0dBs = np.arange(-20, 38, 2 )

coherence = 'coherent' #'coherent'/'noncoherent'-only for FSK

mod_type = 'PSK' # Set 'PSK' or 'QAM' or 'PAM
# arrayOfM = [2, 4, 8, 16, 32] # array of M values to simulate
M = 2

# mod_type = 'QAM'
# arrayOfM=[4, 16, 64, 256] # uncomment this line if MOD_TYPE='QAM'
# M = 16
k = np.log2(M)
EsN0dBs = 10 * np.log10(k) + EbN0dBs
modem_dict = {'psk': PSKModem,'qam':QAMModem,'pam':PAMModem}

if mod_type.lower()=='fsk':
    modem = modem_dict[mod_type.lower()](M, coherence)#choose modem from dictionary
else: #for all other modulations
    modem = modem_dict[mod_type.lower()](M)#choose modem from dictionary

colors = ['b', 'g', 'r', 'c', 'm', 'k']
marker = ['o', '*', 'v', 'd']
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
for m, nrx in enumerate(N):
    ser_MRC = np.zeros(EsN0dBs.size)
    ser_EGC = np.zeros(EsN0dBs.size)
    ser_SC = np.zeros(EsN0dBs.size)

    ## Transmitter
    # uu = np.random.randint(0, 2, size = nSym * k).astype(np.int8)
    # s = modem.modulate(uu)
    d = np.random.randint(low = 0, high = M, size = nSym)
    s = modem.modulate(d) #modulate
    s_diversity = np.kron(np.ones((nrx, 1)), s)
    for i, EsN0dB in enumerate(EsN0dBs):
        h = (np.random.randn(nrx, nSym) + 1j * np.random.randn(nrx, nSym))/np.sqrt(2)
        signal = h * s_diversity

        gamma = 10**(EsN0dB/10)
        P = np.sum(np.abs(signal)**2, axis = 1)/nSym
        N0 = P/gamma
        noise = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))*(np.sqrt(N0/2))[:,None]
        r = signal + noise

        ## MRC processing assuming perfect channel estimates
        s_MRC = np.sum(h.conjugate() * r, axis = 0)/np.sum(np.abs(h)**2, axis = 0)
        if mod_type.lower()=='fsk': #demodulate (Refer Chapter 3)
            d_MRC = modem.demodulate(s_MRC, coherence)
        else: #demodulate (Refer Chapter 3)
            d_MRC = modem.demodulate(s_MRC)

        ## EGC processing assuming perfect channel estimates
        h_phases = np.exp(-1j*np.angle(h))
        s_EGC = np.sum(h_phases * r, axis = 0)/np.sum(np.abs(h), axis = 0)
        if mod_type.lower()=='fsk': #demodulate (Refer Chapter 3)
            d_EGC = modem.demodulate(s_EGC, coherence)
        else: #demodulate (Refer Chapter 3)
            d_EGC = modem.demodulate(s_EGC)

        ## SC processing assuming perfect channel estimates
        idx = np.abs(h).argmax(axis = 0)
        h_best = h[idx, np.arange(h.shape[-1])]
        y = r[idx, np.arange(r.shape[-1])]

        s_SC = y * h_best.conjugate() / np.abs(h_best)**2
        if mod_type.lower()=='fsk': #demodulate (Refer Chapter 3)
            d_SC = modem.demodulate(s_SC, coherence)
        else: #demodulate (Refer Chapter 3)
            d_SC = modem.demodulate(s_SC)

        ser_MRC[i] = np.sum(d != d_MRC)/nSym
        ser_EGC[i] = np.sum(d != d_EGC)/nSym
        ser_SC[i] = np.sum(d != d_SC)/nSym

    axs.semilogy(EbN0dBs, ser_MRC, color = 'b', ls = '-', marker = marker[m], ms = 5, label = f"MRC, Nr = {nrx}")
    axs.semilogy(EbN0dBs, ser_EGC, color = 'r', ls = '--', marker = marker[m], ms = 5, label = f"EGC, Nr = {nrx}")
    axs.semilogy(EbN0dBs, ser_SC, color = 'g', ls = '-.', marker = marker[m], ms = 5, label = f"SC, Nr = {nrx}")

SER_theory = ser_rayleigh(EbN0dBs, mod_type, M) #theory SER
axs.semilogy(EbN0dBs, SER_theory, color = 'k', ls = '-.', marker = marker[m+1], ms = 5, label = "Theo, Nr = 1")
axs.set_ylim(1e-5, 1.1)
axs.set_xlabel( 'Eb/N0(dB)',)
axs.set_ylabel('SER (Ps)',)
axs.set_title(f"Symbol Error Rate for M-{mod_type.upper()} over Rayleigh flat fading Channel")
axs.legend(fontsize = 20)

plt.show()
plt.close()

#%% Program 12.2: Alamouti.m: Receive diversity error rate performance simulation
nSym = 200000
# N = [1, 2, 20]
EbN0dBs = np.arange(-20, 38, 2)
coherence = 'coherent'      # coherent/noncoherent-only for FSK

mod_type = 'PSK'            # Set 'PSK' or 'QAM' or 'PAM
# arrayOfM = [2, 4, 8, 16, 32] # array of M values to simulate
M = 2
# mod_type = 'QAM'
# arrayOfM=[4, 16, 64, 256] # uncomment this line if MOD_TYPE='QAM'
# M = 16
k = np.log2(M)
EsN0dBs = 10 * np.log10(k) + EbN0dBs
modem_dict = {'psk': PSKModem,'qam':QAMModem,'pam':PAMModem}

if mod_type.lower()=='fsk':
    modem = modem_dict[mod_type.lower()](M, coherence)   # choose modem from dictionary
else: # for all other modulations
    modem = modem_dict[mod_type.lower()](M)              # choose modem from dictionary

colors = ['b', 'g', 'r', 'c', 'm', 'k']
marker = ['o', '*', 'v']
ser_Alamouti = np.zeros(EsN0dBs.size)
ser_MRC = np.zeros(EsN0dBs.size)
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
for i, EsN0dB in enumerate(EsN0dBs):
    d = np.random.randint(low = 0, high = M, size = nSym)
    s = modem.modulate(d) #modulate
    ### Alamouti coded
    ss = np.kron(s.reshape(-1,2).T, np.ones((1,2)))
    h = (np.random.randn(2, int(nSym/2)) + 1j * np.random.randn(2, int(nSym/2)))/np.sqrt(2)
    H = np.kron(h, np.ones((1,2)))
    H[:,1::2] = np.flipud(H[:,1::2]).conjugate()
    H[1,1::2] = -1 * H[1,1::2]       # Alamouti coded channel coeffs

    signal = np.sum(H*ss, axis = 0)
    gamma = 10**(EsN0dB/10.0)
    P = np.mean(np.abs(signal)**2)
    N0 = P/gamma
    # The performance of Alamouti scheme is 3-dB worse than that of MRC scheme.
    # This is because the simulations assume that each transmit antenna radiate symbols with half the energy so that the total combined energy radiated is same as that of a single transmit antenna.
    # In the Alamouti scheme, if each of the transmit antennas radiate same energy as that of the 1 transmit-2 receive antenna configuration used for MRC scheme, their performance curves would match.
    # N0 = P/gamma/2
    noise = np.sqrt(N0/2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))

    r = signal + noise
    rVec = np.kron(r.reshape(-1, 2).T, np.ones((1, 2)))
    Hest = H   # perfect channel estimation
    Hest[0, 0::2] = Hest[0, 0::2].conjugate()
    Hest[1, 1::2] = Hest[1, 1::2].conjugate()

    y = np.sum(Hest*rVec, axis = 0)
    sCap = y/np.sum(Hest.conjugate() * Hest, axis = 0)

    if mod_type.lower()=='fsk': #demodulate (Refer Chapter 3)
        dCap = modem.demodulate(sCap, coherence)
    else: #demodulate (Refer Chapter 3)
        dCap = modem.demodulate(sCap)
    ser_Alamouti[i] = np.sum(d != dCap)/nSym

    ### MRC
    nrx = 2
    s_diversity = np.kron(np.ones((nrx, 1)), s)
    h = (np.random.randn(nrx, nSym) + 1j * np.random.randn(nrx, nSym))/np.sqrt(2)
    signal = h * s_diversity

    gamma = 10**(EsN0dB/10)
    P = np.sum(np.abs(signal)**2, axis = 1)/nSym
    N0 = P/gamma
    noise = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))*(np.sqrt(N0/2))[:,None]
    r = signal + noise

    ## MRC processing assuming perfect channel estimates
    s_MRC = np.sum(h.conjugate() * r, axis = 0)/np.sum(np.abs(h)**2, axis = 0)
    if mod_type.lower()=='fsk': #demodulate (Refer Chapter 3)
        d_MRC = modem.demodulate(s_MRC, coherence)
    else: #demodulate (Refer Chapter 3)
        d_MRC = modem.demodulate(s_MRC)
    ser_MRC[i] = np.sum(d != d_MRC)/nSym

axs.semilogy(EbN0dBs, ser_Alamouti, color = 'r', ls = '--', marker = marker[0], ms = 5, label = "Alamouti, Nr = 1, Nt = 2")
axs.semilogy(EbN0dBs, ser_MRC, color = 'b', ls = '--', marker = marker[1], ms = 5, label = "MRC, Nr = 2, Nt = 1")
axs.set_ylim(1e-4, 1.1)
axs.set_xlabel( 'Eb/N0(dB)',)
axs.set_ylabel('SER (Ps)',)
axs.set_title("2x1 Transmit diversity - Alamouti coding")
axs.legend(fontsize = 20)

plt.show()
plt.close()































































































































