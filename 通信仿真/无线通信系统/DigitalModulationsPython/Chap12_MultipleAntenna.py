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
from scipy.special import erfc
from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
from DigiCommPy.channels import awgn
from DigiCommPy.errorRates import ser_awgn


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
marker = ['o', '*', 'v']
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

axs.set_ylim(1e-5, 1.1)
axs.set_xlabel( 'Eb/N0(dB)',)
axs.set_ylabel('SER (Ps)',)
axs.set_title(f"Symbol Error Rate for M-{mod_type.upper()} over Rayleigh flat fading Channel")
axs.legend(fontsize = 20)

plt.show()
plt.close()



#%% Program 12.2: Alamouti.m: Receive diversity error rate performance simulation














































































































































