#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 01:16:33 2025

@author: jack
"""

import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import commpy

#%% Program 6.2: ser awgn.m: Theoretical SERs for various modulation schemes over AWGN channel
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

# Program 6.3: perf over awgn.m: Performance of various modulations over AWGN channel
nSym = 100000
EbN0dB = np.arange(-4, 26, 2 )
# MOD_TYPE = "qam"
# arrayOfM = [4, 16, 64, 256]

MOD_TYPE = "pam"    ## "pam"
arrayOfM = [2, 4, 8, 16, 32]

colors = ['b', 'g', 'r', 'c', 'm', 'k']
channelModel = "awgn"

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
for m, M in enumerate(arrayOfM):
    k = int(np.log2(M))
    EsN0dB = 10 * np.log10(k) + EbN0dB
    SER_sim = np.zeros(EbN0dB.size)
    modem, Es, bps = modulator(MOD_TYPE, M)
    map_table, demap_table = modem.getMappTable()
    uu = np.random.randint(0, 2, size = nSym * bps).astype(np.int8)
    s = modem.modulate(uu)
    d = np.array([demap_table[sym] for sym in s])
    for i, snrdb in enumerate(EsN0dB):
        if channelModel == 'awgn':
            h = np.ones((1, nSym))
        hs = h * s
        r, _ = add_awgn_noise(hs, snrdb)
        sCap = modem.demodulate(r, 'hard')
        dCap = []
        for j in range(nSym):
            dCap.append( int(''.join([str(num) for num in sCap[j*k:(j+1)*k]]), base = 2) )
        dCap = np.array(dCap)
        # dCap = np.array([int(''.join([str(num) for num in uu[j*k:(j+1)*k]])) for j in range(nSym)])
        SER_sim[i] = np.sum(d != dCap) / nSym
    SER_theory = ser_awgn(EbN0dB, MOD_TYPE, M)
    axs.semilogy(EbN0dB, SER_sim, color = colors[m], ls = 'none', marker = "o", ms = 12, )
    axs.semilogy(EbN0dB, SER_theory, color = colors[m], ls = '-', label = f'{M}-{MOD_TYPE.upper()}' )

axs.set_ylim(1e-6, 1)
axs.set_xlabel( 'Eb/N0(dB)',)
axs.set_ylabel('SER (Ps)',)
axs.set_title(f"Symbol Error Rate for M-{MOD_TYPE.upper()} over AWGN")
axs.legend(fontsize = 20)

plt.show()
plt.close()


#%% Program 6.5: ser rayleigh.m: Theoretical symbol error rates over Rayleigh fading channel
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

# Program 6.6: perf over rayleigh ﬂat fading.m: Performance of modulations over Rayleigh ﬂat fading
nSym = 100000
EbN0dB = np.arange(-10, 26, 2 )
# MOD_TYPE = "qam"
# arrayOfM = [4, 16, 64, 256]

MOD_TYPE = "pam"  # pam, psk
arrayOfM = [2, 4, 8, 16, 32]

colors = ['b', 'g', 'r', 'c', 'm', 'k']
channelModel = "rayleigh"

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
for m, M in enumerate(arrayOfM):
    k = int(np.log2(M))
    EsN0dB = 10 * np.log10(k) + EbN0dB
    SER_sim = np.zeros(EbN0dB.size)
    modem, Es, bps = modulator(MOD_TYPE, M)
    map_table, demap_table = modem.getMappTable()
    uu = np.random.randint(0, 2, size = nSym * bps).astype(np.int8)
    s = modem.modulate(uu)
    d = np.array([demap_table[sym] for sym in s])
    for i, snrdb in enumerate(EsN0dB):
        if channelModel == 'rayleigh':
            h = (np.random.randn(1, nSym) + 1j * np.random.randn(1, nSym))/np.sqrt(2)

        hs = h * s
        r, _ = add_awgn_noise(hs, snrdb)
        y = r*h.conjugate()/np.abs(h)**2
        sCap = modem.demodulate(y, 'hard')
        dCap = []
        for j in range(nSym):
            dCap.append( int(''.join([str(num) for num in sCap[j*k:(j+1)*k]]), base = 2) )
        dCap = np.array(dCap)
        # dCap = np.array([int(''.join([str(num) for num in uu[j*k:(j+1)*k]])) for j in range(nSym)])
        SER_sim[i] = np.sum(d != dCap) / nSym
    SER_theory = ser_rayleigh(EbN0dB, MOD_TYPE, M)
    axs.semilogy(EbN0dB, SER_sim,    color = colors[m], ls = 'none', marker = "o", ms = 12, )
    axs.semilogy(EbN0dB, SER_theory, color = colors[m], ls = '-', label = f'{M}-{MOD_TYPE.upper()}' )

axs.set_ylim(1e-3, 1.4)
axs.set_xlabel( 'Eb/N0(dB)',)
axs.set_ylabel('SER (Ps)',)
axs.set_title(f"Symbol Error Rate for M-{MOD_TYPE.upper()} over Rayleigh flat fading Channel")
axs.legend(fontsize = 20)

plt.show()
plt.close()


#%%  Program 6.8: ser rician.m: Theoretical symbol error rates over Ricean fading channel

from Modulations import modulator
from ChannelModels import add_awgn_noise

def ser_rician(EbN0dB, K_dB, MOD_TYPE, M):
    EbN0 = 10**(EbN0dB/10)
    EsN0 = np.log2(M) * EbN0
    K = 10**(K_dB/10)
    SER = np.zeros(EbN0dB.size)

    if MOD_TYPE.lower() == "psk":
        SER = np.zeros(EsN0.size)
        for i in range(len(EsN0)):
            g = np.sin(np.pi/M)**2
            fun = lambda x: (1+K)*(np.sin(x)**2)/((1+K)*(np.sin(x)**2) + g*EsN0[i]) * np.exp(-K*g*EsN0[i]/((1+K)*(np.sin(x)**2) + g*EsN0[i]))
            SER[i] = 1/np.pi * scipy.integrate.quad(fun, 0, np.pi*(M-1)/M)[0]
    elif MOD_TYPE.lower() == "qam":
        SER = np.zeros(EsN0.size)
        for i in range(len(EsN0)):
            g = 1.5 / (M-1)
            fun = lambda x: (1+K)*(np.sin(x)**2)/((1+K)*(np.sin(x)**2) + g*EsN0[i]) * np.exp(-K*g*EsN0[i]/((1+K)*(np.sin(x)**2) + g*EsN0[i]))
            SER[i] = 4/np.pi * (1-1/np.sqrt(M)) * scipy.integrate.quad(fun, 0, np.pi/2)[0] - 4/np.pi * (1-1/np.sqrt(M))**2 * scipy.integrate.quad(fun, 0, np.pi/4)[0]
    elif MOD_TYPE.lower() == "pam":
        SER = np.zeros(EsN0.size)
        for i in range(len(EsN0)):
            g = 3/(M**2 - 1)
            fun = lambda x: (1+K)*(np.sin(x)**2)/((1+K)*(np.sin(x)**2) + g*EsN0[i]) * np.exp(-K*g*EsN0[i]/((1+K)*(np.sin(x)**2) + g*EsN0[i]))
            SER[i] = 2*(M-1)/(M*np.pi) * scipy.integrate.quad(fun, 0, np.pi/2)[0]
    return SER

# Program 6.9: perf over rician flat fading.m: Performance of modulations over Ricean ﬂat fading channel
nSym = 100000
EbN0dB = np.arange(0, 24, 2 )
# MOD_TYPE = "qam"
M = 8  #  [4, 16, 64, 256]

MOD_TYPE = "pam"  # pam, psk

K_dB = [3, 5, 10, 20]
colors = ['b', 'g', 'r', 'c', 'm', 'k']
channelModel = "rician"

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
for idx, KdB in enumerate(K_dB):
    k = int(np.log2(M))
    EsN0dB = 10 * np.log10(k) + EbN0dB
    K = 10**(KdB/10)
    SER_sim = np.zeros(EbN0dB.size)
    modem, Es, bps = modulator(MOD_TYPE, M)
    map_table, demap_table = modem.getMappTable()
    uu = np.random.randint(0, 2, size = nSym * bps).astype(np.int8)
    s = modem.modulate(uu)
    d = np.array([demap_table[sym] for sym in s])
    for i, snrdb in enumerate(EsN0dB):
        if channelModel == 'rician':
            h = np.sqrt(K/(K+1))*(np.ones((1, nSym)) + 1j * np.ones((1, nSym)))/np.sqrt(2) + np.sqrt(1/(K+1))*(np.random.randn(1, nSym) + 1j * np.random.randn(1, nSym))/np.sqrt(2)

        hs = h * s
        r, _ = add_awgn_noise(hs, snrdb)
        y = r*h.conjugate()/np.abs(h)**2
        sCap = modem.demodulate(y, 'hard')
        dCap = []
        for j in range(nSym):
            dCap.append( int(''.join([str(num) for num in sCap[j*k:(j+1)*k]]), base = 2) )
        dCap = np.array(dCap)
        # dCap = np.array([int(''.join([str(num) for num in uu[j*k:(j+1)*k]])) for j in range(nSym)])
        SER_sim[i] = np.sum(d != dCap) / nSym
    SER_theory = ser_rician(EbN0dB, KdB, MOD_TYPE, M)
    axs.semilogy(EbN0dB, SER_sim,    color = colors[idx], ls = 'none', marker = "o", ms = 12, )
    axs.semilogy(EbN0dB, SER_theory, color = colors[idx], ls = '-', label = f"K = {KdB} dB" )

axs.set_ylim(1e-6, 1)
axs.set_xlabel( 'Eb/N0(dB)',)
axs.set_ylabel('SER (Ps)',)
axs.set_title(f"Symbol Error Rate for {M}-{MOD_TYPE.upper()} over Rician flat fading Channel")
axs.legend(fontsize = 20)

plt.show()
plt.close()










































































































































































































