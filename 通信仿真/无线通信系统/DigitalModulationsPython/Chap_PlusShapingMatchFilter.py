#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 01:37:26 2025

@author: jack
"""

import sys
sys.path.append("..")
import scipy
import numpy as np # for numerical computing
import matplotlib.pyplot as plt # for plotting functions
from matplotlib import cm # colormap for color palette
from scipy.special import erfc
from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
from DigiCommPy.channels import awgn
from DigiCommPy.errorRates import ser_awgn


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

#%%  Performance of modulations in AWGN
#---------Input Fields------------------------
nSym = 10**6 # Number of symbols to transmit
EbN0dBs = np.arange(start = -4, stop = 26, step = 2) # Eb/N0 range in dB for simulation
mod_type = 'PSK' # Set 'PSK' or 'QAM' or 'PAM' or 'FSK'
arrayOfM = [2, 4, 8, 16, 32] # array of M values to simulate
coherence = 'coherent' #'coherent'/'noncoherent'-only for FSK

mod_type = 'QAM'
nSym = 10**8
arrayOfM = [4, 16, 64, 256] # uncomment this line if MOD_TYPE='QAM'

beta = 0.3
span = 8
L = 4
p, t, filtDelay = srrcFunction(beta, L, span)

modem_dict = {'psk': PSKModem,'qam':QAMModem,'pam':PAMModem,'fsk':FSKModem}
colors = plt.cm.jet(np.linspace(0, 1, len(arrayOfM))) # colormap
fig, ax = plt.subplots(nrows = 1, ncols = 1)

for i, M in enumerate(arrayOfM):
    print(f" {M} in {arrayOfM}")
    #-----Initialization of various parameters----
    k = np.log2(M)
    EsN0dBs = 10*np.log10(k)+EbN0dBs # EsN0dB calculation
    SER_sim = np.zeros(len(EbN0dBs)) # simulated Symbol error rates

    if mod_type.lower()=='fsk':
        modem=modem_dict[mod_type.lower()](M, coherence)#choose modem from dictionary
    else: #for all other modulations
        modem = modem_dict[mod_type.lower()](M)#choose modem from dictionary

    for j, EsN0dB in enumerate(EsN0dBs):

        d = np.random.randint(low=0, high = M, size=nSym) # uniform random symbols from 0 to M-1
        u = modem.modulate(d) #modulate

        ##  脉冲成型 + 上变频-> 基带信号
        s = scipy.signal.upfirdn(p, u, L)
        ## channel
        r = awgn(s, EsN0dB, L)

        ##  下采样 + 匹配滤波 -> 恢复的基带信号
        z = scipy.signal.upfirdn(p, r, 1, L)  ## 此时默认上采样为1，即不进行上采样

        ## 选取最佳采样点,
        decision_site = int((z.size - u.size) / 2)

        ## 每个符号选取一个点作为判决
        u_hat = z[decision_site:decision_site + u.size] / L
        # dCap = modem.demodulate(u_hat, outputtype = 'int',)

        if mod_type.lower()=='fsk': #demodulate (Refer Chapter 3)
            dCap = modem.demodulate(u_hat, coherence)
        else: #demodulate (Refer Chapter 3)
            dCap = modem.demodulate(u_hat)

        SER_sim[j] = np.sum(dCap != d)/nSym

    SER_theory = ser_awgn(EbN0dBs, mod_type, M, coherence) #theory SER
    ax.semilogy(EbN0dBs, SER_sim, color = colors[i], marker='o', linestyle='', label='Sim '+str(M)+'-'+mod_type.upper())
    ax.semilogy(EbN0dBs, SER_theory, color = colors[i], linestyle='-', label='Theory '+str(M)+'-'+mod_type.upper())

ax.set_ylim(1e-6, 1)
ax.set_xlabel('Eb/N0(dB)')
ax.set_ylabel('SER ($P_s$)')
ax.set_title('Probability of Symbol Error for M-'+str(mod_type)+' over AWGN')
ax.legend(fontsize = 12)
plt.show()
plt.close()



# #%%  Performance of modulations in AWGN
# #---------Input Fields------------------------
# nSym = 10**6 # Number of symbols to transmit
# EbN0dBs = np.arange(start = -4, stop = 26, step = 2) # Eb/N0 range in dB for simulation
# mod_type = 'PSK' # Set 'PSK' or 'QAM' or 'PAM' or 'FSK'
# arrayOfM = [2, 4, 8, 16, 32] # array of M values to simulate
# coherence = 'coherent' #'coherent'/'noncoherent'-only for FSK

# # mod_type = 'QAM'
# # nSym = 10**8
# # arrayOfM = [4, 16, 64, 256] # uncomment this line if MOD_TYPE='QAM'

# beta = 0.3
# span = 8
# L = 4
# p, t, filtDelay = srrcFunction(beta, L, span)

# modem_dict = {'psk': PSKModem,'qam':QAMModem,'pam':PAMModem,'fsk':FSKModem}
# colors = plt.cm.jet(np.linspace(0, 1, len(arrayOfM))) # colormap
# fig, ax = plt.subplots(nrows = 1, ncols = 1)

# for i, M in enumerate(arrayOfM):
#     print(f" {M} in {arrayOfM}")
#     #-----Initialization of various parameters----
#     k = np.log2(M)
#     EsN0dBs = 10*np.log10(k)+EbN0dBs # EsN0dB calculation
#     SER_sim = np.zeros(len(EbN0dBs)) # simulated Symbol error rates

#     if mod_type.lower()=='fsk':
#         modem=modem_dict[mod_type.lower()](M, coherence)#choose modem from dictionary
#     else: #for all other modulations
#         modem = modem_dict[mod_type.lower()](M)#choose modem from dictionary

#     for j, EsN0dB in enumerate(EsN0dBs):

#         d = np.random.randint(low=0, high = M, size=nSym) # uniform random symbols from 0 to M-1
#         u = modem.modulate(d) #modulate
#         ## Upper sample
#         v = np.vstack((u, np.zeros((L-1, u.size))))
#         v = v.T.flatten()
#         ## plus shaping
#         s = scipy.signal.convolve(v, p, 'full')

#         ## channel
#         r = awgn(s, EsN0dB, L)

#         ## receiver
#         ## match filter
#         vCap = scipy.signal.convolve(r, p, 'full')
#         ## Down sampling
#         u_hat = vCap[int(2 * filtDelay) : int(vCap.size - 2*filtDelay) : L ] / L

#         if mod_type.lower()=='fsk': #demodulate (Refer Chapter 3)
#             dCap = modem.demodulate(u_hat, coherence)
#         else: #demodulate (Refer Chapter 3)
#             dCap = modem.demodulate(u_hat)

#         SER_sim[j] = np.sum(dCap != d)/nSym

#     SER_theory = ser_awgn(EbN0dBs, mod_type, M, coherence) #theory SER
#     ax.semilogy(EbN0dBs, SER_sim, color = colors[i], marker='o', linestyle='', label='Sim '+str(M)+'-'+mod_type.upper())
#     ax.semilogy(EbN0dBs, SER_theory, color = colors[i], linestyle='-', label='Theory '+str(M)+'-'+mod_type.upper())

# ax.set_ylim(1e-6, 1)
# ax.set_xlabel('Eb/N0(dB)')
# ax.set_ylabel('SER ($P_s$)')
# ax.set_title('Probability of Symbol Error for M-'+str(mod_type)+' over AWGN')
# ax.legend(fontsize = 12)
# plt.show()
# plt.close()





















