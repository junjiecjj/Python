#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 23:46:56 2025

@author: jack
"""
import scipy
import numpy as np
import matplotlib.pyplot as plt # for plotting functions



#%% Program 7.18: PRSignaling.m: Function to generate PR signals at symbol sampling instants

def PRSignaling(Q, L, span):
    # Generate the impulse response of a partial response System given,
    # Q - partial response polynomial for Q(D)
    # L - oversampling factor (Tsym/Ts)
    # Nsym - filter span in symbol durations
    # Returns the impulse response b(t) that spans the discrete-time base t=-Nsym:1/L:Nsym

    ## excite the Q(f) filter with an impulse to get the PR response
    qn = scipy.signal.lfilter(Q, 1.0, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Partial response filter Q(D) <-> q(t) and its upsampled version
    q = np.vstack((qn, np.zeros((L-1, qn.size)))) # Insert L-1 zero between each symbols
    q = q.T.flatten()                             # Convert to a single stream, output is at sampling rate
    # convolve q(t) with Nyqyist criterion satisfying sinc filter g(t)
    # Note: any filter that satisfy Nyquist first criterion can be used
    Tsym = 1                                # g(t) generated for 1 symbol duration
    t = np.arange(-span/2, span/2, 1/L)     # discrete-time base for 1 symbol duration
    g = np.sin(np.pi*t/Tsym)/(np.pi*t/Tsym) # sinc function
    g[np.argwhere(np.isnan(g))] = 1
    b = scipy.signal.convolve(g, q, 'same') # convolve q(t) and g(t)
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

    fig, axs = plt.subplots(1, 2, figsize = (12, 5), constrained_layout = True)
    axs[0].stem(t[0:t.size:L], b[0:t.size:L], linefmt = 'r-', markerfmt = 'D', )
    axs[0].plot(t, b, 'b-', )
    axs[0].set_xlabel('t/$T_{sym}$', fontsize = 12)
    axs[0].set_ylabel('b(t)', fontsize = 12)
    axs[0].set_title(f"{titles[i]}-b(t)" , fontsize = 16)
    # axs[0].set_xlim(-4 , 4)  #拉开坐标轴范围显示投影
    # axs[0].legend()

    axs[1].plot(norm_freq, norm_response, 'b-', )
    axs[1].set_xlabel('f/$F_{sym}$', fontsize = 12)
    axs[1].set_ylabel('|Q(D)|', fontsize = 12)
    axs[1].set_title(f"{titles[i]}-Frequency response Q(D)" , fontsize = 16)
    # axs[1].set_xlim(-1.5 , 1.5)  #拉开坐标轴范围显示投影
    # axs[1].legend()

    plt.suptitle("Impulse response and frequency response of various Partial response (PR) signaling schemes", fontsize = 16)
    plt.show()
    plt.close()


#%% Program 7.20: PR1 precoded system.m: Discrete-time equivalent partial response class 1 signaling model
M = 4
N = 100000
a = np.random.randint(0, M, N)
# Q = [1 , 2 , 0 , -2 , -1 ]
Q = [1, 1]
# Q = [1, 2, 1]
x = np.zeros(a.size) # output of precoder
D = np.zeros(len(Q), dtype = np.int32) # memory elements in the precoder

# Precoder (Inverse filter of Q(z)) with Modulo-M operation
for k in range(a.size):
    x[k] = (a[k] - np.sum(D[1:] * Q[1:]) ) % M
    D[1] = x[k]
    if D.size > 2:
        D[2:] = D[1:-1]

# Sampled received sequence-if desired,add noise here
bn = scipy.signal.lfilter(Q, 1, x)
# modulo-M reduction at receiver
acap = bn % M
error = np.sum(a != acap)


#%%  Performance of modulations in AWGN, 暂时没系统级的PR性能程序
# import sys
# sys.path.append("..")
# import scipy
# import numpy as np # for numerical computing
# import matplotlib.pyplot as plt # for plotting functions
# from matplotlib import cm # colormap for color palette
# from scipy.special import erfc
# from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
# from DigiCommPy.channels import awgn
# from DigiCommPy.errorRates import ser_awgn

#
# #---------Input Fields------------------------
# nSym = 10**6 # Number of symbols to transmit
# EbN0dBs = np.arange(start = -4, stop = 26, step = 2) # Eb/N0 range in dB for simulation
# mod_type = 'PSK' # Set 'PSK' or 'QAM' or 'PAM' or 'FSK'
# arrayOfM = [2, 4, 8, 16, 32] # array of M values to simulate
# coherence = 'coherent' #'coherent'/'noncoherent'-only for FSK

# # mod_type = 'QAM'
# # nSym =  10**7
# # arrayOfM = [ 64,  ] # uncomment this line if MOD_TYPE='QAM'

# span = 8
# L = 4
# Q = [1.0, 0.0, -2.0, 0.0, 1.0]
# p, t = PRSignaling(Q, L, span)

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
#         modem = modem_dict[mod_type.lower()](M, coherence)#choose modem from dictionary
#     else: #for all other modulations
#         modem = modem_dict[mod_type.lower()](M)#choose modem from dictionary

#     for j, EsN0dB in enumerate(EsN0dBs):

#         d = np.random.randint(low = 0, high = M, size = nSym) # uniform random symbols from 0 to M-1
#         u = modem.modulate(d) #modulate

#         ##  脉冲成型 + 上变频-> 基带信号
#         s = scipy.signal.upfirdn(p, u, L)
#         ## channel
#         r = awgn(s, EsN0dB, L)

#         ##  下采样 + 匹配滤波 -> 恢复的基带信号
#         z = scipy.signal.upfirdn(p, r, 1, L)  ## 此时默认上采样为1，即不进行上采样

#         ## 选取最佳采样点,
#         decision_site = int((z.size - u.size) / 2)

#         ## 每个符号选取一个点作为判决
#         u_hat = z[decision_site:decision_site + u.size] / L
#         # dCap = modem.demodulate(u_hat, outputtype = 'int',)

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




















