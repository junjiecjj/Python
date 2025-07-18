#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 00:22:31 2025

@author: jack

<Wireless communication systems in Matlab> Chap3
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
plt.rcParams['lines.markersize'] = 6    # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22

#%% Program 3.1: Shannon limit.m: Dependency of spectral and power efficiencies for AWGN channel
k = np.arange(0.1, 15, 0.001)
EbN0 = (2**k-1)/k
EbN0dB = 10 * np.log10(EbN0)

##### plot
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

axs.semilogy(EbN0dB, k, color = 'b', label = 'capacity boundary')
axs.set_xlabel(r'$\mathrm{E_b}/\mathrm{N_0}$(dB)',)
axs.set_ylabel('Spectral Efficiency (Bit/s/Hz)',)
axs.set_title("Channel Capacity & Power efficiency limit")

axs.vlines(x = -1.59, ymin = 0.08 , ymax = 14, color = 'red', ls = '--')
axs.text( 4, 8, "R > C\nUnattainable", color='red', size = 25, rotation = 0., ha = "center", va="center",)
axs.text(16, 0.3, "R < C\nPractical systems", color='g', size = 25, rotation = 0., ha = "center", va="center",)
axs.text( 16, 5, "R = C\nCapacity boundary", color='k', size = 25, rotation = 15., ha = "center", va="center",)
axs.annotate("Shannon limit\n-1.59dB", xy = (0.17, 0.2) , xytext = (0.06,0.01) , size = 20, xycoords = 'figure fraction', textcoords='figure fraction', arrowprops=dict(color='r',arrowstyle='<|-',connectionstyle='arc3'))
# axs.legend()
plt.show()
plt.close()

#%% Program 3.7: ergodic capacity limits.m: Simulating the ergodic capacity of a fading channel
snrdB = np.arange(-10, 30, 1/2)
h = (np.random.randn(1, 10000) + 1j * np.random.randn(1, 10000))/np.sqrt(2)
sigma_z = 1
snr = 10**(snrdB/10)
P = (sigma_z**2) * snr / np.mean(np.abs(h)**2)

C_awgn = np.log2(1 + np.mean(np.abs(h)**2) * P / (sigma_z**2))
C_fading = np.mean(np.log2(1 + (np.abs(h)**2).T @ P.reshape(1, -1) / sigma_z**2 ), axis = 0)

##### plot
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

axs.plot(snrdB, C_awgn, color = 'b', ls = '--',  )
axs.plot(snrdB, C_fading, color = 'r', ls = '-',  )
axs.text(18, 6.5, "AWGN channel capacity", color='k', size = 25, rotation = 44., ha = "center", va="center",)
axs.text(20, 5.4, "Fading channel Ergodic capacity", color='k', size = 25, rotation = 42., ha = "center", va="center",)

axs.set_xlabel( 'SNR(dB)',)
axs.set_ylabel('Capacity (Bit/s/Hz)',)
axs.set_title("SISO fading channel - Ergodic capacity")

plt.show()
plt.close()

#%% Program 3.3: CapacityBSC.m: Simulate transmission through a BSC(Binary Symmetric Channel) channel and calculate capacity.
# 只对均匀0/1无偏信源有效，有偏信源不同.
nbits = int(1e5)
errProbs = np.arange(0.01 , 1., 0.01)
C_simulation = np.zeros(len(errProbs))

for i, e in enumerate(errProbs):
    x = np.random.randint(0, 2, size = nbits, dtype = np.int8)        # 信源
    error = np.random.binomial(1, e, size = nbits).astype(np.int8)   # 信道
    y = x ^ error                                                    # 接受序列

    Y = np.tile(A = y, reps = (2,1))
    X = np.tile(np.array([[0], [1]]), reps = (1, nbits))
    # probabilities with respect to each input[0,1]
    prob = (Y == X) * (1 - e) + (Y != X) * e     # BSC p(y/x) equation
    prob = np.maximum(prob, 1e-20)               # this used to avoid NAN in computation
    p = prob / np.tile(A = np.sum(prob, axis = 0), reps = (2, 1)) # normalize probabilities
    ## HYX = -\sum_x P(x) [\sum_y P(Y|X) log2(P(Y|X))],
    HYX = np.mean(-np.sum(p * np.log2(p), axis = 0)) # senders uncertainity

    py0 = np.sum(y == 0) / nbits
    py1 = np.sum(y == 1) / nbits
    HY = -py0 * np.log2(py0) - py1 * np.log2(py1)
    C_simulation[i] = HY - HYX

C_theory = 1 - (-errProbs * np.log2(errProbs) - (1 - errProbs) * np.log2(1 - errProbs))

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
axs.plot(errProbs, C_simulation, color = 'b', ls = '--',marker = 'o', ms = 8, mfc = 'none', markevery = 10, label = 'Simulation' )
axs.plot(errProbs, C_theory, color = 'r', ls = '-', label = 'Theory' )

axs.set_xlabel( 'Cross-over probability - e',)
axs.set_ylabel('Capacity (bits/channel use)',)
axs.set_title("Capacity over Binary Symmetric Channel")
axs.legend(fontsize = 20)

plt.show()
plt.close()

#%% Program 3.6: CapacityDCMC.m: Capacity of M-ary transmission through DCMC AWGN channel, EsN0dBs
from Modulations import NormFactor
from ChannelModels import add_awgn_noise

nSym = 10000
EsN0dBs = np.arange(-10, 40, 1)
# EsN0s = np.arange(-10, 40, 1)
channelModel = 'awgn'

mod_type = 'qam'
if mod_type == 'pam' or mod_type == 'psk':
    arrayOfM = [2, 4, 8, 16, 32, 64]
elif mod_type == 'qam':
    arrayOfM = [4, 16, 64, 256]

plotColor = ['b', 'g', 'c', 'm', 'orange', 'pink']
j = 1

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

for m, M in enumerate(arrayOfM):
    k = int(np.log2(M))
    # EsN0dBs = 10 * np.log10(k) + EbN0dBs
    C_sim = np.zeros(len(EsN0dBs))

    bps = int(np.log2(M))
    if mod_type == 'qam':
        modem = commpy.QAMModem(M)
    elif mod_type == 'psk':
        modem =  commpy.PSKModem(M)
    Es = NormFactor(mod_type = mod_type, M = M,)
    # map_table, demap_table = modem.plot_constellation(f"{M}-{modutype.upper()} ")
    # print(f"map_table = \n{map_table}\ndemap_table = \n{demap_table}")

    uu = np.random.randint(0, 2, size = nSym * bps).astype(np.int8)
    s = modem.modulate(uu)
    for i, EsN0dB in enumerate(EsN0dBs):
        if channelModel == 'rayleigh':
            h = (np.random.randn(1, nSym) + 1j * np.random.randn(1, nSym))/np.sqrt(2)
        elif channelModel == 'awgn':
            h = np.ones((1, nSym))
        hs = h * s
        r, N0 = add_awgn_noise(hs, EsN0dB)

        pdfs = np.exp(-np.abs( np.ones((M, 1)) @ r - modem.constellation[:,None] @ h)**2/N0)
        p = np.maximum(pdfs, 1e-20)
        p = p / np.sum(p, axis = 0)
        symEntropy = -np.sum(p * np.log2(p), axis = 0)
        C_sim[i] = np.log2(M) - np.mean(symEntropy)# / M / (np.sqrt(np.pi))**2
    axs.plot(EsN0dBs, C_sim, color = plotColor[m], ls = '-', label = f'{M}-{mod_type.upper()}' )

if channelModel == 'awgn':
    C_theory = np.log2(1 + 10**(EsN0dBs/10))

elif channelModel == 'rayleigh':
    h = (np.random.randn(1, nSym) + 1j * np.random.randn(1, nSym))/np.sqrt(2)
    sigma_z = 1
    snr = 10**(EsN0dBs/10)
    P = (sigma_z**2) * snr / np.mean(np.abs(h)**2)
    C_theory = np.mean(np.log2(1 + (np.abs(h)**2).T @ P.reshape(1, -1) / sigma_z**2 ), axis = 0)
axs.plot(EsN0dBs, C_theory, color = 'r', ls = '-',  )
axs.text(20, 7.2, f"Capacity limit on {channelModel.upper()} channel", color='r', size = 20, rotation = 41., ha = "center", va="center",)

axs.set_xlabel( 'Es/N0 (dB)',)
axs.set_ylabel('Capacity (bits/sym)',)
axs.set_title(f"Constrained Capacity for {mod_type.upper()} on {channelModel.upper()} channel")
axs.legend(fontsize = 20)
axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
plt.show()
plt.close()

#%% Program 3.6: CapacityDCMC.m: Capacity of M-ary transmission through DCMC AWGN channel, EbN0dBs
from Modulations import NormFactor
from ChannelModels import add_awgn_noise

nSym = 10000
EbN0dBs = np.arange(-10, 40, 1)
# EsN0s = np.arange(-10, 40, 1)
# channelModel = 'rayleigh'

mod_type = 'qam'
if mod_type == 'pam' or mod_type == 'psk':
    arrayOfM = [2, 4, 8, 16, 32, 64]
elif mod_type == 'qam':
    arrayOfM = [4, 16, 64, 256]

plotColor = ['b', 'g', 'c', 'm', 'orange', 'pink']
j = 1

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

for m, M in enumerate(arrayOfM):
    k = int(np.log2(M))
    EsN0dBs = 10 * np.log10(k) + EbN0dBs
    C_sim = np.zeros(len(EsN0dBs))

    bps = int(np.log2(M))
    if mod_type == 'qam':
        modem = commpy.QAMModem(M)
    elif mod_type == 'psk':
        modem =  commpy.PSKModem(M)
    Es = NormFactor(mod_type = mod_type, M = M,)
    # map_table, demap_table = modem.plot_constellation(f"{M}-{modutype.upper()} ")
    # print(f"map_table = \n{map_table}\ndemap_table = \n{demap_table}")

    uu = np.random.randint(0, 2, size = nSym * bps).astype(np.int8)
    s = modem.modulate(uu)
    for i, EsN0dB in enumerate(EsN0dBs):
        if channelModel == 'rayleigh':
            h = (np.random.randn(1, nSym) + 1j * np.random.randn(1, nSym))/np.sqrt(2)
        elif channelModel == 'awgn':
            h = np.ones((1, nSym))
        hs = h * s
        r, N0 = add_awgn_noise(hs, EsN0dB)

        pdfs = np.exp(-np.abs( np.ones((M, 1)) @ r - modem.constellation[:,None] @ h)**2/N0)
        p = np.maximum(pdfs, 1e-20)
        p = p / np.sum(p, axis = 0)
        symEntropy = -np.sum(p * np.log2(p), axis = 0)
        C_sim[i] = np.log2(M) - np.mean(symEntropy)# / M / (np.sqrt(np.pi))**2
    axs.plot(EsN0dBs, C_sim, color = plotColor[m], ls = '-', label = f'{M}-{mod_type.upper()}' )

if channelModel == 'awgn':
    C_theory = np.log2(1 + 10**(EbN0dBs/10))
    # k = np.arange(0.001, 15, 0.001)
    # EbN0 = (2**k-1)/k
    # EbN0dB = 10 * np.log10(EbN0)
    axs.plot(EbN0dBs, C_theory, color = 'r', ls = '-',  )
elif channelModel == 'rayleigh':
    h = (np.random.randn(1, nSym) + 1j * np.random.randn(1, nSym))/np.sqrt(2)
    sigma_z = 1
    snr = 10**(EbN0dBs/10)
    P = (sigma_z**2) * snr / np.mean(np.abs(h)**2)
    C_theory = np.mean(np.log2(1 + (np.abs(h)**2).T @ P.reshape(1, -1) / sigma_z**2 ), axis = 0)
    axs.plot(EbN0dBs, C_theory, color = 'r', ls = '-', )
axs.text(20, 7.2, f"Capacity limit on {channelModel.upper()} channel", color='r', size = 20, rotation = 41., ha = "center", va="center",)

axs.set_xlabel( 'Eb/N0 (dB)',)
axs.set_ylabel('Capacity (bits/sym)',)
axs.set_title(f"Constrained Capacity for {mod_type.upper()} on {channelModel.upper()} channel")
axs.legend(fontsize = 20)
axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
plt.show()
plt.close()



#%% get_AWGN_capacity.m











































