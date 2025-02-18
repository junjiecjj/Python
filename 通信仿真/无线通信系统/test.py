#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:00:10 2025

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






#%% Program 3.6: CapacityDCMC.m: Capacity of M-ary transmission through DCMC AWGN channel

from Modulations import NormFactor
from ChannelModels import add_awgn_noise

nSym = 10000
snrdB = np.arange(-10, 36, 1)
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
    C_sim = np.zeros(len(snrdB))

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
    for i, snrdb in enumerate(snrdB):
        if channelModel == 'rayleigh':
            h = (np.random.randn(1, nSym) + 1j * np.random.randn(1, nSym))/np.sqrt(2)
        elif channelModel == 'awgn':
            h = np.ones((1, nSym))
        hs = h * s
        r, N0 = add_awgn_noise(hs, snrdb)

        pdfs = np.exp(-np.abs( np.ones((M, 1)) @ r - modem.constellation[:,None] @ h)**2/N0)
        p = np.maximum(pdfs, 1e-20)
        p = p / np.sum(p, axis = 0)
        symEntropy = -np.sum(p * np.log2(p), axis = 0)
        C_sim[i] = np.log2(M) - np.mean(symEntropy)# / M / (np.sqrt(np.pi))**2
    axs.plot(snrdB, C_sim, color = plotColor[m], ls = '-', label = f'{M}-{mod_type.upper()}' )


if channelModel == 'awgn':
    C_theory = np.log2(1 + 10**(snrdB/10))
elif channelModel == 'rayleigh':
    h = (np.random.randn(1, 10000) + 1j * np.random.randn(1, 10000))/np.sqrt(2)
    sigma_z = 1
    snr = 10**(snrdB/10)
    P = (sigma_z**2) * snr / np.mean(np.abs(h)**2)
    C_theory = np.mean(np.log2(1 + (np.abs(h)**2).T @ P.reshape(1, -1) / sigma_z**2 ), axis = 0)

axs.plot(snrdB, C_theory, color = 'r', ls = '-',  )
axs.text(20, 7.2, f"Capacity limit on {channelModel.upper()} channel", color='r', size = 20, rotation = 41., ha = "center", va="center",)

axs.set_xlabel( 'SNR (dB)',)
axs.set_ylabel('Capacity (bits/sym)',)
axs.set_title(f"Constrained Capacity for {mod_type.upper()} on {channelModel.upper()} channel")
axs.legend(fontsize = 20)

plt.show()
plt.close()


