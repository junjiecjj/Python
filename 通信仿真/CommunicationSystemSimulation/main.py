#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 21:15:04 2024

@author: jack
"""
## 系统库
import numpy as np
import scipy
import commpy as cpy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

## 自编库
from Rcosdesign import rcosdesign
from Quantization import QuantizationBbits_NP_int
from Quantization import deQuantizationBbits_NP_int

filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

bit_rate = 1000            # 比特率
M = 1                      # 调制阶数
symbol_rate = bit_rate/M   # 符号率
lf = 1600
sps = 16                   # 每个符号的采样点数
fc = 2000                  # 载波频率， Hz
fs = 16000                 # 对载波的采样频率, Hz

Q = 8                            # 量化比特数
Fs = 400 # bit_rate/Q                  # 对原始基带信号的采样频率
f0 = 20                          # 正弦波的频率
Ts = 1/Fs                        # 采样时间间隔
len_bits = 1000                # 量化后比特长度
N = int(len_bits/Q)              # 采样信号的长度
t = np.linspace(0, N-1, N)*Ts    # 定义信号采样的时间点 t
x = np.sin(2*np.pi*f0*t )

BER = []
###================================================================================
###          发送
###================================================================================

#%% 量化
Tx_bits = QuantizationBbits_NP_int(x, Q)
#%% 调制
symbol = 1 - 2 * Tx_bits
# BPSK = cpy.PSKModem(2)
# symbol = BPSK.modulate(Tx_bits)

#%% 插值
I_n = np.zeros(sps * symbol.size, )
I_n[::sps] = symbol


#%% 脉冲成型滤波器
beta = 0.5        # 滚降因子,可调整
h = rcosdesign(beta, 6, sps, 'sqrt')
## 脉冲成型
s_t = scipy.signal.lfilter(h, 1, I_n) # 进行滤波

#%% 载波
time = np.arange(s_t.size)
s_RF = s_t * np.cos(2 * np.pi * fc * time / fs)
## 后续考虑直接用复数载波信号实现，更加方便

#%% 信道
ebn0  = np.arange(1, 10)
SNR = ebn0 - 10 * np.log10(0.5*16);
# SNR = np.arange(-10, 1)
for idx, snr in enumerate(SNR):
    ## AWGN
    signal_pwr = np.mean(abs(s_RF)**2)
    noise_pwr = signal_pwr/(10**(snr/10))
    # noise = 1/np.sqrt(2) * (np.random.randn(len(s_RF)) + 1j * np.random.randn(len(s_RF))) * np.sqrt(noise_pwr)
    noise =  np.random.randn(len(s_RF)) * np.sqrt(noise_pwr)
    y = s_RF + noise

    #%%===============================================================================
    ###          接受机
    ###===============================================================================
    ## 相干解调
    y_coherent = y * np.cos(2 * np.pi * fc * time / fs)

    ##%% 低通滤波
    Wn = 2 * lf / fs

    ###### 方法1
    # [Bb, Ba] = scipy.signal.butter(30, Wn, 'low')
    # y_lowpass = scipy.signal.lfilter(Bb, Ba, y_coherent) # 进行滤波

    ###### 方法2
    h = scipy.signal.firwin(int(128), Wn )
    y_lowpass = scipy.signal.lfilter(h, 1, y_coherent) # 进行滤波

    #%% 匹配滤波
    beta = 0.5        # 滚降因子,可调整
    h = rcosdesign(beta, 6, sps, 'sqrt')
    ## 匹配滤波
    s_t_hat = scipy.signal.lfilter(h, 1, y_lowpass) # 进行滤波

    #%%选取最佳采样点
    decision_site = 160      # (96+128+96)/2 =160 三个滤波器的延迟 96 128 96

    ## 每个符号选取一个点作为判决
    I_n_hat = s_t_hat[decision_site - 1 :: sps]
    # 涉及到三个滤波器，固含有滤波器延迟累加

    ## 解调
    # Rx_bits = BPSK.demodulate(I_n_hat, demod_type = 'hard')
    Rx_bits = np.array( (1 - np.sign(I_n_hat)) /2, dtype = np.int8)

    ber = np.sum(Rx_bits != Tx_bits[:Rx_bits.size])/Rx_bits.size
    BER.append(ber)

print(BER)
#%% 反量化
x_hat = deQuantizationBbits_NP_int(Rx_bits, Q)


#%% 画BER
# width = 6
# high = 4
# horvizen = 1
# vertical = 1
# fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*width, vertical*high), constrained_layout = True)
# labelsize = 20

# axs.semilogy(ebn0, BER, label = 'BER', linewidth = 2, color = 'b',  )

# font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
# axs.set_xlabel('Eb/N0',fontproperties=font)
# axs.set_ylabel('BER',fontproperties=font)
# font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# #  edgecolor='black',
# # facecolor = 'y', # none设置图例legend背景透明
# legend1 = axs.legend(loc='best',  prop=font1, borderaxespad=0,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# # frame1.set_facecolor('none')  # 设置图例legend背景透明

# axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
# axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
# axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
# axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

# axs.tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
# labels = axs.get_xticklabels() + axs.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(25) for label in labels] #刻度值字号
# axs.grid( linestyle = '--', linewidth = 0.5, )

# out_fig = plt.gcf()
# # out_fig.savefig('BER.eps', )
# # out_fig.savefig('BER.png', dpi=1000,)
# plt.show()


#%% 画前后波形
width = 6
high = 4
horvizen = 1
vertical = 1
fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*width, vertical*high), constrained_layout = True)
labelsize = 20

axs.plot(t, x, label = 'transmit', linewidth = 2, color = 'b',  )
t1 = t[1:]
axs.plot(t1, x_hat, label = 'receive', linewidth = 2, color = 'r', linestyle = '-' )


font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
axs.set_xlabel('Eb/N0',fontproperties=font)
axs.set_ylabel('BER',fontproperties=font)
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
#  edgecolor='black',
# facecolor = 'y', # none设置图例legend背景透明
legend1 = axs.legend(loc='best',  prop=font1, borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs.tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(25) for label in labels] #刻度值字号
axs.grid( linestyle = '--', linewidth = 0.5, )

out_fig = plt.gcf()
out_fig.savefig('wave.eps', )
out_fig.savefig('wave.png', dpi=1000,)
plt.show()















































































































































































































































































































































































































































































































































































































































































































