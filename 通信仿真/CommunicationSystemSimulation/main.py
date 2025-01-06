#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 21:15:04 2024

@author: jack

上变频和脉冲成型是分离的，下变频和匹配滤波也是分离的.

"""
## 系统库
import numpy as np
import scipy
import os
import commpy as cpy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

## 自编库
from Rcosdesign import rcosdesign
from Quantization import QuantizationBbits_NP_int
from Quantization import deQuantizationBbits_NP_int
from sourcesink import SourceSink
from utility import draw_mod_constellation
from utility import draw_trx_constellation

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

Modulation_type = 'QAM16'
savedir = f"./figures/{Modulation_type}/"
os.makedirs(savedir, exist_ok = True)


m_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "QAM16": 4, "QAM64": 6}
M = m_map[Modulation_type]   # 调制阶数

sps = 16                     # 每个符号的采样点数
fc = 200000                  # 载波频率, Hz
fs = 800000                  # 对载波的采样频率, Hz

Q = 8                            # 量化比特数
Fs = 400 # bit_rate/Q            # 对原始基带信号的采样频率
Ts = 1/Fs                        # 采样时间间隔
bit_rate = Fs*Q                  # 比特率
symbol_rate = bit_rate/M         # 符号率
print(f"symbol_rate = {symbol_rate}, bit_rate = {bit_rate}\n")

f0 = 20                          # 正弦波的频率
f1 = 60
# f2 = 40

## 时间
t = np.arange(0, 20/f0, Ts) # 定义信号采样的时间点 t
## 初始波形
x = 1/3 * np.sin(2*np.pi*f0*t ) + 2/3 * np.sin(2*np.pi*f1*t )

isplot = 0

#%% 原始信号的频谱
if isplot:
    # FFTN = x.size
    FFTN = 1000000        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
    # 对时域采样信号, 执行快速傅里叶变换 FFT
    X = scipy.fftpack.fft(x, n = FFTN)

    #============
    # 半谱
    #============
    # 消除相位混乱
    X[np.abs(X) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

    # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
    X = X/x.size               # 将频域序列 X 除以序列的长度 N

    ## 将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
    Y = scipy.fftpack.fftshift(X, )

    # 计算频域序列 Y 的幅值和相角
    A = abs(Y);                       # 计算频域序列 Y 的幅值
    Pha = np.angle(Y,deg=True);       # 计算频域序列 Y 的相角 (弧度制)
    R = np.real(Y)                    # 计算频域序列 Y 的实部
    I = np.imag(Y)                    # 计算频域序列 Y 的虚部

    #  定义序列 Y 对应的频率刻度
    df = Fs/FFTN                           # 频率间隔
    if FFTN%2==0:
        # 方法一
        f = np.arange(-int(FFTN/2), int(FFTN/2))*df      # 频率刻度,N为偶数
        #或者如下， 方法二：
        # f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/Fs))

    width = 6
    high = 5
    horvizen = 2
    vertical = 1
    fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*high, vertical*width), constrained_layout = True)

    ## s(t)
    axs[0].plot(t, x, label = 'transmit', linewidth = 2, color = 'b',  )

    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
    axs[0].set_xlabel('Time(s)',fontproperties=font)
    axs[0].set_ylabel('x',fontproperties=font)
    # axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
    font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    #  edgecolor='black',
    # facecolor = 'y', # none设置图例legend背景透明
    legend1 = axs[0].legend(loc='best',  prop=font1, borderaxespad=0,)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    axs[0].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    axs[0].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
    axs[0].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
    axs[0].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

    axs[0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels] #刻度值字号
    axs[0].grid( linestyle = '--', linewidth = 0.5, )

    ##  FFT s(t)
    axs[1].plot(f, A, label = '|FFT(x)|', linewidth = 2, color = 'b',  )
    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
    axs[1].set_xlabel('Frequancy(Hz)',fontproperties=font)
    axs[1].set_ylabel('|FFT(x)|',fontproperties=font)

    font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    #  edgecolor='black',
    # facecolor = 'y', # none设置图例legend背景透明
    legend1 = axs[1].legend(loc='best',  prop=font1, borderaxespad=0,)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    axs[1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    axs[1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
    axs[1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
    axs[1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

    axs[1].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels] #刻度值字号
    axs[1].grid( linestyle = '--', linewidth = 0.5, )

    out_fig = plt.gcf()
    out_fig.savefig(savedir + 'RawSig.eps', )
    out_fig.savefig(savedir + 'RawSig.png', dpi=1000,)
    plt.show()


#%%
BER = []
###===============================================================================
###          发送
###===============================================================================
#%% 量化
Tx_bits = QuantizationBbits_NP_int(x, Q)
#%% 调制
if Modulation_type == "BPSK":
    modem = cpy.PSKModem(2)
elif Modulation_type == "QPSK":
    modem = cpy.PSKModem(4)
elif Modulation_type == "8PSK":
    modem = cpy.PSKModem(8)
elif Modulation_type == "QAM16":
    modem = cpy.QAMModem(16)
elif Modulation_type == "QAM64":
    modem = cpy.QAMModem(64)

map_table = {}
for idx, posi in enumerate(modem.constellation):
    map_table[idx] = posi
draw_mod_constellation(map_table, modu_type = Modulation_type)

symbol = modem.modulate(Tx_bits)

## 发送符号的星座图
draw_trx_constellation(symbol, map_table, tx = 1, modu_type = Modulation_type)

#%% 插值
I_n = np.zeros(sps * symbol.size, dtype = complex)
I_n[::sps] = symbol

#%% 脉冲成型滤波器
beta = 0.5        #   滚降因子,可调整
span = 6
h = rcosdesign(beta, span, sps, 'sqrt')
## 脉冲成型
s_t = scipy.signal.lfilter(h, 1, I_n) # 进行滤波

#%% 基带信号 s(t) 频谱
if isplot:
    t_st = np.arange(0, s_t.size) * (1/(symbol_rate * sps))
    # FFTN = 2000        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
    # 对时域采样信号, 执行快速傅里叶变换 FFT
    FFTs_t = scipy.fftpack.fft(s_t)
    # 消除相位混乱
    FFTs_t[np.abs(FFTs_t) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

    FFTN = FFTs_t.size
    # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
    FFTs_t = FFTs_t/FFTN               # 将频域序列 X 除以序列的长度 N

    assert FFTs_t.size %2 == 0
    ## 将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
    Y = scipy.fftpack.fftshift(FFTs_t, )

    # 计算频域序列 Y 的幅值和相角
    A = np.abs(Y)                        # 计算频域序列 Y 的幅值
    Pha = np.angle(Y, deg = True)      # 计算频域序列 Y 的相角 (弧度制)
    R = np.real(Y)                    # 计算频域序列 Y 的实部
    I = np.imag(Y)                    # 计算频域序列 Y 的虚部

    # 定义序列 Y 对应的频率刻度
    df = symbol_rate * sps /FFTN             # 频率间隔
    # 方法一
    # f = np.arange(-int(FFTN/2), int(FFTN/2))*df      # 频率刻度,N为偶数
    #或者如下， 方法二：
    f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/(symbol_rate * sps) ))
    width = 6
    high = 5
    horvizen = 2
    vertical = 1
    fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*high, vertical*width), constrained_layout = True)

    ## s(t)
    axs[0].plot(t_st, s_t, label = 's(t)', linewidth = 2, color = 'b',  )

    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
    axs[0].set_xlabel('Time(s)',fontproperties=font)
    axs[0].set_ylabel('s(t)',fontproperties=font)
    axs[0].set_title(f'{Modulation_type}',  fontproperties=font )
    font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    #  edgecolor='black',
    # facecolor = 'y', # none设置图例legend背景透明
    legend1 = axs[0].legend(loc='best',  prop=font1, borderaxespad=0,)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    axs[0].spines['bottom'].set_linewidth(2)    ###设置底部坐标轴的粗细
    axs[0].spines['left'].set_linewidth(2)      ###设置左边坐标轴的粗细
    axs[0].spines['right'].set_linewidth(2)     ###设置右边坐标轴的粗细
    axs[0].spines['top'].set_linewidth(2)       ###设置上部坐标轴的粗细

    axs[0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels] #刻度值字号
    axs[0].grid( linestyle = '--', linewidth = 0.5, )

    ##  FFT s(t)
    axs[1].plot(f, A, label = 'FFT|s(t)|', linewidth = 2, color = 'b',  )

    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
    axs[1].set_xlabel('Frequancy(Hz)',fontproperties=font)
    axs[1].set_ylabel('|FFT(s(t))|',fontproperties=font)
    # axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
    font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

    legend1 = axs[1].legend(loc='best',  prop=font1, borderaxespad=0, facecolor = 'none')
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    axs[1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    axs[1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
    axs[1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
    axs[1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

    axs[1].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels] #刻度值字号
    axs[1].grid(linestyle = '--', linewidth = 0.5, )

    out_fig = plt.gcf()
    out_fig.savefig(savedir + f's(t)_{Modulation_type}.eps', )
    out_fig.savefig(savedir + f's(t)_{Modulation_type}.png', dpi = 1000,)
    plt.show()

#%% 载波
time = np.arange(s_t.size)/ fs
s_fc = np.exp(1j * 2 * np.pi * fc * time)
s_RF = s_t * s_fc

#%% 载波频谱
if isplot:
    ##========================================== 载波 ===========================================
    # t_st = np.arange(0, s_RF.size) * (1/(symbol_rate * sps))
    # FFTN = 2000        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
    # 对时域采样信号, 执行快速傅里叶变换 FFT
    FFTs_fc = scipy.fftpack.fft(s_fc)
    # 消除相位混乱
    FFTs_fc[np.abs(FFTs_fc) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

    FFTN = FFTs_fc.size
    # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
    FFTs_fc = FFTs_fc/FFTN               # 将频域序列 X 除以序列的长度 N

    assert FFTs_fc.size %2 == 0
    ## 将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
    Y = scipy.fftpack.fftshift(FFTs_fc, )

    # 计算频域序列 Y 的幅值和相角
    A = abs(Y);                       # 计算频域序列 Y 的幅值
    Pha = np.angle(Y,deg = True)      # 计算频域序列 Y 的相角 (弧度制)
    R = np.real(Y)                    # 计算频域序列 Y 的实部
    I = np.imag(Y)                    # 计算频域序列 Y 的虚部

    # 定义序列 Y 对应的频率刻度
    df = fs /FFTN             # 频率间隔
    # 方法一
    # f = np.arange(-int(FFTN/2), int(FFTN/2))*df      # 频率刻度,N为偶数
    #或者如下， 方法二：
    f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/fs ))
    width = 6
    high = 5
    horvizen = 2
    vertical = 1
    fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*high, vertical*width), constrained_layout = True)

    ## fc(t)
    axs[0].plot(time, s_fc, label = 's(t)', linewidth = 2, color = 'b',  )

    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
    axs[0].set_xlabel('Time(s)',fontproperties=font)
    axs[0].set_ylabel('f(t)',fontproperties=font)
    # axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
    font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    #  edgecolor='black',
    # facecolor = 'y', # none设置图例legend背景透明
    legend1 = axs[0].legend(loc='best',  prop=font1, borderaxespad=0,)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    axs[0].spines['bottom'].set_linewidth(2)    ###设置底部坐标轴的粗细
    axs[0].spines['left'].set_linewidth(2)      ###设置左边坐标轴的粗细
    axs[0].spines['right'].set_linewidth(2)     ###设置右边坐标轴的粗细
    axs[0].spines['top'].set_linewidth(2)       ###设置上部坐标轴的粗细

    axs[0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels] #刻度值字号
    axs[0].grid( linestyle = '--', linewidth = 0.5, )

    ##  FFT fc(t)
    axs[1].plot(f, A, label = '|FFT(f(t))|', linewidth = 2, color = 'b',  )

    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
    axs[1].set_xlabel('Frequancy(Hz)',fontproperties=font)
    axs[1].set_ylabel('|FFT(f(t))|',fontproperties=font)
    # axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
    font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

    legend1 = axs[1].legend(loc='best',  prop=font1, borderaxespad=0, facecolor = 'none')
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    axs[1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    axs[1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
    axs[1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
    axs[1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

    axs[1].set_xscale('log', base = 10)
    axs[1].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels] #刻度值字号
    axs[1].grid(linestyle = '--', linewidth = 0.5, )

    out_fig = plt.gcf()
    out_fig.savefig(savedir + 'Carrier.eps', )
    out_fig.savefig(savedir + 'Carrier.png', dpi = 1000,)
    plt.show()



#%% 已调信号频谱
if isplot:
    ##========================================== 已调信号 ===========================================
    # 对时域采样信号, 执行快速傅里叶变换 FFT
    FFTs_rf = scipy.fftpack.fft(s_RF)
    # 消除相位混乱
    FFTs_rf[np.abs(FFTs_rf) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

    FFTN = FFTs_rf.size
    # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
    FFTs_rf = FFTs_rf/FFTN               # 将频域序列 X 除以序列的长度 N

    assert FFTs_rf.size %2 == 0
    ## 将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
    Y = scipy.fftpack.fftshift(FFTs_rf, )

    # 计算频域序列 Y 的幅值和相角
    A = abs(Y);                       # 计算频域序列 Y 的幅值
    Pha = np.angle(Y,deg = True)      # 计算频域序列 Y 的相角 (弧度制)
    R = np.real(Y)                    # 计算频域序列 Y 的实部
    I = np.imag(Y)                    # 计算频域序列 Y 的虚部

    # 定义序列 Y 对应的频率刻度
    df = fs /FFTN             # 频率间隔
    # 方法一
    # f = np.arange(-int(FFTN/2), int(FFTN/2))*df      # 频率刻度,N为偶数
    #或者如下， 方法二：
    f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/fs ))
    width = 6
    high = 5
    horvizen = 2
    vertical = 1
    fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*high, vertical*width), constrained_layout = True)

    ## s(t)
    axs[0].plot(time, s_RF, label = 's(t)', linewidth = 2, color = 'b',  )

    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
    axs[0].set_xlabel('Time(s)',fontproperties=font)
    axs[0].set_ylabel('RF(t)',fontproperties=font)
    axs[0].set_title(f'{Modulation_type}',  fontproperties=font )
    font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    #  edgecolor='black',
    # facecolor = 'y', # none设置图例legend背景透明
    legend1 = axs[0].legend(loc='best',  prop=font1, borderaxespad=0,)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    axs[0].spines['bottom'].set_linewidth(2)    ###设置底部坐标轴的粗细
    axs[0].spines['left'].set_linewidth(2)      ###设置左边坐标轴的粗细
    axs[0].spines['right'].set_linewidth(2)     ###设置右边坐标轴的粗细
    axs[0].spines['top'].set_linewidth(2)       ###设置上部坐标轴的粗细

    axs[0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 4 )
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels] #刻度值字号
    axs[0].grid( linestyle = '--', linewidth = 0.5, )

    ##  FFT s_rf(t)
    axs[1].plot(f, A, label = '|FFT(RF(t))|', linewidth = 2, color = 'b',  )

    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
    axs[1].set_xlabel('Frequancy(Hz)',fontproperties=font)
    axs[1].set_ylabel('|FFT(RF(t))|',fontproperties=font)
    # axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
    font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

    legend1 = axs[1].legend(loc='best',  prop=font1, borderaxespad=0, facecolor = 'none')
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    axs[1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    axs[1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
    axs[1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
    axs[1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

    axs[1].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=6, pad = 1 )
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels] #刻度值字号

    axs[1].grid(linestyle = '--', linewidth = 0.5, )
    axs[1].set_xscale('log', base = 10)
    axs[1].set_xlim(10**4, 10**6)  #拉开坐标轴范围显示投影
    out_fig = plt.gcf()
    out_fig.savefig(savedir + f'RF(t)_{Modulation_type}.eps', )
    out_fig.savefig(savedir + f'RF(t)_{Modulation_type}.png', dpi = 1000,)
    plt.show()

source = SourceSink()
source.InitLog( )
#%% 信道
ebn0  = np.arange(5, 7, 1)
SNR = ebn0 - 10 * np.log10(sps)
# SNR = np.arange(-10, 1)
for idx, snr in enumerate(SNR):
    ext = f"_{Modulation_type}_snr={int(snr)}"
    source.ClrCnt()
    ## AWGN
    signal_pwr = np.mean(abs(s_RF)**2)
    noise_pwr = signal_pwr/(10**(snr/10))
    noise = 1/np.sqrt(2) * (np.random.randn(len(s_RF)) + 1j * np.random.randn(len(s_RF))) * np.sqrt(noise_pwr)
    # noise =  np.random.randn(len(s_RF)) * np.sqrt(noise_pwr)
    y = s_RF + noise

    #%% 过信道后信号的频谱
    if isplot:
        if idx == ebn0.size - 1:
            t_lp = np.arange(0, s_t.size) * (1/(symbol_rate * sps))
            # FFTN = 2000        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
            # 对时域采样信号, 执行快速傅里叶变换 FFT
            FFT_y = scipy.fftpack.fft(y)
            # 消除相位混乱
            FFT_y[np.abs(FFT_y) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

            FFTN = FFT_y.size
            # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
            FFT_y = FFT_y/FFTN               # 将频域序列 X 除以序列的长度 N

            assert FFT_y.size %2 == 0
            ## 将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
            Y = scipy.fftpack.fftshift(FFT_y, )

            # 计算频域序列 Y 的幅值和相角
            A = abs(Y);                       # 计算频域序列 Y 的幅值
            Pha = np.angle(Y,deg = True)      # 计算频域序列 Y 的相角 (弧度制)
            R = np.real(Y)                    # 计算频域序列 Y 的实部
            I = np.imag(Y)                    # 计算频域序列 Y 的虚部

            # 定义序列 Y 对应的频率刻度
            df = fs /FFTN             # 频率间隔
            # 方法一
            # f = np.arange(-int(FFTN/2), int(FFTN/2))*df      # 频率刻度,N为偶数
            #或者如下， 方法二：
            f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/(fs) ))
            width = 6
            high = 5
            horvizen = 2
            vertical = 1
            fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*high, vertical*width), constrained_layout = True)

            ## s_hat(t)
            axs[0].plot(t_lp, s_t, label = 'channel(t)', linewidth = 2, color = 'b',  )

            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
            axs[0].set_xlabel('Time(s)',fontproperties=font)
            axs[0].set_ylabel('channel(t)',fontproperties=font)
            axs[0].set_title(f'{Modulation_type}, SNR = {int(snr)}(dB)',  fontproperties=font )
            font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
            #  edgecolor='black',
            # facecolor = 'y', # none设置图例legend背景透明
            legend1 = axs[0].legend(loc='best',  prop=font1, borderaxespad=0,)
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs[0].spines['bottom'].set_linewidth(2)    ###设置底部坐标轴的粗细
            axs[0].spines['left'].set_linewidth(2)      ###设置左边坐标轴的粗细
            axs[0].spines['right'].set_linewidth(2)     ###设置右边坐标轴的粗细
            axs[0].spines['top'].set_linewidth(2)       ###设置上部坐标轴的粗细

            axs[0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 4 )
            labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(25) for label in labels] #刻度值字号
            axs[0].grid( linestyle = '--', linewidth = 0.5, )

            ##  FFT s_hat(t)
            axs[1].plot(f, A, label = 'FFT|channel(t)|', linewidth = 2, color = 'b',  )

            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
            axs[1].set_xlabel('Frequancy(Hz)',fontproperties=font)
            axs[1].set_ylabel('|FFT(channel(t))|',fontproperties=font)
            # axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
            font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

            legend1 = axs[1].legend(loc='best',  prop=font1, borderaxespad=0, facecolor = 'none')
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs[1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
            axs[1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
            axs[1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
            axs[1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

            axs[1].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
            labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(25) for label in labels] #刻度值字号
            axs[1].grid(linestyle = '--', linewidth = 0.5, )

            axs[1].set_xscale('log', base = 10)
            axs[1].set_xlim(10**4, 10**6)  #拉开坐标轴范围显示投影

            out_fig = plt.gcf()
            out_fig.savefig(savedir + f'channel_{Modulation_type}_snr={int(snr)}(dB).eps', )
            out_fig.savefig(savedir + f'channel_{Modulation_type}_snr={int(snr)}(dB).png', dpi = 1000,)
            plt.show()

    #%%===============================================================================
    ###          接受机
    ###===============================================================================
    ## 相干解调
    time = np.arange(s_t.size)/ fs
    y_coherent = y * np.exp(-1j * 2 * np.pi * fc * time )

    #%% 相干解调后信号的频谱
    if isplot:
        if idx == ebn0.size - 1:
            t_lp = np.arange(0, s_t.size) * (1/(symbol_rate * sps))
            # FFTN = 2000        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
            # 对时域采样信号, 执行快速傅里叶变换 FFT
            FFT_coherent = scipy.fftpack.fft(y_coherent)
            # 消除相位混乱
            FFT_coherent[np.abs(FFT_coherent) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

            FFTN = FFT_coherent.size
            # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
            FFT_coherent = FFT_coherent/FFTN               # 将频域序列 X 除以序列的长度 N

            assert FFT_coherent.size %2 == 0
            ## 将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
            Y = scipy.fftpack.fftshift(FFT_coherent, )

            # 计算频域序列 Y 的幅值和相角
            A = abs(Y);                       # 计算频域序列 Y 的幅值
            Pha = np.angle(Y,deg = True)      # 计算频域序列 Y 的相角 (弧度制)
            R = np.real(Y)                    # 计算频域序列 Y 的实部
            I = np.imag(Y)                    # 计算频域序列 Y 的虚部

            # 定义序列 Y 对应的频率刻度
            df = symbol_rate * sps /FFTN             # 频率间隔
            # 方法一
            # f = np.arange(-int(FFTN/2), int(FFTN/2))*df      # 频率刻度,N为偶数
            #或者如下， 方法二：
            f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/(symbol_rate * sps) ))
            width = 6
            high = 5
            horvizen = 2
            vertical = 1
            fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*high, vertical*width), constrained_layout = True)

            ## s_lp(t)
            axs[0].plot(t_lp, s_t, label = 'coherent demod', linewidth = 2, color = 'b',  )

            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
            axs[0].set_xlabel('Time(s)',fontproperties=font)
            axs[0].set_ylabel('coherent demod',fontproperties=font)
            axs[0].set_title(f'{Modulation_type}, SNR = {int(snr)}(dB)',  fontproperties=font )
            font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
            #  edgecolor='black',
            # facecolor = 'y', # none设置图例legend背景透明
            legend1 = axs[0].legend(loc='best',  prop=font1, borderaxespad=0,)
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs[0].spines['bottom'].set_linewidth(2)    ###设置底部坐标轴的粗细
            axs[0].spines['left'].set_linewidth(2)      ###设置左边坐标轴的粗细
            axs[0].spines['right'].set_linewidth(2)     ###设置右边坐标轴的粗细
            axs[0].spines['top'].set_linewidth(2)       ###设置上部坐标轴的粗细

            axs[0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
            labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(25) for label in labels] #刻度值字号
            axs[0].grid( linestyle = '--', linewidth = 0.5, )

            ##  FFT s_lp(t)
            axs[1].plot(f, A, label = 'FFT|coherent|', linewidth = 2, color = 'b',  )

            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
            axs[1].set_xlabel('Frequancy(Hz)',fontproperties=font)
            axs[1].set_ylabel('|FFT(coherent)|',fontproperties=font)
            # axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
            font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

            legend1 = axs[1].legend(loc='best',  prop=font1, borderaxespad=0, facecolor = 'none')
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs[1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
            axs[1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
            axs[1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
            axs[1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

            axs[1].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
            labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(25) for label in labels] #刻度值字号
            axs[1].grid(linestyle = '--', linewidth = 0.5, )

            out_fig = plt.gcf()
            out_fig.savefig(savedir + f'coherent{Modulation_type}_snr={int(snr)}(dB).eps', )
            out_fig.savefig(savedir + f'coherent_{Modulation_type}_snr={int(snr)}(dB).png', dpi = 1000,)
            plt.show()

    #%% 低通滤波
    # Wn = 2 * lf / fs
    Wn = 0.2 #  截止频率为0.2*(fs/2)
    numtaps = 128
    ###### 方法1
    # [Bb, Ba] = scipy.signal.butter(numtaps, Wn, 'low')
    # y_lowpass = scipy.signal.lfilter(Bb, Ba, y_coherent) # 进行滤波

    ###### 方法2
    h = scipy.signal.firwin(numtaps, Wn )
    y_lowpass = scipy.signal.lfilter(h, 1, y_coherent) # 进行滤波

    #%% 下变频后信号的频谱
    if isplot:
        if idx == ebn0.size - 1:
            t_lp = np.arange(0, s_t.size) * (1/(symbol_rate * sps))
            # FFTN = 2000        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
            # 对时域采样信号, 执行快速傅里叶变换 FFT
            FFT_lp = scipy.fftpack.fft(y_lowpass)
            # 消除相位混乱
            FFT_lp[np.abs(FFT_lp) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

            FFTN = FFT_lp.size
            # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
            FFT_lp = FFT_lp/FFTN               # 将频域序列 X 除以序列的长度 N

            assert FFT_lp.size %2 == 0
            ## 将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
            Y = scipy.fftpack.fftshift(FFT_lp, )

            # 计算频域序列 Y 的幅值和相角
            A = abs(Y);                       # 计算频域序列 Y 的幅值
            Pha = np.angle(Y,deg = True)      # 计算频域序列 Y 的相角 (弧度制)
            R = np.real(Y)                    # 计算频域序列 Y 的实部
            I = np.imag(Y)                    # 计算频域序列 Y 的虚部

            # 定义序列 Y 对应的频率刻度
            df = symbol_rate * sps /FFTN             # 频率间隔
            # 方法一
            # f = np.arange(-int(FFTN/2), int(FFTN/2))*df      # 频率刻度,N为偶数
            #或者如下， 方法二：
            f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/(symbol_rate * sps) ))
            width = 6
            high = 5
            horvizen = 2
            vertical = 1
            fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*high, vertical*width), constrained_layout = True)

            ## s_lp(t)
            axs[0].plot(t_lp, s_t, label = 's_lp(t)', linewidth = 2, color = 'b',  )

            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
            axs[0].set_xlabel('Time(s)',fontproperties=font)
            axs[0].set_ylabel('s_lp(t)',fontproperties=font)
            axs[0].set_title(f'{Modulation_type}, SNR = {int(snr)}(dB)',  fontproperties=font )
            font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
            #  edgecolor='black',
            # facecolor = 'y', # none设置图例legend背景透明
            legend1 = axs[0].legend(loc='best',  prop=font1, borderaxespad=0,)
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs[0].spines['bottom'].set_linewidth(2)    ###设置底部坐标轴的粗细
            axs[0].spines['left'].set_linewidth(2)      ###设置左边坐标轴的粗细
            axs[0].spines['right'].set_linewidth(2)     ###设置右边坐标轴的粗细
            axs[0].spines['top'].set_linewidth(2)       ###设置上部坐标轴的粗细

            axs[0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
            labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(25) for label in labels] #刻度值字号
            axs[0].grid( linestyle = '--', linewidth = 0.5, )

            ##  FFT s_lp(t)
            axs[1].plot(f, A, label = 'FFT|s_lp(t)|', linewidth = 2, color = 'b',  )

            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
            axs[1].set_xlabel('Frequancy(Hz)',fontproperties=font)
            axs[1].set_ylabel('|FFT(s_lp(t))|',fontproperties=font)
            # axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
            font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

            legend1 = axs[1].legend(loc='best',  prop=font1, borderaxespad=0, facecolor = 'none')
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs[1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
            axs[1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
            axs[1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
            axs[1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

            axs[1].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
            labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(25) for label in labels] #刻度值字号
            axs[1].grid(linestyle = '--', linewidth = 0.5, )

            out_fig = plt.gcf()
            out_fig.savefig(savedir + f'lowpass_{Modulation_type}_snr={int(snr)}(dB).eps', )
            out_fig.savefig(savedir + f'lowpass_{Modulation_type}_snr={int(snr)}(dB).png', dpi = 1000,)
            plt.show()


    #%% 匹配滤波
    beta = 0.5        # 滚降因子,可调整
    h = rcosdesign(beta, span, sps, 'sqrt')
    s_t_hat = scipy.signal.lfilter(h, 1, y_lowpass) # 进行滤波

    #%% 匹配滤波后信号的频谱
    if isplot:
        if idx == ebn0.size - 1:
            t_lp = np.arange(0, s_t.size) * (1/(symbol_rate * sps))
            # FFTN = 2000        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
            # 对时域采样信号, 执行快速傅里叶变换 FFT
            FFT_rcos = scipy.fftpack.fft(s_t_hat)
            # 消除相位混乱
            FFT_rcos[np.abs(FFT_rcos) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

            FFTN = FFT_rcos.size
            # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
            FFT_rcos = FFT_rcos/FFTN               # 将频域序列 X 除以序列的长度 N

            assert FFT_rcos.size %2 == 0
            ## 将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
            Y = scipy.fftpack.fftshift(FFT_rcos, )

            # 计算频域序列 Y 的幅值和相角
            A = abs(Y)/4                       # 计算频域序列 Y 的幅值
            Pha = np.angle(Y,deg = True)      # 计算频域序列 Y 的相角 (弧度制)
            R = np.real(Y)                    # 计算频域序列 Y 的实部
            I = np.imag(Y)                    # 计算频域序列 Y 的虚部

            # 定义序列 Y 对应的频率刻度
            df = symbol_rate * sps /FFTN             # 频率间隔
            # 方法一
            # f = np.arange(-int(FFTN/2), int(FFTN/2))*df      # 频率刻度,N为偶数
            #或者如下， 方法二：
            f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/(symbol_rate * sps) ))
            width = 6
            high = 5
            horvizen = 2
            vertical = 1
            fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*high, vertical*width), constrained_layout = True)

            ## s_hat(t)
            axs[0].plot(t_lp, s_t, label = 's_hat(t)', linewidth = 2, color = 'b',  )

            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
            axs[0].set_xlabel('Time(s)',fontproperties=font)
            axs[0].set_ylabel('s_hat(t)',fontproperties=font)
            axs[0].set_title(f'{Modulation_type}, SNR = {int(snr)}(dB)',  fontproperties=font )
            font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
            #  edgecolor='black',
            # facecolor = 'y', # none设置图例legend背景透明
            legend1 = axs[0].legend(loc='best',  prop=font1, borderaxespad=0,)
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs[0].spines['bottom'].set_linewidth(2)    ###设置底部坐标轴的粗细
            axs[0].spines['left'].set_linewidth(2)      ###设置左边坐标轴的粗细
            axs[0].spines['right'].set_linewidth(2)     ###设置右边坐标轴的粗细
            axs[0].spines['top'].set_linewidth(2)       ###设置上部坐标轴的粗细

            axs[0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
            labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(25) for label in labels] #刻度值字号
            axs[0].grid( linestyle = '--', linewidth = 0.5, )

            ##  FFT s_hat(t)
            axs[1].plot(f, A, label = 'FFT|s_hat(t)|', linewidth = 2, color = 'b',  )

            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
            axs[1].set_xlabel('Frequancy(Hz)',fontproperties=font)
            axs[1].set_ylabel('|FFT(s_hat(t))|',fontproperties=font)
            # axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
            font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

            legend1 = axs[1].legend(loc='best',  prop=font1, borderaxespad=0, facecolor = 'none')
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs[1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
            axs[1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
            axs[1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
            axs[1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

            axs[1].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
            labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(25) for label in labels] #刻度值字号
            axs[1].grid(linestyle = '--', linewidth = 0.5, )

            out_fig = plt.gcf()
            out_fig.savefig(savedir + f'rcosdesig_{Modulation_type}_snr={int(snr)}(dB).eps', )
            out_fig.savefig(savedir + f'rcosdesig_{Modulation_type}_snr={int(snr)}(dB).png', dpi = 1000,)
            plt.show()

    #%%选取最佳采样点,延迟采样点个数=（成形滤波器阶数+低通滤波器阶数+匹配滤波器阶数）/2
    decision_site = int((span*sps + numtaps + span*sps)/2)  # (96+128+96)/2 =160 三个滤波器的延迟 96 128 96, 96 = span * sps

    ## 每个符号选取一个点作为判决
    I_n_hat = s_t_hat[decision_site - 1 :: sps]
    # 涉及到三个滤波器，固含有滤波器延迟累加

    ## 恢复符号的星座图
    # draw_trx_constellation(I_n_hat, map_table, tx = 0, snr = int(snr), channel='awgn', modu_type = Modulation_type)

    ## 解调
    Rx_bits = modem.demodulate(I_n_hat, demod_type = 'hard')
    source.CntErr(Tx_bits[:Rx_bits.size], Rx_bits)
    # ber = np.sum(Rx_bits != Tx_bits[:Rx_bits.size])/Rx_bits.size
    # BER.append(ber)
    # source.SaveToFile(snr = ebn0[idx])
    # print("  *** *** *** *** ***")
    source.PrintScreen(snr = snr)
    # print("  *** *** *** *** ***\n")
# print(BER)
#%% 反量化
x_hat = deQuantizationBbits_NP_int(Rx_bits, Q)


#%% 画前后波形
width = 6
high = 5
horvizen = 1
vertical = 1
fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*width, vertical*high), constrained_layout = True)
labelsize = 20

N = 50
axs.plot(t[:N], x[:N], label = 'transmit', linewidth = 2, color = 'b',  )
t1 = t[x.size - x_hat.size:]
axs.plot(t1[:N], x_hat[:N], label = 'receive', linewidth = 2, color = 'r', linestyle = '-' )

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
axs.set_xlabel('Time(s)',fontproperties=font)
axs.set_ylabel('Wave',fontproperties=font)
axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
#  edgecolor='black',
# facecolor = 'y', # none设置图例legend背景透明
legend1 = axs.legend(loc='best',  prop=font1, borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs.tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(25) for label in labels] #刻度值字号
axs.grid( linestyle = '--', linewidth = 0.5, )

out_fig = plt.gcf()
out_fig.savefig(savedir + f'wave_{Modulation_type}_EbN0={ebn0[-1]}(dB).eps', )
out_fig.savefig(savedir + f'wave_{Modulation_type}_EbN0={ebn0[-1]}(dB).png', dpi=1000,)
plt.show()





