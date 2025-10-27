#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 21:15:04 2024

@author: jack

上变频和脉冲成型是一起的，下变频和匹配滤波也是一起的, upfirdn

Eb/N0-> SNR:
    https://wenku.csdn.net/answer/7umbbeqdrq#:~:text=%E5%B0%86SNR%E8%BD%AC%E6%8D%A2%E4%B8%BAEb%2FN0%EF%BC%8C%E5%8F%AF%E4%BB%A5%E4%BD%BF%E7%94%A8%E4%BB%A5%E4%B8%8B%E5%85%AC%E5%BC%8F%EF%BC%9A%20Eb%2FN0%20%3D%20SNR,%2F%20%28log2%20%28M%29%20%2A%20R%29

https://mp.weixin.qq.com/s?__biz=MzIxNTQ1NDM2OA==&mid=2247488886&idx=1&sn=8bfa9a91c3fe937f0ab19e3347b1b63d&chksm=979952b9a0eedbafbd0d51d0812eb524d8dbbdecdc231341f15aa284d22935bb2fd6b8e36be9&mpshare=1&scene=1&srcid=0804bmYW6ucAqcCjQNVt2aym&sharer_shareinfo=b2f9db6c5b1538f2b7dce733853cd6b7&sharer_shareinfo_first=b2f9db6c5b1538f2b7dce733853cd6b7&exportkey=n_ChQIAhIQpQwYljFyQJhJ%2BQxbaK7vDBKfAgIE97dBBAEAAAAAALUCFGHtXCIAAAAOpnltbLcz9gKNyK89dVj0ag2eslKVksUO1Xb6xnEbgja6z7rRQ33hqLmK5NeF4CZBMAU%2B5iCnj%2Fo1S70hICUMnObMou%2FxaAUk0YjrERe1yTNKFHZbSSD5fD%2FwBZyWFSN4p%2FrWxxugkk3PejnGOZ%2FJTQ5c000rzQQqZRu%2BgaQqzmi%2BeagnIC4SKiKCgQOlfYx2xOFLMOjz0qqaw4x%2F%2BL1MwQ3froyZg86iMkfw68EpraZ4rnZdbvQH6110v2x1c0rIGCQ4Do8GenfH2pUishOM3xuUe28kwbOgGFVjZycM4OHzBulf2yhVax9h%2Fl%2FJXWZFeHt9J2v85f2HavShVr2bTi5EGEu1PAP2&acctmode=0&pass_ticket=Lw4FgclVXK%2FrYgor%2BcHdksv%2FYctkz7NnkkuQQZiKDMNgS8udyuWROHCjUzo7FH6H&wx_header=0#rd

https://blog.csdn.net/fengshao1370/article/details/106059388

转换为dB单位则为：
对于复信号： Es/N0(dB)=10log10(Tsym/Tsamp)+SNR(dB);
对于实信号：Es/N0(dB)=10log10(0.5*Tsym/Tsamp)+SNR(dB);
    N0:噪声的单边功率谱密度
    Rb：比特率，即每秒传输多少个bit的二进制数
    Rs：符号率，每秒传输多少个符号的数据
    K：每个符号所承载的二进制bit数。
    Tsym：符号周期，每个符号持续的时间，易知Tsym = 1/Rs，单位秒。
    Tsamp：采样周期，每个采样点持续的时间，Tsamp = 1/Fs，其中Fs为采样率。
    Bn：噪声带宽，单位赫兹，对于awgn噪声，有 Bn=Fs=1/Tsamp 。
    sps：每个符号的采样个数，显然sps = Fs/Rs

Tsym/Tsamp = Fs/Rs = sps

Es/N0和Eb/N0的关系: 其中Es和Eb分别是每个符号和每个比特上的能量
    Es/N0(dB) = Eb/N0(dB) + 10log10(k)(dB), 其中k为每个符号上的信息比特数，k = log2(M),M为调制阶数
Es/N0和SNR的关系:
    对于复信号： Es/N0(dB)=10log10(Tsym/Tsamp)+SNR(dB) ;
    对于实信号：Es/N0(dB)=10log10(0.5*Tsym/Tsamp)+SNR(dB) ;
Eb/N0和SNR的关系:
对于复信号： Eb/N0(dB)=10log10(Tsym/Tsamp)+SNR(dB) - 10log10(k);
对于实信号：Eb/N0(dB)=10log10(0.5*Tsym/Tsamp)+SNR(dB) - 10log10(k);


https://www.docin.com/p-1557770101.html
https://www.cnblogs.com/devindd/articles/16793289.html
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

Modulation_type = 'BPSK'
savedir = f"./figures/{Modulation_type}/"
os.makedirs(savedir, exist_ok = True)

m_map = {"BPSK": [1,2], "QPSK": [2,4], "8PSK": [3,8], "QAM16": [4,16], "QAM64": [6,64]}
M = m_map[Modulation_type][1]
k = m_map[Modulation_type][0]   # 调制阶数 k = log2(M)

sps = 16                     # 每个符号的采样点数
fc = 200000                  # 载波频率, Hz
fs = 800000                  # 对载波的采样频率, Hz

Q = 8                            # 量化比特数
Fs = 400 # bit_rate/Q            # 对原始基带信号的采样频率
Ts = 1/Fs                        # 采样时间间隔
bit_rate = Fs*Q                  # 比特率
symbol_rate = bit_rate/k         # 符号率
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
    # out_fig.savefig(savedir + 'RawSig.eps', )
    # out_fig.savefig(savedir + 'RawSig.png', dpi=1000,)
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
    k = 1
elif Modulation_type == "QPSK":
    modem = cpy.PSKModem(4)
    k = 2
elif Modulation_type == "8PSK":
    modem = cpy.PSKModem(8)
    k = 3
elif Modulation_type == "QAM16":
    modem = cpy.QAMModem(16)
    k = 4
elif Modulation_type == "QAM64":
    modem = cpy.QAMModem(64)
    k = 6

map_table = {}
for idx, posi in enumerate(modem.constellation):
    map_table[idx] = posi
# draw_mod_constellation(map_table, modu_type = Modulation_type)

symbol = modem.modulate(Tx_bits)

## 发送符号的星座图
# draw_trx_constellation(symbol, map_table, tx = 1, modu_type = Modulation_type)


#%% 设置脉冲成型滤波器
beta = 0.5        # 滚降因子,可调整
span = 6
h = rcosdesign(beta, span, sps, 'sqrt')

## 脉冲成型 + 上变频-> 基带信号
s_t = scipy.signal.upfirdn(h, symbol, sps)

#%% 基带信号 s(t) 频谱
if isplot:
    if s_t.size % 2 == 1:
        FFTN = s_t.size + 1        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
    else:
        FFTN = s_t.size
    # FFTN = 1000000
    t_st = np.arange(0, s_t.size) * (1/(symbol_rate * sps))

    # 对时域采样信号, 执行快速傅里叶变换 FFT
    FFTs_t = scipy.fftpack.fft(s_t, n = FFTN)
    # 消除相位混乱
    FFTs_t[np.abs(FFTs_t) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

    # FFTN = FFTs_t.size
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
    # df = symbol_rate * sps /FFTN             # 频率间隔
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
    # out_fig.savefig(savedir + f's(t)_{Modulation_type}.eps', )
    # out_fig.savefig(savedir + f's(t)_{Modulation_type}.png', dpi = 1000,)
    plt.show()

#%% 载波
time = np.arange(s_t.size)/ fs
s_fc = np.exp(1j * 2 * np.pi * fc * time)
s_RF = s_t * s_fc


#%% 载波频谱
if isplot:
    ##========================================== 载波 ===========================================
    # t_st = np.arange(0, s_RF.size) * (1/(symbol_rate * sps))
    if s_fc.size % 2 == 1:
        FFTN = s_fc.size + 1        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
    else:
        FFTN = s_fc.size
    # FFTN = 1000000
    # 对时域采样信号, 执行快速傅里叶变换 FFT
    FFTs_fc = scipy.fftpack.fft(s_fc, n = FFTN)
    # 消除相位混乱
    FFTs_fc[np.abs(FFTs_fc) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

    # FFTN = FFTs_fc.size
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
    axs[0].plot(t_st, s_fc, label = 's(t)', linewidth = 2, color = 'b',  )

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
    # out_fig.savefig(savedir + 'Carrier.eps', )
    # out_fig.savefig(savedir + 'Carrier.png', dpi = 1000,)
    plt.show()


#%% 已调信号频谱
if isplot:
    ##========================================== 已调信号 ===========================================
    # 对时域采样信号, 执行快速傅里叶变换 FFT
    if s_RF.size % 2 ==1:
        FFTN = s_RF.size + 1
    else:
        FFTN = s_RF.size
    # FFTN = 1000000
    FFTs_rf = scipy.fftpack.fft(s_RF, n = FFTN)
    # 消除相位混乱
    FFTs_rf[np.abs(FFTs_rf) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

    # FFTN = FFTs_rf.size
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
    # df = fs /FFTN             # 频率间隔
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
    axs[0].plot(t_st, s_RF, label = 's(t)', linewidth = 2, color = 'b',  )

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
    # out_fig.savefig(savedir + f'RF(t)_{Modulation_type}.eps', )
    # out_fig.savefig(savedir + f'RF(t)_{Modulation_type}.png', dpi = 1000,)
    plt.show()

#%% 信道
source = SourceSink()
source.InitLog()

ebn0  = np.arange(-10, 11, 1) # dB
ebn0  = np.arange(-20, 90, 10) # dB
SNR = ebn0 - 10 * np.log10(sps/2) + 10 * np.log10(k)      # dB
# SNR = np.arange(-10, 1)
for idx, snr in enumerate(SNR):
    ext = f"_{Modulation_type}_snr={int(snr)}"
    source.ClrCnt()
    ## AWGN
    signal_pwr = np.mean(abs(s_RF)**2)
    noise_pwr = signal_pwr/(10**(snr/10))
    noise = 1/np.sqrt(2) * (np.random.randn(len(s_RF)) + 1j * np.random.randn(len(s_RF))) * np.sqrt(noise_pwr)
    y = s_RF + noise

    #%% 过信道后信号的频谱
    if isplot:
        if idx == ebn0.size - 1:
            if y.size % 2 == 1:
                FFTN = y.size + 1
            else:
                FFTN = y.size
            # FFTN = 1000000
            t_y = np.arange(0, s_t.size) * (1/(symbol_rate * sps))

            # 对时域采样信号, 执行快速傅里叶变换 FFT
            FFT_y = scipy.fftpack.fft(y, n = FFTN)
            # 消除相位混乱
            FFT_y[np.abs(FFT_y) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

            # FFTN = FFT_y.size
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
            # df = fs /FFTN             # 频率间隔
            # 方法一
            # f = np.arange(-int(FFTN/2), int(FFTN/2))*df      # 频率刻度,N为偶数
            #或者如下， 方法二：
            f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/(fs) ))
            width = 6
            high = 5
            horvizen = 2
            vertical = 1
            fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*high, vertical*width), constrained_layout = True)

            ## y(t)
            axs[0].plot(t_y, y, label = 'channel(t)', linewidth = 2, color = 'b',  )

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
            # out_fig.savefig(savedir + f'channel_{Modulation_type}_snr={int(snr)}(dB).eps', )
            # out_fig.savefig(savedir + f'channel_{Modulation_type}_snr={int(snr)}(dB).png', dpi = 1000,)
            plt.show()

    #%%===============================================================================
    ###          接受机
    ###===============================================================================
    ## 相干解调
    # time = np.arange(s_t.size)/fs
    y_coherent = y * np.exp(-1j * 2 * np.pi * fc * time )

    #%% 相干解调后信号的频谱
    if isplot:
        if idx == ebn0.size - 1:
            if y_coherent.size % 2 == 1:
                FFTN = y_coherent.size + 1
            else:
                FFTN = y_coherent.size
            # FFTN = 1000000
            t_coherent = np.arange(0, s_t.size) * (1/(symbol_rate * sps))
            # FFTN = 2000        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
            # 对时域采样信号, 执行快速傅里叶变换 FFT
            FFT_coherent = scipy.fftpack.fft(y_coherent, n = FFTN)
            # 消除相位混乱
            FFT_coherent[np.abs(FFT_coherent) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

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
            f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/fs ))
            width = 6
            high = 5
            horvizen = 2
            vertical = 1
            fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*high, vertical*width), constrained_layout = True)

            ##  coherent(t)
            axs[0].plot(t_coherent, y_coherent, label = 'coherent demod', linewidth = 2, color = 'b',  )

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

            ##  FFT y_coherent(t)
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
            # axs[1].set_xscale('log', base = 10)

            out_fig = plt.gcf()
            # out_fig.savefig(savedir + f'coherent{Modulation_type}_snr={int(snr)}(dB).eps', )
            # out_fig.savefig(savedir + f'coherent_{Modulation_type}_snr={int(snr)}(dB).png', dpi = 1000,)
            plt.show()
        pass

    #%% 下采样 + 匹配滤波 -> 恢复的基带信号
    s_t_hat = scipy.signal.upfirdn(h, y_coherent, 1, sps)

    #%%选取最佳采样点,
    decision_site = int((s_t_hat.size - symbol.size) / 2)

    ## 每个符号选取一个点作为判决
    I_n_hat = s_t_hat[decision_site:decision_site + symbol.size]
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

N = 30
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
# out_fig.savefig(savedir + f'wave_{Modulation_type}_EbN0={ebn0[-1]}(dB).eps', )
# out_fig.savefig(savedir + f'wave_{Modulation_type}_EbN0={ebn0[-1]}(dB).png', dpi=1000,)
plt.show()





