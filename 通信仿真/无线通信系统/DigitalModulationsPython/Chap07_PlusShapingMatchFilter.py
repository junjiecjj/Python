#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 01:37:26 2025

@author: jack


关于模拟通信，数字基带传输，数字通带传输，均衡，OFDM的一些思考与总结：
(一) 模拟通信：主要是将信息承载信号(调制信号)加载到载波上，可以分为：调幅(AM)、调频(FM)，调相(PM)，后两者统称为角度调制。调幅就是用调制信号乘以载波；调频就是用调制信号的积分作为载波的相位；调相就是用调制信号作为载波的相位；三种模拟调制都可以用相干解调和非相干解调。相干解调主要是需要使用锁相环实现载波同步进而相干解调，而飞相干解调主要是通过Hilbert信号得到包络或者相位；
(二)  数字传输系统的数字信息既可以是来自计算机等终端的数字信号，也可以是模拟信号数字化处理化处理后的脉冲编码信号，这些信号所占据的频率是从零开始或很低频率开始，成为数字基带信号。数字基带传输的含义非常丰富：
     (1) 教科书上主要是从无码间串扰的角度阐述的，也就是奈奎斯特第一准则，使用脉冲成型和匹配滤波实现无码间串扰ISI。
         脉冲成型和匹配滤波:比特-> 符号 -> 上采样 -> 脉冲成型 -> AWGN信道 -> 匹配滤波 -> 下采样 -> 解调 -> 比特.
     (2) 而现在的很多通信书籍主要是将通信一步步模型抽象为 y = hx + n 的角度阐述的。
(三) 数字带通传输: 实际中大多数信道(如无线)因具有带通特性而不能直接传送基带信号，这是因为数字基带信号含有丰富的低频分量。为了使数字信号在带通信道中传输，必须使用数字基带信号对载波进行调制，以使得信号与信道匹配。这种用数字基带信号控制载波把数字基带信号变为数字带通信号的过程成为数字调制。把包括调制和解调过程的数字传输系统叫做数字带通传输系统。包括 幅移键控ASK、频移键控FSK、相移键控PSK已经新型数字带通调制技术：最小频移键控MSK和高斯最小频移键控GMSK。

(四) 均衡:首先理清楚什么时候需要均衡，均衡用在哪里。虽然从奈奎斯特第一准则中理论上找到了实现无ISI的方法，但是实际实现时难免存在滤波器的设计误差和信道特性的变化，无法实现理想的传输特性，故在抽样时刻总会存在一定的码间串扰，
    从而导致性能的下降。为了减少码间串扰的影响，通常在系统中插入一种可调滤波器来矫正或者补偿系统特性，这种起补偿作用的滤波器成为均衡器。
    通常在均衡系统中，已经把发送端滤波器，信道和接收方滤波器看成一个整体h(t)，y[n] = x[t]*h[t] + n[t]，注意，这时候其实是基带的模型，已经把模拟调制载波等细节都隐含在了h[t]中，信道其实已经是考虑了衰落等的。
    发送符号 -> 与综合信道卷积 -> +AWGN -> 与均衡系数卷积 -> 解调。

(五) OFDM: 在多径衰落的无线信道上，也会产生码间串扰，这时候为了解决这个问题，除了均衡器之外，还可以采用OFDM。注意，OFDM也是基带模型，发送符号 -> IFFT -> Add CP -> 信道卷积 (计算频域信道) -> AWGN -> Remove CP -> FFT -> 除以频域信道(信道估计) -> 解调。



比特-> 符号 -> 上采样 -> 脉冲成型 -> AWGN信道 -> 匹配滤波 -> 下采样 -> 解调 -> 比特

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

#%% 根升余弦滚降滤波器
# https://blog.csdn.net/dreamer_2001/article/details/130505733
def rcosdesign_srv(rolloff, L , span,):
    # 输入容错
    if rolloff == 0:
        rolloff = np.finfo(float).tiny
    # 对sps向下取整
    L = int(np.floor(L))
    if (L*L) % 2 == 1:
        raise ValueError('Invalid Input: OddFilterOrder!')
    # 初始化
    delaroll_AM = int(span*L/2)
    t = np.arange(-delaroll_AM, delaroll_AM+1)/L
    # 设计跟升余弦滤波器, 找到中点
    idx1 = np.where(t == 0)[0][0]
    rrc_filter = np.zeros_like(t)

    if idx1 is not None:
        rrc_filter[idx1] = -1.0 / (np.pi*L) * (np.pi*(rolloff-1) - 4*rolloff)
    # 查找非零的分母索引
    idx2 = np.where(np.abs(np.abs(4*rolloff*t) - 1.0) < np.sqrt(np.finfo(float).eps))[0]
    if len(idx2) > 0:
        rrc_filter[idx2] = 1.0/(2*np.pi*L) * (np.pi*(rolloff+1) * np.sin(np.pi*(rolloff+1)/(4*rolloff)) - 4*rolloff*np.sin(np.pi*(rolloff-1)/(4*rolloff)) + np.pi*(rolloff-1) * np.cos(np.pi*(rolloff-1)/(4*rolloff)))
    # 计算非零分母索引的值
    ind = np.arange(len(t))
    ind = np.delete(ind, [idx1, *idx2])
    nind = t[ind]
    rrc_filter[ind] = -4*rolloff/L * (np.cos((1+rolloff)*np.pi*nind) + np.sin((1-rolloff)*np.pi*nind) / (4*rolloff*nind)) / (np.pi * ((4*rolloff*nind)**2 - 1))
    # 能量归一化
    rrc_filter = rrc_filter / np.sqrt(sum(rrc_filter ** 2))
    filtDelay = (len(rrc_filter)-1)/2
    return rrc_filter, 0, filtDelay

# https://gist.github.com/Dreistein/8d546eab7236876882f08c0b487dad28
def rcosdesign(beta: float, L: float, span: float, shape='normal'):
    """
    %%% b = rcosdesign(beta,span,L,shape)
    %%% beta：滚降因子
    %%% L: 每个符号的采样数, L
    %%% span: 表示截断的符号范围，对滤波器取了几个Ts的长度
    %%% shape:可选择'normal'或者'sqrt'
    %%% b : 1+L*span的行向量，升余弦或余弦滤波器的系数
    在 MATLAB 的 `rcosdesign` 函数中，`span` 参数指的是滤波器脉冲响应（抽头系数）跨越的符号周期数。也就是说，`span` 决定了设计的根升余弦滤波器脉冲响应在时间上的长度。这个长度是以数据传输中的符号周期（即一个数据符号的持续时间）为单位的。
    详细来说：
    - **span**：定义了滤波器的非零脉冲响应覆盖多少个符号周期。例如，如果 `span` 为 6，那么滤波器的脉冲响应将从当前符号的中心开始，并向前后各扩展 3 个符号周期的长度。脉冲响应在这个时域区间之外将为零。
    这意味着，如果你增加 `span` 的值，滤波器的时间响应将会变长，滤波器的频域响应将会相对变得更加平坦（增加了时间长度，减少了频宽）。这可以帮助减少码间干扰（ISI），但是也导致了增加的系统延迟，并且在实际应用中会需要更多的计算资源来处理因响应扩展导致的更多样本。
    具体到 `rcosdesign` 函数的脉冲响应计算，当你提供 `span` 参数时，函数会生成一个长度为 `span * L + 1` 的滤波器脉冲响应，其中 `L` 是每个符号周期的采样点数。`span * L` 确定了响应的总采样数，而 `+1` 是因为滤波器的中心抽头被计算在内。
    理解 `span` 对滤波器设计的影响对于选择满足特定系统要求和约束的滤波器参数至关重要。例如，在一个需要较低延迟的实时系统中，你可能会选择一个较小的 `span` 值。对于一个需要很高码间干扰抑制能力的系统，你可能会选择一个较大的 `span` 值。

Raised cosine FIR filter design
    (1) Calculates square root raised cosine FIR filter coefficients with a rolloff factor of `beta`.
    (2) The filter is truncated to `span` symbols and each symbol is represented by `L` samples.
    (3) rcosdesign designs a symmetric filter. Therefore, the filter order, which is `L*span`, must be even. The filter energy is one.

    Keyword arguments:
    beta  -- rolloff factor of the filter (0 <= beta <= 1)
    span  -- number of symbols that the filter spans
    sps   -- number of samples per symbol
    shape -- `normal` to design a normal raised cosine FIR filter or `sqrt` to design a sqre root raised cosine filter
    """
    if beta < 0 or beta > 1:
        raise ValueError("parameter beta must be float between 0 and 1, got {}".format(beta))
    if span < 0:
        raise ValueError("parameter span must be positive, got {}".format(span))
    if L < 0:
        raise ValueError("parameter sps must be positive, got {}".format(L))
    if ((L*span) % 2) == 1:
        raise ValueError("rcosdesign:OddFilterOrder {}, {}".format(L, span))
    if shape != 'normal' and shape != 'sqrt':
        raise ValueError("parameter shape must be either 'normal' or 'sqrt'")
    eps = np.finfo(float).eps
    # design the raised cosine filter
    delay = span*L/2
    t = np.arange(-delay, delay)
    if len(t) % 2 == 0:
        t = np.concatenate([t, [delay]])
    t = t / L
    b = np.empty(len(t))
    if shape == 'normal':
        # design normal raised cosine filter
        # find non-zero denominator
        denom = (1-np.power(2*beta*t, 2))
        idx1 = np.nonzero(np.fabs(denom) > np.sqrt(eps))[0]
        # calculate filter response for non-zero denominator indices
        b[idx1] = np.sinc(t[idx1])*(np.cos(np.pi*beta*t[idx1])/denom[idx1])/L
        # fill in the zeros denominator indices
        idx2 = np.arange(len(t))
        idx2 = np.delete(idx2, idx1)
        b[idx2] = beta * np.sin(np.pi/(2*beta)) / (2*L)
    else:
        # design a square root raised cosine filter
        # find mid-point
        idx1 = np.nonzero(t == 0)[0]
        if len(idx1) > 0:
            b[idx1] = -1 / (np.pi*L) * (np.pi * (beta-1) - 4*beta)
        # find non-zero denominator indices
        idx2 = np.nonzero(np.fabs(np.fabs(4*beta*t) - 1) < np.sqrt(eps))[0]
        if idx2.size > 0:
            b[idx2] = 1 / (2*np.pi*L) * (np.pi * (beta+1) * np.sin(np.pi * (beta+1) / (4*beta)) - 4*beta * np.sin(np.pi * (beta-1) / (4*beta)) + np.pi*(beta-1) * np.cos(np.pi * (beta-1) / (4*beta)))
        # fill in the zeros denominator indices
        ind = np.arange(len(t))
        idx = np.unique(np.concatenate([idx1, idx2]))
        ind = np.delete(ind, idx)
        nind = t[ind]
        b[ind] = -4*beta/L * (np.cos((1+beta)*np.pi*nind) + np.sin((1-beta)*np.pi*nind) / (4*beta*nind)) / (  np.pi * (np.power(4*beta*nind, 2) - 1))
    # normalize filter energy
    b = b / np.sqrt(np.sum(np.power(b, 2)))
    filtDelay = (len(b)-1)/2
    return b, 0, filtDelay

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
    p = p / np.sqrt(np.sum(np.power(p, 2))) # both Add and Delete this line is OK.
    return p, t, filtDelay

#%%  Performance of modulations in AWGN
## 使用upfirdn函数
#---------Input Fields------------------------
nSym = 10**6 # Number of symbols to transmit
EbN0dBs = np.arange(start = -4, stop = 26, step = 2) # Eb/N0 range in dB for simulation
mod_type = 'PSK' # Set 'PSK' or 'QAM' or 'PAM' or 'FSK'
arrayOfM = [2, 4, 8, 16, 32] # array of M values to simulate
coherence = 'coherent' #'coherent'/'noncoherent'-only for FSK

# mod_type = 'QAM'
# nSym =  10**7
# arrayOfM = [ 64,  ] # uncomment this line if MOD_TYPE='QAM'

beta = 0.3
span = 8
L = 4
# p, t, filtDelay = rcosdesign_srv(beta, L, span)
p, t, filtDelay = rcosdesign(beta, L, span, shape = 'sqrt')
# p, t, filtDelay = srrcFunction(beta, L, span)

modem_dict = {'psk': PSKModem,'qam':QAMModem,'pam':PAMModem,'fsk':FSKModem}
colors = plt.cm.jet(np.linspace(0, 1, len(arrayOfM))) # colormap
fig, ax = plt.subplots(nrows = 1, ncols = 1)

for i, M in enumerate(arrayOfM):
    print(f" {M} in {arrayOfM}")
    #----- Initialization of various parameters ----
    k = np.log2(M)
    EsN0dBs = 10*np.log10(k) + EbN0dBs # EsN0dB calculation
    SER_sim = np.zeros(len(EbN0dBs)) # simulated Symbol error rates

    if mod_type.lower() == 'fsk':
        modem=modem_dict[mod_type.lower()](M, coherence)#choose modem from dictionary
    else: # for all other modulations
        modem = modem_dict[mod_type.lower()](M)#choose modem from dictionary

    for j, EsN0dB in enumerate(EsN0dBs):
        d = np.random.randint(low = 0, high = M, size = nSym) # uniform random symbols from 0 to M-1
        u = modem.modulate(d) #modulate

        ##  脉冲成型 + 上变频 -> 基带信号
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

#%%  Performance of modulations in AWGN
# 不使用upfirdn函数，手动实现
#---------Input Fields------------------------
nSym = 10**6 # Number of symbols to transmit
EbN0dBs = np.arange(start = -4, stop = 26, step = 2) # Eb/N0 range in dB for simulation
mod_type = 'PSK' # Set 'PSK' or 'QAM' or 'PAM' or 'FSK'
arrayOfM = [2, 4, 8, 16, 32] # array of M values to simulate
coherence = 'coherent' #'coherent'/'noncoherent'-only for FSK

# mod_type = 'QAM'
# nSym = 10**8
# arrayOfM = [4, 16, 64, 256] # uncomment this line if MOD_TYPE='QAM'

beta = 0.3
span = 8
L = 4
# p, t, filtDelay = srrcFunction(beta, L, span)

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

        d = np.random.randint(low=0, high = M, size = nSym) # uniform random symbols from 0 to M-1
        u = modem.modulate(d) #modulate
        ## Upper sample
        v = np.vstack((u, np.zeros((L-1, u.size))))
        v = v.T.flatten()
        ## plus shaping
        s = scipy.signal.convolve(v, p, 'full')

        ## channel
        r = awgn(s, EsN0dB, L)

        ## receiver
        ## match filter
        vCap = scipy.signal.convolve(r, p, 'full')
        ## Down sampling
        u_hat = vCap[int(2 * filtDelay) : int(vCap.size - 2*filtDelay) : L ] / L

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





















