#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:58:12 2024

@author: jack

https://blog.csdn.net/dreamer_2001/article/details/130505733
https://wenku.csdn.net/answer/1ihyhaf96f
https://blog.csdn.net/qq_42580533/article/details/106950272

https://blog.csdn.net/Insomnia_X/article/details/126324735

https://zhuanlan.zhihu.com/p/640245945

https://www.cnblogs.com/fangying7/p/4049101.html
"""




import numpy as np
import scipy
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
# from pylab import tick_params
# import copy
from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


# https://blog.csdn.net/dreamer_2001/article/details/130505733
def rcosdesign_srv(rolloff, span, sps):
    # 输入容错
    if rolloff == 0:
        rolloff = np.finfo(float).tinroll_AM

    # 对sps向下取整
    sps = int(np.floor(sps))

    if (sps*span) % 2 == 1:
        raise ValueError('Invalid Input: OddFilterOrder!')

    # 初始化
    delaroll_AM = int(span*sps/2)
    t = np.arange(-delaroll_AM, delaroll_AM+1)/sps

    # 设计跟升余弦滤波器
    # 找到中点
    idx1 = np.where(t == 0)[0][0]
    rrc_filter = np.zeros_like(t)

    if idx1 is not None:
        rrc_filter[idx1] = -1.0 / (np.pi*sps) * (np.pi*(rolloff-1) - 4*rolloff)

    # 查找非零的分母索引
    idx2 = np.where(np.abs(np.abs(4*rolloff*t) - 1.0) < np.sqrt(np.finfo(float).eps))[0]
    if len(idx2) > 0:
        rrc_filter[idx2] = 1.0/(2*np.pi*sps) * (np.pi*(rolloff+1) * np.sin(np.pi*(rolloff+1)/(4*rolloff)) - 4*rolloff*np.sin(np.pi*(rolloff-1)/(4*rolloff)) + np.pi*(rolloff-1) * np.cos(np.pi*(rolloff-1)/(4*rolloff)))

    # 计算非零分母索引的值
    ind = np.arange(len(t))
    ind = np.delete(ind, [idx1, *idx2])
    nind = t[ind]

    rrc_filter[ind] = -4*rolloff/sps * (np.cos((1+rolloff)*np.pi*nind) + np.sin((1-rolloff)*np.pi*nind) / (4*rolloff*nind)) / (np.pi * ((4*rolloff*nind)**2 - 1))

    # 能量归一化
    rrc_filter = rrc_filter / np.sqrt(sum(rrc_filter ** 2))
    return rrc_filter


# https://gist.github.com/Dreistein/8d546eab7236876882f08c0b487dad28
def rcosdesign(beta: float, span: float, sps: float, shape='normal'):
    """
    %%% b = rcosdesign(beta,span,sps,shape)
    %%% beta：滚降因子
    %%% span: 表示截断的符号范围，对滤波器取了几个Ts的长度
    %%% sps: 每个符号的采样数
    %%% shape:可选择'normal'或者'sqrt'
    %%% b:1*（sps*span）的行向量，升余弦或余弦滤波器的系数
    在 MATLAB 的 `rcosdesign` 函数中，`span` 参数指的是滤波器脉冲响应（抽头系数）跨越的符号周期数。也就是说，`span` 决定了设计的根升余弦滤波器脉冲响应在时间上的长度。这个长度是以数据传输中的符号周期（即一个数据符号的持续时间）为单位的。
    详细来说：
    - **span**：定义了滤波器的非零脉冲响应覆盖多少个符号周期。例如，如果 `span` 为 6，那么滤波器的脉冲响应将从当前符号的中心开始，并向前后各扩展 3 个符号周期的长度。脉冲响应在这个时域区间之外将为零。
    这意味着，如果你增加 `span` 的值，滤波器的时间响应将会变长，滤波器的频域响应将会相对变得更加平坦（增加了时间长度，减少了频宽）。这可以帮助减少码间干扰（ISI），但是也导致了增加的系统延迟，并且在实际应用中会需要更多的计算资源来处理因响应扩展导致的更多样本。
    具体到 `rcosdesign` 函数的脉冲响应计算，当你提供 `span` 参数时，函数会生成一个长度为 `span * sps + 1` 的滤波器脉冲响应，其中 `sps` 是每个符号周期的采样点数。`span * sps` 确定了响应的总采样数，而 `+1` 是因为滤波器的中心抽头被计算在内。
    理解 `span` 对滤波器设计的影响对于选择满足特定系统要求和约束的滤波器参数至关重要。例如，在一个需要较低延迟的实时系统中，你可能会选择一个较小的 `span` 值。对于一个需要很高码间干扰抑制能力的系统，你可能会选择一个较大的 `span` 值。


Raised cosine FIR filter design
    (1) Calculates square root raised cosine FIR filter coefficients with a rolloff factor of `beta`.
    (2) The filter is truncated to `span` symbols and each symbol is represented by `sps` samples.
    (3) rcosdesign designs a symmetric filter. Therefore, the filter order, which is `sps*span`, must be even. The filter energy is one.

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

    if sps < 0:
        raise ValueError("parameter sps must be positive, got {}".format(span))

    if ((sps*span) % 2) == 1:
        raise ValueError("rcosdesign:OddFilterOrder {}, {}".format(sps, span))

    if shape != 'normal' and shape != 'sqrt':
        raise ValueError("parameter shape must be either 'normal' or 'sqrt'")

    eps = np.finfo(float).eps

    # design the raised cosine filter
    delay = span*sps/2
    t = np.arange(-delay, delay)

    if len(t) % 2 == 0:
        t = np.concatenate([t, [delay]])
    t = t / sps
    b = np.empty(len(t))

    if shape == 'normal':
        # design normal raised cosine filter
        # find non-zero denominator
        denom = (1-np.power(2*beta*t, 2))
        idx1 = np.nonzero(np.fabs(denom) > np.sqrt(eps))[0]

        # calculate filter response for non-zero denominator indices
        b[idx1] = np.sinc(t[idx1])*(np.cos(np.pi*beta*t[idx1])/denom[idx1])/sps

        # fill in the zeros denominator indices
        idx2 = np.arange(len(t))
        idx2 = np.delete(idx2, idx1)

        b[idx2] = beta * np.sin(np.pi/(2*beta)) / (2*sps)

    else:
        # design a square root raised cosine filter
        # find mid-point
        idx1 = np.nonzero(t == 0)[0]
        if len(idx1) > 0:
            b[idx1] = -1 / (np.pi*sps) * (np.pi * (beta-1) - 4*beta)
        # find non-zero denominator indices
        idx2 = np.nonzero(np.fabs(np.fabs(4*beta*t) - 1) < np.sqrt(eps))[0]
        if idx2.size > 0:
            b[idx2] = 1 / (2*np.pi*sps) * (np.pi * (beta+1) * np.sin(np.pi * (beta+1) / (4*beta)) - 4*beta * np.sin(np.pi * (beta-1) / (4*beta)) + np.pi*(beta-1)   * np.cos(np.pi * (beta-1) / (4*beta)))

        # fill in the zeros denominator indices
        ind = np.arange(len(t))
        idx = np.unique(np.concatenate([idx1, idx2]))
        ind = np.delete(ind, idx)
        nind = t[ind]

        b[ind] = -4*beta/sps * (np.cos((1+beta)*np.pi*nind) + np.sin((1-beta)*np.pi*nind) / (4*beta*nind)) / (  np.pi * (np.power(4*beta*nind, 2) - 1))

    # normalize filter energy
    b = b / np.sqrt(np.sum(np.power(b, 2)))
    return b

beta = 0.25
span = 6
sps = 4
shape = 'sqrt'

# h =  rcosdesign_srv(beta, span, sps, )
h =  rcosdesign(beta, span, sps, shape = 'root' )
t = np.arange(h.size)
print(h)

width = 10
high = 6
horvizen = 1
vertical = 1
fig, axs = plt.subplots(1, 1, figsize = (horvizen*width, vertical*high), constrained_layout = True)
labelsize = 20

axs.plot(t, h, label = 'raw')

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs.set_xlabel('time',fontproperties=font)
axs.set_ylabel('value',fontproperties=font)
#font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
#  edgecolor='black',
# facecolor = 'y', # none设置图例legend背景透明
legend1 = axs.legend(loc='best',  prop=font1, bbox_to_anchor=(0.5, -0.2), ncol = 3,  borderaxespad=0,)
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
[label.set_fontsize(20) for label in labels] #刻度值字号

plt.show()



































