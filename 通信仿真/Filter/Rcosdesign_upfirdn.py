#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:46:04 2024

@author: jack
https://zhuanlan.zhihu.com/p/694101308

https://zhuanlan.zhihu.com/p/36711152
https://blog.csdn.net/weixin_46136963/article/details/106691783

https://blog.csdn.net/weixin_43870101/article/details/106794354

https://blog.csdn.net/lanluyug/article/details/80401943
"""

import numpy as np
import scipy

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

#%%=================================================================================
## 上采样
##==================================================================================
c = scipy.signal.upfirdn([1, 1, 1], [1, 1, 1])   # FIR filter
# array([1., 2., 3., 2., 1.])
c = np.convolve([1, 1, 1],[1, 1, 1])
# array([1, 2, 3, 2, 1])


c = scipy.signal.upfirdn([1], [1, 2, 3], 3)  # upsampling with zeros insertion
# array([ 1.,  0.,  0.,  2.,  0.,  0.,  3.])
c = np.convolve([1,0,0,2,0,0,3],[1])
# array([1, 0, 0, 2, 0, 0, 3])


c = scipy.signal.upfirdn([1, 1, 1], [1, 2, 3], 3)  # upsampling with sample-and-hold
# array([ 1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  3.])
c = np.convolve([1,0,0,2,0,0,3], [1,1,1])
# array([1, 1, 1, 2, 2, 2, 3, 3, 3])

c = scipy.signal.upfirdn([.5, 1, .5], [1, 1, 1], 2)  # linear interpolation
# array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  0.5])
c = np.convolve([1,0,1,0,1], [.5, 1, .5])
# array([0.5, 1. , 1. , 1. , 1. , 1. , 0.5])

## 从以上结果可以看出，
# scipy.signal.upfirdn(h, x, up ,  )
# 就是先用 up - 1 个0填充x的每两个元素之间，然后与h做卷积


#%%=================================================================================
## 下采样
##==================================================================================
c = scipy.signal.upfirdn([1], np.arange(10), 1, 3)  # decimation by 3
# array([ 0.,  3.,  6.,  9.])
c = np.convolve([1], [0,3,6,9])
# array([0, 3, 6, 9])

c = scipy.signal.upfirdn([1,1,1], np.arange(10), 1, 3)  # decimation by 3
# array([ 0.,  6., 15., 24.])
c = scipy.signal.upfirdn([1,1,1], np.arange(10), 1)
# array([ 0.,  1.,  3.,  6.,  9., 12., 15., 18., 21., 24., 17.,  9.])
c = np.convolve([1,1,1], np.arange(10))
# array([ 0,  1,  3,  6,  9, 12, 15, 18, 21, 24, 17,  9])

c = scipy.signal.upfirdn([1,1,1], np.arange(10), 2, 3)  # decimation by 3
# array([ 0.,  1.,  5.,  4., 11.,  7., 17.])
c = scipy.signal.upfirdn([1,1,1], np.arange(10), 2)
# array([ 0.,  0.,  1.,  1.,  3.,  2.,  5.,  3.,  7.,  4.,  9.,  5., 11.,  6., 13.,  7., 15.,  8., 17.,  9.,  9.])
c = np.convolve([1,1,1], [0,0,1,0,2,0,3,0,4,0,5,0,6,0,7,0,8,0,9])
# array([ 0,  0,  1,  1,  3,  2,  5,  3,  7,  4,  9,  5, 11,  6, 13,  7, 15,  8, 17,  9,  9])


## 从以上结果可以看出，
# scipy.signal.upfirdn(h, x, up, down,  )
# 就是先用 up - 1 个0填充x的每两个元素之间，然后与h做卷积，接着从头开始每down个位置取一个元素作为输出



c = scipy.signal.upfirdn([.5, 1, .5], np.arange(10), 2, 3)  # linear interp, rate 2/3
# array([ 0. ,  1. ,  2.5,  4. ,  5.5,  7. ,  8.5])
c = np.convolve([.5, 1, .5], [0,0,1,0,2,0,3,0,4,0,5,0,6,0,7,0,8,0,9])
# array([0. , 0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5,  6. , 6.5, 7. , 7.5, 8. , 8.5, 9. , 4.5])


###==========================================================================

## %% https://gist.github.com/Dreistein/8d546eab7236876882f08c0b487dad28
def rcosdesign(beta: float, span: float, sps: float, shape='normal'):
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

#%% ==================== 输入信号  ====================
x = 2*np.random.randint(0,2,(10,)) - 1
x = np.array([0,0,1,1,1,1,1,1,0,0])
# ==================== 设置滤波器 ====================
span = 6
sps = 4
h = rcosdesign(0.5, span, sps, 'sqrt')

# ==================== 脉冲成型 + 上变频-> 基带信号 ====================
#对输入信号进行上采样
y = scipy.signal.upfirdn(h, x, sps)

# ==================== 载波 ====================
fc = 0.1 # Hz
t = np.arange(y.size)
# s_fc = np.cos(2 * np.pi * fc * t)
s_fc = np.exp(1j * 2 * np.pi * fc * t)

y_fc = y * s_fc

# ==================== 噪声 ====================
r = y_fc + 0.01 * ( np.random.normal(size=(y.size,)) + 1j * np.random.normal(size=(y.size,)))

# ==================== 相干解调 ====================
r_coherent = r * np.exp(-1j * 2 * np.pi * fc * t)

#==================== 下采样 + 匹配滤波 -> 恢复的基带信号 ====================
z = scipy.signal.upfirdn(h, r_coherent, 1, sps)  ## 此时默认上采样为1，即不进行上采样

width = 6
high = 3
horvizen = 3
vertical = 2
fig, axs = plt.subplots(horvizen, vertical, figsize = (vertical*width, horvizen*high), constrained_layout = True)

## h
axs[0,0].stem(h,  linefmt='--' )

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
# axs[1].set_xlabel('Frequancy(Hz)',fontproperties=font)
axs[0,0].set_ylabel('h',fontproperties=font)
# axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

axs[0,0].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs[0,0].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs[0,0].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs[0,0].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs[0,0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
labels = axs[0,0].get_xticklabels() + axs[0,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(25) for label in labels] #刻度值字号
axs[0,0].grid(linestyle = '--', linewidth = 0.5, )

## x
axs[0,1].stem(x,  linefmt='--' )

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
# axs[0].set_xlabel('Time(s)',fontproperties=font)
axs[0,1].set_ylabel(r'$\mathrm{I_n}$',fontproperties=font)
# axs[0,1].set_title( 'x',  fontproperties=font )
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

axs[0,1].spines['bottom'].set_linewidth(2)    ###设置底部坐标轴的粗细
axs[0,1].spines['left'].set_linewidth(2)      ###设置左边坐标轴的粗细
axs[0,1].spines['right'].set_linewidth(2)     ###设置右边坐标轴的粗细
axs[0,1].spines['top'].set_linewidth(2)       ###设置上部坐标轴的粗细

axs[0,1].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
labels = axs[0,1].get_xticklabels() + axs[0,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(25) for label in labels] #刻度值字号
axs[0,1].grid( linestyle = '--', linewidth = 0.5, )

## y,r_coherent
axs[1,0].plot(y, label = 'y', linewidth = 2, color = 'b',  )
axs[1,0].plot(np.real(r_coherent), label = 'r', linewidth = 2, color = 'r',  )

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
# axs[1].set_xlabel('Frequancy(Hz)',fontproperties=font)
axs[1,0].set_ylabel('s(t)',fontproperties=font)
# axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

legend1 = axs[1,0].legend(loc='best',  prop=font1, borderaxespad=0, facecolor = 'none')
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,0].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs[1,0].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs[1,0].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs[1,0].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs[1,0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
labels = axs[1,0].get_xticklabels() + axs[1,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(25) for label in labels] #刻度值字号
axs[1,0].grid(linestyle = '--', linewidth = 0.5, )

##  s_fc
axs[1,1].plot(np.real(s_fc), label = 'f_fc', linewidth = 2, color = 'b', )
font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
axs[1,1].set_ylabel("s_fc",fontproperties=font)
# axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

legend1 = axs[1,1].legend(loc='best',  prop=font1, borderaxespad=0, facecolor = 'none')
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs[1,1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs[1,1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs[1,1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs[1,1].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
labels = axs[1,1].get_xticklabels() + axs[1,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(25) for label in labels] #刻度值字号
axs[1,1].grid(linestyle = '--', linewidth = 0.5, )


##  RF
axs[2,0].plot(np.real(y_fc), label = 'RF,Tx', linewidth = 2, color = 'b', )
axs[2,0].plot(np.real(r), label = 'RF, Rx', linewidth = 2, color = 'r', )
font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
axs[2,0].set_ylabel("RF",fontproperties=font)
# axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

legend1 = axs[2,0].legend(loc='best',  prop=font1, borderaxespad=0, facecolor = 'none')
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[2,0].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs[2,0].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs[2,0].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs[2,0].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs[2,0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
labels = axs[2,0].get_xticklabels() + axs[2,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(25) for label in labels] #刻度值字号
axs[2,0].grid(linestyle = '--', linewidth = 0.5, )

## hat I_n
axs[2,1].stem(z,  linefmt='c--' )
font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
# axs[1].set_xlabel('Frequancy(Hz)',fontproperties=font)
axs[2,1].set_ylabel(r'$\hat{\mathrm{I}_n}$',fontproperties=font)
# axs.set_title(f'{Modulation_type}, Eb/N0 = {ebn0[-1]}(dB)',  fontproperties=font )
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

axs[2,1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs[2,1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs[2,1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs[2,1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs[2,1].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, pad = 2 )
labels = axs[2,1].get_xticklabels() + axs[2,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(25) for label in labels] #刻度值字号
axs[2,1].grid(linestyle = '--', linewidth = 0.5, )


out_fig = plt.gcf()
# out_fig.savefig(f'upfirdn.eps', )
# out_fig.savefig(f'upfirdn.png', dpi = 1000,)
plt.show()

































