#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:40:32 2024

@author: jack

https://www.delftstack.com/zh/howto/python/low-pass-filter-python/


"""


import scipy
from scipy.signal import butter
from scipy.signal import freqz
from scipy.signal import lfilter
# from scipy.signal import hanning
from scipy.signal import firls
from scipy.signal import firwin
from matplotlib.pyplot import stem
import numpy as np


import matplotlib.pyplot as plt
import numpy.matlib

#%% 信号和采样率
Fs = 4000 # 采样频率4000Hz
t = np.arange(0, 1, 1/Fs)

c50 = np.cos(2 * np.pi * 50 * t) # 产生50Hz余弦波
c1000 = np.cos(2 * np.pi * 1000 * t) # 产生1000Hz余弦波
s = c50 + c1000  # 信号叠加


#%% ------------------------IIR模拟低通滤波器设计---------------------------
Fp = 200 # 通带截止频率200Hz
Fc = 1000 # 阻带截止频率1000Hz
Rp = 1 # 通带波纹最大衰减为1dB
Rs = 40 # 阻带衰减为40dB
#-----------------------计算最小滤波器阶数-----------------------------
na = np.sqrt(10**(0.1*Rp)-1)
ea = np.sqrt(10**(0.1*Rs)-1)
N = np.ceil(np.log10(ea/na)/np.log10(Fc/Fp))   #巴特沃兹阶数
#----------------------巴特沃兹低通滤波器------------------------------
Wn = Fp*2/Fs
[Bb, Ba] = scipy.signal.butter(N, Wn, 'low')
[BW, BH] = scipy.signal.freqz(Bb, Ba)

fig2 = plt.figure(2)
A1 = BW*Fs/(np.pi)
plt.plot(A1,abs(BH))
plt.grid()
plt.xlabel('频率')
plt.ylabel('频响')
plt.title('巴特沃兹低通滤波器')

y = scipy.signal.lfilter(Bb,Ba,s) # 进行滤波
# By = np.fft.fft(Bf,lens)  # 对滤波输出信号做FFT变换

#-----------------------确定相应的FIR数字滤波器指标-------------------------
wp = 2 * np.pi*1000/Fs  #通带截止频率
# ws=2*pi*50/Fs
ws = np.pi/4      #阻带截止频率
Bt = wp-ws     #过渡带带宽
N0 = np.ceil(6.2*np.pi/Bt)  #汉宁窗窗长
N = N0+(N0+1)%2    #由于是高通滤波器，所以窗长N必须为奇数
wc = (wp+ws)/(2*np.pi)   #计算理想高通滤波器通带截止频率（关于pi归一化）
#----------------------FIR数字高通滤波器(汉宁窗)-------------------------
#hn=firls(N-1,wc,'high',hanning(N))
hn = firwin(int(N), wc, window = 'hamming',pass_zero = 'highpass')

[BW,BH] = freqz(hn,1) # 绘制频率响应曲线

fig4=plt.figure(4)  # 画图
A1 = BW*Fs/(2*np.pi)
plt.plot(A1,abs(BH))
plt.grid()
plt.xlabel('频率')
plt.ylabel('频响')
plt.title('汉宁窗设置高通FIR数字滤波器')

Bf=lfilter(hn,1,s) # 进行高通滤波
By = np.fft.fft(Bf,lens)  # 对高通滤波输出信号做len点FFT变换

fig4=plt.figure(5)  # 画图
fig4.tight_layout()

plt.subplot(3,1,1)
plt.plot(t*Fs,c1000,'blue')
plt.grid
plt.axis([0,400,-1,1])
plt.title('原余弦信号')
plt.subplot(3,1,2)
plt.plot(t*Fs,Bf,'red')
plt.grid
plt.axis([0,400,-1,1])
plt.title('高通滤波后')
plt.subplot(3,1,3)
plt.plot(f,abs(By))
plt.grid
plt.title('高通FIR滤波频谱')
plt.xlabel('频率')
plt.ylabel('幅值')

