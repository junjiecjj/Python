#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 23:44:57 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247485426&idx=1&sn=ad1d302e2177b037778ee9e6d405ec33&chksm=c11f0a67f66883717c39bd6deab5a184182dec6192c517967d68b85b25cafe286607cfee1f7d&scene=21#wechat_redirect
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
from mpl_toolkits.mplot3d import Axes3D

#Radar parameters setting
maxR = 200
rangeRes = 1
maxV = 70
fc = 77e9
c = 3e8
r0 = 100
v0 = 70

B = c/(2*rangeRes)
Tchirp = 5.5*2*maxR/c
endle_time = 6.3e-6
slope = B/Tchirp
f_IFmax = (slope*2*maxR)/c
f_IF = (slope*2*r0)/c

Nd = 128
Nr = 1024
vres = (c/fc)/(2*Nd*(Tchirp+endle_time))
Fs = Nr/Tchirp
#Tx = np.zeros(1,len(t))
#Rx = np.zeros(1,len(t))
#Mix = np.zeros(1,len(t))

#Tx波函数参数
t = np.linspace(0,Nd*Tchirp,Nr*Nd) #发射信号和接收信号的采样时间
angle_freq = fc*t+(slope*t*t)/2 #角频率
freq = fc + slope*t #频率
Tx = np.cos(2*np.pi*angle_freq) #发射波形函数

plt.figure(figsize=(12, 12))
plt.subplot(4,2,1)
plt.plot(t[0:1024],Tx[0:1024])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Tx Signal')
plt.subplot(4,2,3)
plt.plot(t[0:1024],freq[0:1024])
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Tx F-T')

r0 = r0+v0*t

#Rx波函数参数
td = 2*r0/c
tx = t
freqRx = fc + slope*(t)
Rx = np.cos(2*np.pi*(fc*(t-td) + (slope*(t-td)*(t-td))/2)) #接受波形函数
plt.subplot(4,2,2)
plt.plot(t[0:1024],Rx[0:1024])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Rx Signal')

plt.subplot(4,2,3)
plt.plot(t[0:1024]+td[0:1024],freqRx[0:1024])
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Chirp F-T')

#IF信号函数参数
IF_angle_freq = fc*t+(slope*t*t)/2 - ((fc*(t-td) + (slope*(t-td)*(t-td))/2))
freqIF = slope*td
IFx = np.cos(-(2*np.pi*(fc*(t-td) + (slope*(t-td)*(t-td))/2))+(2*np.pi*angle_freq))

plt.subplot(4,2,4)
plt.plot(t[0:1024],IFx[0:1024])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('IFx Signal')

#Range FFT
doppler = 10*np.log10(np.abs(np.fft.fft(IFx[0:1024])))
frequency = np.fft.fftfreq(1024, 1/Fs)
range = frequency*c/(2*slope)

plt.subplot(4,2,5)
plt.plot(range[0:512],doppler[0:512])
plt.xlabel('Frequency->Distance')
plt.ylabel('Amplitude')
plt.title('IF Signal FFT')

#2D plot
plt.subplot(4,2,6)
plt.specgram(IFx,1024,Fs)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectogram')

plt.tight_layout(pad=3, w_pad=0.05, h_pad=0.05)
plt.show()

# (二)速度检测Python仿真

#Speed Calculate
#速度维数据提取
chirpamp = []
chirpnum = 1
while(chirpnum<=Nd):
    strat = (chirpnum-1)*1024
    end = chirpnum*1024
    chirpamp.append(IFx[(chirpnum-1)*1024])
    chirpnum = chirpnum + 1
#速度维做FFT得到相位差
doppler = 10*np.log10(np.abs(np.fft.fft(chirpamp)))
FFTfrequency = np.fft.fftfreq(Nd,1/Fs)
velocity = 5*np.arange(0,Nd)/3
plt.figure(figsize=(8, 6))
plt.plot(velocity[0:int(Nd/2)],doppler[0:int(Nd/2)])
plt.xlabel('Velocity')
plt.ylabel('Amplitude')
plt.title('IF Velocity FFT')
plt.show()

#2D plot
mat2D = np.zeros((Nd, Nr))
i = 0
while(i<Nd):
    mat2D[i, :] = IFx[i*1024:(i+1)*1024]
    i = i + 1
#plt.matshow(mat2D)
#plt.title('Original data')
#图形绘制，需要修改一下才好看，这里就留个大家自行修改了。
Z_fft2 = abs(np.fft.fft2(mat2D))
Data_fft2 = Z_fft2[0:64,0:512]
plt.figure(figsize=(8, 6))
plt.imshow(Data_fft2)
plt.xlabel("Range")
plt.ylabel("Velocity")
plt.title('Velocity-Range 2D FFT')

plt.tight_layout(pad=3, w_pad=0.05, h_pad=0.05)
plt.show()





















