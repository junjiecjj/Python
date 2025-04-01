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

# Radar parameters setting
maxR = 200   # 雷达最大探测目标的距离
rangeRes = 1 # 雷达的距离分率
maxV = 70    # 雷达最大检测目标的速度
fc = 77e9    # 雷达工作频率 载频
c = 3e8
r0 = 100  # 目标距离
v0 = 30   # 目标速度

B = c/(2*rangeRes)          # 150MHz
Tchirp = 5.5*2*maxR/c       # 扫频时间 (x-axis), 5.5 = sweep time should be at least 5 o 6 times the round trip time
endle_time = 6.3e-6         # 空闲时间
slope = B/Tchirp            # 调频斜率
f_IFmax = (slope*2*maxR)/c  # 最高中频频率
f_IF = (slope*2*r0)/c       # 当前中频频率

Nd = 128                                 # chirp数量
Nr = 1024                                # ADC采样点数
vres = (c/fc)/(2*Nd*(Tchirp+endle_time)) # 速度分辨率
Fs = Nr/Tchirp   # = 1/(t[1] - t[0])     # 模拟信号采样频率

#Tx波函数参数
t = np.linspace(0, Nd*Tchirp, Nr*Nd)     # 发射信号和接收信号的采样时间
angle_freq = fc * t +  slope /2 * t**2   # 角频率
freq = fc + slope*t                      # 频率
Tx = np.cos(2*np.pi*angle_freq)          # 发射波形函数

r0 = r0 + v0 * t

#Rx波函数参数
td = 2*r0/c
tx = t
freqRx = fc + slope*t
Rx = np.cos(2*np.pi*(fc*(t-td) + (slope*(t-td)*(t-td))/2)) #接受波形函数

#IF信号函数参数
IF_angle_freq = fc*t+(slope*t*t)/2 - ((fc*(t-td) + (slope*(t-td)*(t-td))/2))
freqIF = slope*td
IFx = np.cos(-(2*np.pi*(fc*(t-td) + (slope*(t-td)*(t-td))/2))+(2*np.pi*angle_freq))

#Range FFT
doppler = 10*np.log10(np.abs(np.fft.fft(IFx[0:1024])))
frequency = np.fft.fftfreq(1024, 1/Fs)
range = frequency*c/(2*slope)


# 结果可视化
fig, axs = plt.subplots(3, 2, figsize = (16, 12), constrained_layout = True)

# 发射信号（实部）
axs[0,0].plot(t[0:1024], Tx[0:1024])
axs[0,0].set_title("Tx Signal")
axs[0,0].set_xlabel("Time (s)")
axs[0,0].set_ylabel("Amplitude")

# 接收回波信号（实部）
axs[0,1].plot(t[0:1024], Rx[0:1024])
axs[0,1].set_title("Rx Signal")
axs[0,1].set_xlabel("Time (s)")
axs[0,1].set_ylabel("Amplitude")

#
axs[1,0].plot(t[0:1024]+td[0:1024], freqRx[0:1024])
axs[1,0].set_title("matched_filter (Real Part)")
axs[1,0].set_xlabel("Time")
axs[1,0].set_ylabel("Frequency")

#
axs[1,1].plot(t[0:1024],IFx[0:1024])
axs[1,1].set_title("IFx Signal")
axs[1,1].set_xlabel("Time")
axs[1,1].set_ylabel("Amplitude")

#
axs[2,0].plot(np.arange(512), doppler[0:512])
axs[2,0].set_title("IF Signal FFT")
axs[2,0].set_xlabel("Frequency->Distance")
axs[2,0].set_ylabel("Amplitude")

#
axs[2,1].specgram(IFx, 1024, Fs)
axs[2,1].set_title("Spectogram")
axs[2,1].set_xlabel("Time")
axs[2,1].set_ylabel("Amplitude")

plt.show()
plt.close()


# (二)速度检测Python仿真
# 速度维数据提取
chirpamp = []
chirpnum = 1
while(chirpnum<=Nd):
    strat = (chirpnum-1)*1024
    end = chirpnum*1024
    chirpamp.append(IFx[(chirpnum-1)*1024])
    chirpnum = chirpnum + 1
#速度维做FFT得到相位差
doppler = 10*np.log10(np.abs(np.fft.fft(chirpamp)))
FFTfrequency = np.fft.fftfreq(Nd, 1/Fs)
velocity = 5*np.arange(0, Nd)/3
plt.figure(figsize=(8, 6))
plt.plot(velocity[0:int(Nd/2)], doppler[0:int(Nd/2)])
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
Data_fft2 = Z_fft2[0:512,0:512]
plt.figure(figsize = (8, 8))
plt.imshow(Data_fft2)
plt.xlabel("Range")
plt.ylabel("Velocity")
plt.title('Velocity-Range 2D FFT')

plt.tight_layout(pad = 3, w_pad = 0.05, h_pad = 0.05)
plt.show()





















