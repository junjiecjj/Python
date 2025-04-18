#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 23:44:57 2025

@author: jack
基于Python的FMCW雷达工作原理仿真

https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247485426&idx=1&sn=ad1d302e2177b037778ee9e6d405ec33&chksm=c11f0a67f66883717c39bd6deab5a184182dec6192c517967d68b85b25cafe286607cfee1f7d&scene=21#wechat_redirect
"""


import numpy as np
import scipy
import matplotlib.pyplot as plt
# from numpy import fft
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D

def freqDomainView(x, Fs, FFTN = None, type = 'double'): # N为偶数
    if FFTN == None:
        FFTN = 2**int(np.ceil(np.log2(x.size)))
    X = scipy.fftpack.fft(x, n = FFTN)
    # 消除相位混乱
    threshold = np.max(np.abs(X)) / 10000
    X[np.abs(X) < threshold] = 0
    # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
    X = X/x.size               # 将频域序列 X 除以序列的长度 N
    if type == 'single':
        Y = X[0 : int(FFTN/2)+1].copy()       # 提取 X 里正频率的部分,N为偶数
        Y[1 : int(FFTN/2)] = 2*Y[1 : int(FFTN/2)].copy()
        f = np.arange(0, int(FFTN/2)+1) * (Fs/FFTN)
        # 计算频域序列 Y 的幅值和相角
        A = np.abs(Y)                     # 计算频域序列 Y 的幅值
        Pha = np.angle(Y, deg=1)          # 计算频域序列 Y 的相角 (弧度制)
        R = np.real(Y)                    # 计算频域序列 Y 的实部
        I = np.imag(Y)                    # 计算频域序列 Y 的虚部
    elif type == 'double':
        f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/Fs))
        Y = scipy.fftpack.fftshift(X, )
        # 计算频域序列 Y 的幅值和相角
        A = np.abs(Y)                     # 计算频域序列 Y 的幅值
        Pha = np.angle(Y, deg=1)          # 计算频域序列 Y 的相角 (弧度制)
        R = np.real(Y)                    # 计算频域序列 Y 的实部
        I = np.imag(Y)                    # 计算频域序列 Y 的虚部
    return f, Y, A, Pha, R, I

# Radar parameters setting
maxR = 200   # 雷达最大探测目标的距离
rangeRes = 1 # 雷达的距离分率
maxV = 70    # 雷达最大检测目标的速度
fc = 77e9    # 雷达工作频率 载频
c = 3e8
R0 = 100  # 目标距离
v0 = 50   # 目标速度

B = c/(2*rangeRes)          # 150MHz
Tchirp = 5.5*2*maxR/c       # 扫频时间 (x-axis), 5.5 = sweep time should be at least 5 o 6 times the round trip time
endle_time = 6.3e-6         # 空闲时间
slope = B/Tchirp            # 调频斜率
f_IFmax = (slope*2*maxR)/c  # 最高中频频率
f_IF = (slope*2*R0)/c       # 当前中频频率

Nchirp = 128                                  # chirp数量
Ns = 1024                                     # ADC采样点数
vres = (c/fc)/(2*Nchirp*(Tchirp+endle_time))  # 速度分辨率
Fs = Ns/Tchirp   # = 1/(t[1] - t[0])          # 模拟信号采样频率

# Tx波函数参数
t = np.linspace(0, Nchirp*Tchirp, Nchirp*Ns)   # 发射信号和接收信号的采样时间
# angle_freq = fc * t +  slope / 2 * t**2        # 角频率
freqTx = fc + slope*t                          # 频率
Tx = np.cos(2*np.pi*(fc * t +  slope / 2 * t**2))                # 发射波形函数

r0 = R0 + v0 * t

# Rx波函数参数
td = 2*r0/c
# tx = t
freqRx = fc + slope*t
Rx = np.cos(2*np.pi*(fc*(t-td) + (slope*(t-td)*(t-td))/2)) # 接受波形函数

Mix = Tx * Rx
f, _, A, _, _, _  = freqDomainView(Mix[:Ns], Fs, type = 'double')
fig, axs = plt.subplots(1, 1, figsize = (4, 3), constrained_layout = True)
axs.plot(f, A)
axs.set_title("Mix Signal FFT")
axs.set_xlabel("Frequency")
axs.set_ylabel("Amplitude")
plt.show()
plt.close()

### IF信号函数参数 1
# IF_angle_freq = fc*t+(slope*t*t)/2 - ((fc*(t-td) + (slope*(t-td)*(t-td))/2))
# freqIF = slope*td
IFx = np.cos(2*np.pi*(fc * t +  slope / 2 * t**2) - 2*np.pi*(fc*(t-td) + slope/2*(t-td)**2))

### IF信号函数参数 2
# order = 6
# Wn = slope * 2 * R0 * 3 / c / Fs
# [Bb, Ba] = scipy.signal.butter(order, Wn, 'low')
# IFx = scipy.signal.filtfilt(Bb, Ba, Mix )

f, _, A, _, _, _  = freqDomainView(IFx[:Ns], Fs, type = 'double')
fig, axs = plt.subplots(1, 1, figsize = (4, 3), constrained_layout = True)
axs.plot(f, A)
axs.set_title("IF Signal FFT")
axs.set_xlabel("Frequency")
axs.set_ylabel("Amplitude")
plt.show()
plt.close()

# Range FFT
doppler = 10*np.log10(np.abs(np.fft.fft(IFx[0:Ns])))
frequency = np.fft.fftfreq(Ns, 1/Fs)
Range = frequency*c/(2*slope)  # f_{if} = 2*slope*d/c
# Range = np.arange(Ns) / Ns * Ns * d_res;
# 结果可视化
fig, axs = plt.subplots(3, 2, figsize = (12, 12), constrained_layout = True)

# 发射回波信号
axs[0,0].plot(t[0:Ns], Tx[0:Ns])
axs[0,0].set_title("Tx Signal")
axs[0,0].set_xlabel("Time (s)")
axs[0,0].set_ylabel("Amplitude")

# 接收回波信号
axs[0,1].plot(t[0:Ns], Rx[0:Ns])
axs[0,1].set_title("Rx Signal")
axs[0,1].set_xlabel("Time (s)")
axs[0,1].set_ylabel("Amplitude")

axs[1,0].plot(t[0:Ns], freqTx[0:Ns], label = "Frequency of Tx signal")
axs[1,0].plot(t[0:Ns] + td[0:Ns], freqRx[0:Ns], label = "Frequency of Rx signal")
axs[1,0].set_title("Frequency of Tx/Rx signal")
axs[1,0].set_xlabel("Time")
axs[1,0].set_ylabel("Frequency")
axs[1,0].legend()

axs[1,1].plot(t[0:Ns], IFx[0:Ns])
axs[1,1].set_title("IFx Signal")
axs[1,1].set_xlabel("Time")
axs[1,1].set_ylabel("Amplitude")

axs[2,0].plot(Range[0:int(Ns/2)], doppler[0:int(Ns/2)])
axs[2,0].set_title("IF Signal FFT")
axs[2,0].set_xlabel("Frequency->Distance")
axs[2,0].set_ylabel("Amplitude")

axs[2,1].specgram(IFx, NFFT = Ns, Fs = Fs)
axs[2,1].set_title("Spectogram")
axs[2,1].set_xlabel("Time")
axs[2,1].set_ylabel("Frequency")

plt.show()
plt.close()

### (二)速度检测Python仿真
# 步骤1：速维度数据提取
# 每个啁啾提取一个采样点，对于具有 128 个啁啾的帧，将有 128 个点的列表。
chirpamp = []
chirpnum = 1
while(chirpnum <= Nchirp):
    strat = (chirpnum-1)*Ns
    end = chirpnum*Ns
    chirpamp.append(IFx[(chirpnum-1)*Ns + 0])
    chirpnum = chirpnum + 1
# 步骤2：相位差和速度的速度维度 FFT
doppler1 = 10*np.log10(np.abs(np.fft.fft(chirpamp)))
FFTfrequency = np.fft.fftfreq(Nchirp, 1/Fs)
velocity = 5*np.arange(0, Nchirp)/3

# 2D FFT 和速度-距离关系
mat2D = np.zeros((Nchirp, Ns))
i = 0
while(i < Nchirp):
    mat2D[i, :] = IFx[i*Ns : (i+1)*Ns]
    i = i + 1

#图形绘制
Z_fft2 = np.abs(np.fft.fft2(mat2D))
Data_fft2 = Z_fft2[0:int(Ns/2), 0:int(Ns/2)]

# 结果可视化
fig, axs = plt.subplots(2, 1, figsize = (6, 8), constrained_layout = True)

# 发射回波信号
axs[0].plot(velocity[0:int(Nchirp/2)], doppler1[0:int(Nchirp/2)])
axs[0].set_title("IF Velocity FFT")
axs[0].set_xlabel("Velocity")
axs[0].set_ylabel("Amplitude")

# 接收回波信号
axs[1].imshow(Data_fft2)
axs[1].set_title("Velocity-Range 2D FFT")
axs[1].set_xlabel("Range")
axs[1].set_ylabel("Velocity")

plt.show()
plt.close()

#%% Usage of specgram
# import matplotlib.pyplot as plt
# import numpy as np

# # Fixing random state for reproducibility
# np.random.seed(42)

# dt = 0.0005
# t = np.arange(0.0, 20.5, dt)
# s1 = np.sin(2 * np.pi * 100 * t)
# s2 = 2 * np.sin(2 * np.pi * 400 * t)

# # create a transient "chirp"
# s2[t <= 10] = s2[12 <= t] = 0

# # add some noise into the mix
# nse = 0.01 * np.random.random(size = len(t))

# x = s1 + s2 + nse  # the signal
# NFFT = 1024        # the length of the windowing segments
# Fs = 1/dt          # the sampling frequency

# fig, (ax1, ax2) = plt.subplots(nrows = 2, sharex = True)
# ax1.plot(t, x)
# ax1.set_ylabel('Signal')

# Pxx, freqs, bins, im = ax2.specgram(x, NFFT = NFFT, Fs = Fs)
# # The `specgram` method returns 4 objects. They are:
# # - Pxx: the periodogram
# # - freqs: the frequency vector
# # - bins: the centers of the time bins
# # - im: the .image.AxesImage instance representing the data in the plot
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('Frequency (Hz)')
# ax2.set_xlim(0, 20)

# plt.show()


#%%


#%%



#%%


#%%












