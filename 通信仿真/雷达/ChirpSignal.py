#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 02:40:34 2025

@author: jack

频率随时间变化的信号称为“线性调频脉冲(Chirp)” ，线性调频信号的频率可以从低频到高频（上调频），也可以从高频到低频（低调频）变化。线性调频信号在许多应用中都会遇到，包括雷达、声纳、扩频、光通信、图像处理[1] 等 。
例如，在汽车雷达中，大部分雷达厂商采用上调频信号，而有的厂商采用下调频信号（德国大陆）。

"""


import numpy as np
import scipy
import matplotlib.pyplot as plt
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22


#%%
def chirp_signal(t, f0, t1, f1, phase = 0):
    t0 = t[0]
    T = t1 - t0
    k = (f1-f0)/T
    g = np.cos(2 * np.pi * (k/2 * t + f0)*t + phase)
    return g

fs = 500
t = np.arange(0, 1, 1/fs)
f0 = 1
f1 = fs/20
g = chirp_signal(t, f0, 1, f1)
fg0, ax0 = plt.subplots()
ax0.plot(t, g)
ax0.set_title(r"Linear Chirp from $f(0)=1\,$Hz to $f(1)=25\,$Hz")
ax0.set_xlabel( "Time $t$ in Seconds", )
ax0.set_ylabel( r"Amplitude $x_{lin}(t)$")
plt.show()
plt.close()


#%%
fs = 500
t = np.arange(0, 1 + 1/fs, 1/fs)
f0 = 1
f1 = fs/20
x_lin = scipy.signal.chirp(t, f0 = f0, f1 = f1, t1 = 1, method='linear')
fg0, ax0 = plt.subplots()
ax0.plot(t, x_lin)
ax0.set_title(f"Linear Chirp from {f0} Hz to {f1} Hz")
ax0.set_xlabel("Time $t$ in Seconds", )
ax0.set_ylabel(r"Amplitude $x_{lin}(t)$")

plt.show()
plt.close()

#%% 线性调频信号的FFT和PSD仿真

fs = 1000
t = np.arange(0, 1 + 1/fs, 1/fs)
f0 = 1
f1 = fs/20
x_lin = scipy.signal.chirp(t, f0 = f0, f1 = f1, t1 = 1, method = 'linear')
x_lin = chirp_signal(t, f0 = f0, t1 = 1, f1 = f1,  )

Nfft = 1024
df = fs/Nfft
X = scipy.fftpack.fft(x_lin, n = Nfft)
X[np.abs(X) < 1e-8] = 0
# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
X = X/Nfft               # 将频域序列 X 除以序列的长度 N

# 半谱图
f = np.arange(0, int(Nfft/2)+1)*df
Y = X[0 : int(Nfft/2)+1].copy()                 # 提取 X 里正频率的部分,N为偶数
Y[1 : int(Nfft/2)] = 2*Y[1 : int(Nfft/2)].copy()
A = np.abs(Y)
# 单边带-功率谱密度
X1 = X[0 : int(Nfft/2)+1]
Pxx = X1*X1.conjugate()/(Nfft**2)

# 全谱图
f1 = np.arange(-int(Nfft/2), int(Nfft/2))*df
Y1 = scipy.fftpack.fftshift(X, )
A1 = np.abs(Y1)
# 双边带-功率谱密度
Pxx1 = Y1*Y1.conjugate()/(Nfft**2)

fig, axs = plt.subplots(2, 3, figsize = (18, 8), constrained_layout = True)
axs[0,0].plot(t, x_lin)
axs[0,0].set_title( "线性调频信号")
axs[0,0].set_xlabel("Time(s)", )
axs[0,0].set_ylabel(r"Amplitude $")

axs[0,1].plot(f, A)
axs[0,1].set_title( "单边带FFT")
axs[0,1].set_xlabel("频率(Hz)", )
axs[0,1].set_ylabel( "Amplitude ")
axs[0,1].set_xlim(0, 100)

axs[0,2].plot(f1, A1)
axs[0,2].set_title( "双边带FFT")
axs[0,2].set_xlabel("频率(Hz)", )
axs[0,2].set_ylabel( "Amplitude ")
axs[0,2].set_xlim(-100, 100)

axs[1,0].plot(f1, np.real(Pxx1))
axs[1,0].set_title( "双边带功率谱密度")
axs[1,0].set_xlabel("频率(Hz)", )
axs[1,0].set_ylabel( "PSD dB/Hz")
axs[1,0].set_xlim(-50, 50)

axs[1,1].plot(f, np.real(Pxx))
axs[1,1].set_title( "单边带功率谱密度")
axs[1,1].set_xlabel("频率(Hz)", )
axs[1,1].set_ylabel( "PSD dB/Hz")
axs[1,1].set_xlim(0, 100)

axs[1,2].plot(f1, np.real(Pxx1))
axs[1,2].set_title( "双边带功率谱密度")
axs[1,2].set_xlabel("频率(Hz)", )
axs[1,2].set_ylabel( "PSD dB/Hz")
axs[1,2].set_xlim(-100, 100)

plt.show()
plt.close('all')

#%% 复指数 线性调频信号的FFT和PSD仿真

# f0 = 10e6
# T = 10e-6
# B = 25e6
# K = B/T
# fs = 2*(B+f0)
# Ts = 1/fs
# N = int(T/Ts)
# # t = np.linspace(-T/2, T/2, N)
# t = np.arange(-T/2, T/2, Ts )
# x_lin = np.exp(1j * 2 * np.pi * (f0 * t + K/2 * t**2))

c = 3e8      # 光速
## FMCW 波形参数
f0 = 77e9    # Carrier Frequency (Hz)
B = 4e9      # Bandwidth (Hz)
Tc = 40e-6
K = B/Tc
fs = 3 * (B + f0) # 采样率一定要足,很多博客为了画图时不是一坨,采样率都是错的,用了很低的采用率。
Ts = 1/fs
# N = 128
t = np.arange(0, Tc-Ts, Ts)  # 生成时间向量
x_lin = np.exp(1j * 2 * np.pi * (f0*t + K/2 * t**2))

Nfft = t.size if t.size % 2 == 0 else t.size + 1
df = fs/Nfft
X = scipy.fftpack.fft(x_lin, n = Nfft)
X[np.abs(X) < 1e-8] = 0
# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
X = X/Nfft               # 将频域序列 X 除以序列的长度 N

# 半谱图
f = np.arange(0, int(Nfft/2)+1)*df
Y = X[0 : int(Nfft/2)+1].copy()                 # 提取 X 里正频率的部分,N为偶数
Y[1 : int(Nfft/2)] = 2*Y[1 : int(Nfft/2)].copy()
A = np.abs(Y)
# 单边带-功率谱密度
X1 = X[0 : int(Nfft/2)+1]
Pxx = X1*X1.conjugate()/(Nfft**2)

# 全谱图
f1 = np.arange(-int(Nfft/2), int(Nfft/2))*df
Y1 = scipy.fftpack.fftshift(X,)
A1 = np.abs(Y1)
# 双边带-功率谱密度
Pxx1 = Y1*Y1.conjugate()/(Nfft**2)

fig, axs = plt.subplots(2, 3, figsize = (18, 8), constrained_layout = True)
axs[0,0].plot(t, np.real(x_lin))
axs[0,0].set_title( "线性调频信号")
axs[0,0].set_xlabel("Time(s)", )
axs[0,0].set_ylabel(r"Amplitude $")

axs[0,1].plot(f, A)
axs[0,1].set_title( "单边带FFT")
axs[0,1].set_xlabel("频率(Hz)", )
axs[0,1].set_ylabel( "Amplitude ")
# axs[0,1].set_xlim(0, 100)

axs[0,2].plot(f1, A1)
axs[0,2].set_title( "双边带FFT")
axs[0,2].set_xlabel("频率(Hz)", )
axs[0,2].set_ylabel( "Amplitude ")
# axs[0,2].set_xlim(-100, 100)

axs[1,0].plot(f1, np.real(Pxx1))
axs[1,0].set_title( "双边带功率谱密度")
axs[1,0].set_xlabel("频率(Hz)", )
axs[1,0].set_ylabel( "PSD dB/Hz")
# axs[1,0].set_xlim(-50, 50)

axs[1,1].plot(f, np.real(Pxx))
axs[1,1].set_title( "单边带功率谱密度")
axs[1,1].set_xlabel("频率(Hz)", )
axs[1,1].set_ylabel( "PSD dB/Hz")
# axs[1,1].set_xlim(0, 100)

axs[1,2].plot(f1, np.real(Pxx1))
axs[1,2].set_title( "双边带功率谱密度")
axs[1,2].set_xlabel("频率(Hz)", )
axs[1,2].set_ylabel( "PSD dB/Hz")
# axs[1,2].set_xlim(-100, 100)

plt.show()
plt.close('all')



#%%
###  1
c = 3e8  # 光速
# FMCW 波形参数
f0 = 77e9    # Carrier Frequency (Hz)
B = 4e9     # Bandwidth (Hz)
Tc = 40e-6
K = B/Tc
fs = 4 * (B + f0) # 采样率一定要足，很多博客为了画图时不是一坨，采样率都是错的,用了很低的采用率。
Ts = 1/fs
N = 128
t = np.arange(0, Tc, Ts)  # 生成时间向量

# 初始化
st = np.cos(2 * np.pi * f0 * t + np.pi * K * (t ** 2))
ft = f0 + K * t


##### plot
fig, axs = plt.subplots(2, 1, figsize = (6, 8), constrained_layout = True)

##
axs[0].plot(t, st, color = 'b', label = '')
axs[0].set_xlabel('时间/s',)
axs[0].set_ylabel('幅值',)
# axs[0].legend()

axs[1].plot(t, ft/10e9, color = 'b', label = '')
axs[1].set_xlabel('时间/s',)
axs[1].set_ylabel('Frequency/GHz',)
axs[1].set_title("Frequency Variation of FMCW Signal at Transmitter")

plt.show()
plt.close()


###  2
f0 = 1
fs = 500
f1 = fs / 20
t1 = 1
t = np.arange(0, 1, 1/fs)
T = (t1 - t[0])
K = (f1 - f0) / T

# 初始化
st = np.cos(2 * np.pi * f0 * t + np.pi * K * (t ** 2))
ft = f0 + K * t

##### plot
fig, axs = plt.subplots(2, 1, figsize = (6, 8), constrained_layout = True)

axs[0].plot(t, st, color = 'b', label = '')
axs[0].set_xlabel('时间/s',)
axs[0].set_ylabel('幅值',)
# axs[0].legend()

axs[1].plot(t, ft, color = 'b', label = '')
axs[1].set_xlabel('时间/s',)
axs[1].set_ylabel('Frequency/Hz',)
axs[1].set_title("Frequency Variation of FMCW Signal at Transmitter")

plt.show()
plt.close()



































