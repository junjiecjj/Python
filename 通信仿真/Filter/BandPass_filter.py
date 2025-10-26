#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:40:32 2024

@author: jack
https://www.delftstack.com/zh/howto/python/low-pass-filter-python/
"""


import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)

#%% 信号和采样率
Fs = 3000 # 采样频率4000Hz
t = np.arange(0, 0.1, 1/Fs)

c50 = np.cos(2 * np.pi * 50 * t)     # 产生50Hz余弦波
c500 = np.cos(2 * np.pi * 500 * t)   # 产生1000Hz余弦波
c1000 = np.cos(2 * np.pi * 1000 * t) # 产生1000Hz余弦波

x = 3*c50 + 2*c1000 + 1*c500 # 信号叠加
N_smaple = x.size
#----------------------- 原始信号的FFT变换 -----------------------------
FFTN = 2000        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
X = scipy.fftpack.fft(x, n = FFTN)

# 消除相位混乱
X[np.abs(X) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零
# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
X = X/N_smaple               # 将频域序列 X 除以序列的长度 N

# 提取 X 里正频率的部分, 并且将 X 里负频率的部分合并到正频率
if FFTN%2 == 0:
     X_hat = X[0 : int(FFTN/2)+1]                 # 提取 X 里正频率的部分,N为偶数
     X_hat[1 : int(FFTN/2)] = 2*X_hat[1 : int(FFTN/2)]   # 将 X 里负频率的部分合并到正频率,N为偶数

# 计算频域序列 Y 的幅值和相角
AX = abs(X_hat)                        # 计算频域序列 Y 的幅值
PhaX = np.angle(X_hat,deg=True)        # 计算频域序列 Y 的相角 (弧度制)
RX = np.real(X_hat)                    # 计算频域序列 Y 的实部
IX = np.imag(X_hat)                    # 计算频域序列 Y 的虚部

#  定义序列 Y 对应的频率刻度
df = Fs/FFTN                         # 频率间隔
if FFTN%2 == 0:
     fx = np.arange(0, int(FFTN/2)+1)*df      # 频率刻度,N为偶数

#%%================================ IIR -巴特沃兹带通 滤波器  =====================================
lf = 200    # 通带截止频率200Hz
hf = 700
Fc = 1000   # 阻带截止频率1000Hz
Rp = 1      # 通带波纹最大衰减为1dB
Rs = 40     # 阻带衰减为40dB
#-----------------------计算最小滤波器阶数-----------------------------
na = np.sqrt(10**(0.1*Rp)-1)
ea = np.sqrt(10**(0.1*Rs)-1)
order  = np.ceil(np.log10(ea/na)/np.log10(Fc/lf))  #巴特沃兹阶数
f1 = lf*2/Fs
f2 = hf*2/Fs
#---------------------- 带通滤波  -----------------------------
##方法1
[Bb, Ba] = scipy.signal.butter(order, [f1, f2], 'bandpass')
# ## [BW, BH] = scipy.signal.freqz(Bb, Ba)
# y = scipy.signal.lfilter(Bb, Ba, x) # 进行滤波
y = scipy.signal.filtfilt(Bb, Ba, x )

# ###方法2
# h = scipy.signal.firwin(int(31), [lf, hf], fs = Fs, pass_zero = "bandpass" )
# y = scipy.signal.lfilter(h, 1, x) # 进行滤波

# ###方法3
# h = scipy.signal.firwin(int(31), [f1, f2], pass_zero = "bandpass" )
# y = scipy.signal.lfilter(h, 1, x) # 进行滤波

#----------------------- 滤波后信号的FFT变换 -----------------------------
Y = scipy.fftpack.fft(y, n = FFTN)
N = Y.size
# 消除相位混乱
Y[np.abs(Y) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零
# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
Y = Y/N_smaple               # 将频域序列 X 除以序列的长度 N

# 提取 X 里正频率的部分, 并且将 X 里负频率的部分合并到正频率
if FFTN%2 == 0:
     Y_hat = Y[0 : int(FFTN/2)+1]                 # 提取 X 里正频率的部分,N为偶数
     Y_hat[1 : int(FFTN/2)] = 2*Y_hat[1 : int(FFTN/2)]   # 将 X 里负频率的部分合并到正频率,N为偶数

# 计算频域序列 Y 的幅值和相角
AY = abs(Y_hat)                        # 计算频域序列 Y 的幅值
PhaY = np.angle(Y_hat,deg=True)        # 计算频域序列 Y 的相角 (弧度制)
RY = np.real(Y_hat)                    # 计算频域序列 Y 的实部
IY = np.imag(Y_hat)                    # 计算频域序列 Y 的虚部

#%%==================== 画图 =================================
width = 6
high = 4
horvizen = 2
vertical = 2
fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*width, vertical*high), constrained_layout = True)
labelsize = 20

#%%==================== x =================================
axs[0,0].plot(t, x, label = 'raw')

# font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs[0,0].set_xlabel('time', )
axs[0,0].set_ylabel('x', )

legend1 = axs[0,0].legend(loc='best', borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)

#%%==================== x =================================
axs[0,1].plot(fx, AX, label = 'FFT(x)', color = 'red')

# font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs[0,1].set_xlabel('Frequancy(Hz)', )
axs[0,1].set_ylabel('FFT(x)', )

legend1 = axs[0,1].legend(loc='best', borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)

#%%==================== y =================================
axs[1,0].plot(t, y, label = 'y')

# font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs[1,0].set_xlabel('time', )
axs[1,0].set_ylabel('y', )

legend1 = axs[0,0].legend(loc='best',  borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)

#%%==================== FFT Y =================================
axs[1,1].plot(fx, AY, label = 'FFT(y)', color = 'red')

# font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs[1,1].set_xlabel('Frequancy(Hz)', )
axs[1,1].set_ylabel('FFT(y)', )

legend1 = axs[1,1].legend(loc='best', borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)

plt.show()
plt.close()

# #-----------------------确定相应的FIR数字滤波器指标-------------------------
# wp = 2 * np.pi*1000/Fs  #通带截止频率
# # ws=2*pi*50/Fs
# ws = np.pi/4      #阻带截止频率
# Bt = wp-ws     #过渡带带宽
# N0 = np.ceil(6.2*np.pi/Bt)  #汉宁窗窗长
# N = N0+(N0+1)%2    #由于是高通滤波器，所以窗长N必须为奇数
# wc = (wp+ws)/(2*np.pi)   #计算理想高通滤波器通带截止频率（关于pi归一化）
# #----------------------FIR数字高通滤波器(汉宁窗)-------------------------
# #hn=firls(N-1,wc,'high',hanning(N))
# hn = firwin(int(N), wc, window = 'hamming',pass_zero = 'highpass')

# [BW,BH] = freqz(hn,1) # 绘制频率响应曲线


# Bf = lfilter(hn,1,s) # 进行高通滤波
# # By = np.fft.fft(Bf,lens)  # 对高通滤波输出信号做len点FFT变换





































