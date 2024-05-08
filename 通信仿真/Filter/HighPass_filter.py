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

filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


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

#%%================================ IIR -巴特沃兹高通滤波器  =====================================
lf = 700    # 通带截止频率200Hz
# hf = 700
Fc = 1000   # 阻带截止频率1000Hz
Rp = 1      # 通带波纹最大衰减为1dB
Rs = 40     # 阻带衰减为40dB
#-----------------------计算最小滤波器阶数-----------------------------
na = np.sqrt(10**(0.1*Rp)-1)
ea = np.sqrt(10**(0.1*Rs)-1)
order  = np.ceil(np.log10(ea/na)/np.log10(Fc/lf))  #巴特沃兹阶数
Wn = lf*2/Fs
#---------------------- 高通滤波  -----------------------------
### 方法1
# [Bb, Ba] = scipy.signal.butter(order, Wn, 'high')
# ## [BW, BH] = scipy.signal.freqz(Bb, Ba)
# y = scipy.signal.lfilter(Bb, Ba, x) # 进行滤波

## 方法2
h = scipy.signal.firwin(int(31), lf, fs = Fs, pass_zero = "highpass")
y = scipy.signal.lfilter(h, 1, x) # 进行滤波

### 方法3
# h = scipy.signal.firwin(int(31), Wn,  pass_zero = "highpass" )
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

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs[0,0].set_xlabel('time',fontproperties=font)
axs[0,0].set_ylabel('x',fontproperties=font)
#font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=22)
#  edgecolor='black',
# facecolor = 'y', # none设置图例legend背景透明
legend1 = axs[0,0].legend(loc='best',  prop=font1, borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0,0].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs[0,0].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs[0,0].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs[0,0].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs[0,0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
labels = axs[0,0].get_xticklabels() + axs[0,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号

#%%==================== x =================================
axs[0,1].plot(fx, AX, label = 'FFT(x)', color = 'red')

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs[0,1].set_xlabel('Frequancy(Hz)',fontproperties=font)
axs[0,1].set_ylabel('FFT(x)',fontproperties=font)
#font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=22)
#  edgecolor='black',
# facecolor = 'y', # none设置图例legend背景透明
legend1 = axs[0,1].legend(loc='best',  prop=font1, borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0,1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs[0,1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs[0,1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs[0,1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs[0,1].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
labels = axs[0,1].get_xticklabels() + axs[0,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号




#%%==================== y =================================
axs[1,0].plot(t, y, label = 'y')

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs[1,0].set_xlabel('time',fontproperties=font)
axs[1,0].set_ylabel('y',fontproperties=font)
#font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=22)
#  edgecolor='black',
# facecolor = 'y', # none设置图例legend背景透明
legend1 = axs[0,0].legend(loc='best',  prop=font1, borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,0].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs[1,0].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs[1,0].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs[1,0].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs[1,0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
labels = axs[1,0].get_xticklabels() + axs[1,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号

#%%==================== FFT Y =================================
axs[1,1].plot(fx, AY, label = 'FFT(y)', color = 'red')

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs[1,1].set_xlabel('Frequancy(Hz)',fontproperties=font)
axs[1,1].set_ylabel('FFT(y)',fontproperties=font)
#font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=22)
#  edgecolor='black',
# facecolor = 'y', # none设置图例legend背景透明
legend1 = axs[1,1].legend(loc='best',  prop=font1, borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs[1,1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs[1,1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs[1,1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs[1,1].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
labels = axs[1,1].get_xticklabels() + axs[1,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号




plt.show()








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





































