#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:57:55 2024
@author: jack
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import scipy

filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)

fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)

## ======================================================
## ===========  定义时域采样信号
## ======================================================
Fs = 400                          # 信号采样频率
Ts = 1/Fs                         # 采样时间间隔

ts = 0.1                          # x(t) = sinc(t/ts),
B = 1/(2*ts)
m = 10
t = np.arange(-m*ts, m*ts, Ts)        # 定义信号采样的时间点 t
N = t.size                            # 采样信号的长度
## 基带信号
x =  np.sinc(t/ts)

## 采样脉冲序列
fs = 100                          # 冲击采样脉冲的频率
p = int(Fs/fs)
bplus = [0]*p
bplus[0] = 1
plus = np.array(bplus*int(x.size/p))

## 采样后的信号
x_sample = x * plus

#%%================================== x的 FFT =======================================
# 对时域采样信号, 执行快速傅里叶变换 FFT
FFTN = t.size        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
X = scipy.fftpack.fft(x, n = FFTN)

# IFFT
IX = scipy.fftpack.ifft(scipy.fftpack.fft(x))

#====================
# 全谱图
#=====================
# 消除相位混乱
X[np.abs(X)<1e-8] = 0      # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
X = X/N               # 将频域序列 X 除以序列的长度 N

# 方法一，二：将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
Y = scipy.fftpack.fftshift(X)      # 新的频域序列 Y

# 计算频域序列 Y 的幅值和相角
A = abs(Y)                        # 计算频域序列 Y 的幅值
Pha = np.angle(Y, deg = True)     # 计算频域序列 Y 的相角 (弧度制)
R = np.real(Y)                    # 计算频域序列 Y 的实部
I = np.imag(Y)                    # 计算频域序列 Y 的虚部

###  定义序列 Y 对应的频率刻度
df = Fs/FFTN                           # 频率间隔
if FFTN%2 == 0:
    # 方法一
    # f1 = np.arange(-int(FFTN/2),int(FFTN/2))*df      # 频率刻度,N为偶数
    #或者如下， 方法二：
    f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/Fs))
else:#奇数时下面的有问题
    f = np.arange(-int(FFTN/2),int(FFTN/2)+1)*df      # 频率刻度,N为奇数

#%%================================== plus 的 FFT =======================================
# 对时域采样信号, 执行快速傅里叶变换 FFT
FFTN = t.size        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
Xplus = scipy.fftpack.fft(plus, n = FFTN)

#  IFFT
IXplus = scipy.fftpack.ifft(scipy.fftpack.fft(plus))

#==================================================
# 全谱图
#==================================================

# 消除相位混乱
Xplus[np.abs(Xplus)<1e-8] = 0      # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
Xplus = Xplus/N               # 将频域序列 X 除以序列的长度 N

# 方法一，二：将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
Yplus = scipy.fftpack.fftshift(Xplus)      # 新的频域序列 Y

# 计算频域序列 Y 的幅值和相角
Aplus = abs(Yplus)                        # 计算频域序列 Y 的幅值
Phaplus = np.angle(Yplus, deg = True)     # 计算频域序列 Y 的相角 (弧度制)
Rplus = np.real(Yplus)                    # 计算频域序列 Y 的实部
Iplus = np.imag(Yplus)                    # 计算频域序列 Y 的虚部


#%%================================== x_sample 的 FFT =======================================
# 对时域采样信号, 执行快速傅里叶变换 FFT
FFTN = t.size        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
X_sampled = scipy.fftpack.fft(x_sample, n = FFTN)

#%% IFFT
IXsampled = scipy.fftpack.ifft(scipy.fftpack.fft(x_sample))

#====================
# 全谱图
#====================
# 消除相位混乱
X_sampled[np.abs(X_sampled)<1e-8] = 0      # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
X_sampled = X_sampled/N               # 将频域序列 X 除以序列的长度 N

#%% 方法一，二：将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
Ysampled = scipy.fftpack.fftshift(X_sampled)      # 新的频域序列 Y

# 计算频域序列 Y 的幅值和相角
ASampled = abs(Ysampled)                        # 计算频域序列 Y 的幅值
PhaSampled = np.angle(Ysampled, deg = True)     # 计算频域序列 Y 的相角 (弧度制)
RSampled = np.real(Ysampled)                    # 计算频域序列 Y 的实部
ISampled = np.imag(Ysampled)                    # 计算频域序列 Y 的虚部

#====================================== 开始画图 ===============================================
width = 4
high = 3
horvizen = 2
vertical = 3
fig, axs = plt.subplots(vertical, horvizen, figsize=(horvizen*width, vertical*high), constrained_layout=True)
labelsize = 20


#%% x
#======================================= 0,0 =========================================
axs[0,0].plot(t, x, color='b', linestyle='-', label='原始信号值',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[0,0].set_xlabel(r'时间(s)', fontproperties=font3)
axs[0,0].set_ylabel(r'原始信号值', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[0,0].legend(loc='best', borderaxespad=0,  edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[0,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,0].get_xticklabels() + axs[0,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


#======================================= 0,1 =========================================
axs[0,1].plot(f, A, color='r', linestyle='-', label='幅度',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[0,1].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[0,1].set_ylabel(r'幅度', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[0,1].legend(loc='best', borderaxespad=0,  edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[0,1].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,1].get_xticklabels() + axs[0,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


# #======================================= 0,2 =========================================
# axs[0,2].plot(f, Pha, color='g', linestyle='-', label='相位',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[0,2].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[0,2].set_ylabel(r'相位', fontproperties=font3)
# #axs[0,0].set_title('信号值', fontproperties=font3)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[0,2].legend(loc='best', borderaxespad=0,  edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs[0,2].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
# labels = axs[0,2].get_xticklabels() + axs[0,2].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


# #======================================= 0,3 =========================================
# axs[0,3].plot(f, R, color='cyan', linestyle='-', label='实部',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[0,3].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[0,3].set_ylabel(r'实部', fontproperties=font3)
# #axs[0,0].set_title('信号值', fontproperties=font3)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[0,3].legend(loc='best', borderaxespad=0,
#                         edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs[0,3].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
# labels = axs[0,3].get_xticklabels() + axs[0,3].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


# #======================================= 0,4 =========================================
# axs[0,4].plot(f, I, color='#FF8C00', linestyle='-', label='虚部',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[0,4].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[0,4].set_ylabel(r'虚部', fontproperties=font3)
# #axs[0,0].set_title('信号值', fontproperties=font3)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[0,4].legend(loc='best', borderaxespad=0,
#                         edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs[0,4].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
# labels = axs[0,4].get_xticklabels() + axs[0,4].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


#%% plus
#======================================= 1,0 =========================================
axs[1,0].plot(t, plus, color='b', marker='o', label='Plus',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1,0].set_xlabel(r'时间(s)', fontproperties=font3)
axs[1,0].set_ylabel(r'plus', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[1,0].legend(loc='best', borderaxespad=0,
                        edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[1,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[1,0].get_xticklabels() + axs[1,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


#======================================= 1,1 =========================================
axs[1,1].plot(f, Aplus, color='r', linestyle='-', label='幅度',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1,1].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[1,1].set_ylabel(r'幅度', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[1,1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[1,1].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[1,1].get_xticklabels() + axs[1,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


# #======================================= 1,2 =========================================
# axs[1,2].plot(f , Phaplus, color='g', linestyle='-', label='相位',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[1,2].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[1,2].set_ylabel(r'相位', fontproperties=font3)
# #axs[0,0].set_title('信号值', fontproperties=font3)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[1,2].legend(loc='best', borderaxespad=0,
#                         edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs[1,2].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
# labels = axs[1,2].get_xticklabels() + axs[1,2].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


# #======================================= 1,3 =========================================
# axs[1,3].plot(f, Rplus, color='cyan', linestyle='-', label='实部',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[1,3].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[1,3].set_ylabel(r'实部', fontproperties=font3)
# #axs[0,0].set_title('信号值', fontproperties=font3)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[1,3].legend(loc='best', borderaxespad=0,
#                         edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs[1,3].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
# labels = axs[1,3].get_xticklabels() + axs[1,3].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


# #======================================= 1,4 =========================================
# axs[1,4].plot(f, Iplus, color='#FF8C00', linestyle='-', label='虚部',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[1,4].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[1,4].set_ylabel(r'虚部', fontproperties=font3)
# #axs[0,0].set_title('信号值', fontproperties=font3)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[1,4].legend(loc='best', borderaxespad=0,
#                         edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs[1,4].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
# labels = axs[1,4].get_xticklabels() + axs[1,4].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


#%% 频率刻度错位
#======================================= 2,0 =========================================

# axs[2,0].plot(t[::4], x_sample[::4], color='b', marker = 'o',  label='x_sampled',)
axs[2,0].scatter(t[::4], x_sample[::4], color='b', marker = 'o',  label='x_sampled',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[2,0].set_xlabel(r'时间(s)', fontproperties=font3)
axs[2,0].set_ylabel(r'x_sampled', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[2,0].legend(loc='best', borderaxespad=0,  edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[2,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[2,0].get_xticklabels() + axs[2,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


#======================================= 2,1 =========================================
axs[2,1].plot(f, ASampled, color='r', linestyle='-', label='幅度',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[2,1].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[2,1].set_ylabel(r'幅度', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[2,1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[1,1].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[2,1].get_xticklabels() + axs[2,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


# #======================================= 2,2 =========================================
# axs[2,2].plot(f, PhaSampled, color='g', linestyle='-', label='相位',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[2,2].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[2,2].set_ylabel(r'相位', fontproperties=font3)
# #axs[0,0].set_title('信号值', fontproperties=font3)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[2,2].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs[2,2].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
# labels = axs[2,2].get_xticklabels() + axs[2,2].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


# #======================================= 2,3 =========================================
# axs[2,3].plot(f, RSampled, color='cyan', linestyle='-', label='实部',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[2,3].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[2,3].set_ylabel(r'实部', fontproperties=font3)
# #axs[0,0].set_title('信号值', fontproperties=font3)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[2,3].legend(loc='best', borderaxespad=0,  edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs[2,3].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
# labels = axs[2,3].get_xticklabels() + axs[2,3].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


# #======================================= 2,4 =========================================
# axs[2,4].plot(f, ISampled, color='#FF8C00', linestyle='-', label='虚部',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[2,4].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[2,4].set_ylabel(r'虚部', fontproperties=font3)
# #axs[0,0].set_title('信号值', fontproperties=font3)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[2,4].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

# axs[2,4].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
# labels = axs[2,4].get_xticklabels() + axs[2,4].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#================================= super ===============================================
out_fig = plt.gcf()
#out_fig.savefig(filepath2+'hh.eps',  bbox_inches='tight')
plt.show()
plt.close()
