#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 21:59:54 2022
@author: jack

信号的频谱搬移

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import scipy


filepath2 = '/home/jack/snap/'

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


#%%======================================================
## ===========  定义时域采样信号 x 和 载波 s
##======================================================

f1 = 200
f2 = 400
f3 = 600
fs = 1000                  # 载波频率


## 定义时域采样信号 x
Fs = 4000                     # 信号采样频率
Ts = 1/Fs                     # 采样时间间隔
N = 1400                      # 采样信号的长度
t = np.linspace(0, N-1, N)*Ts    # 定义信号采样的时间点 t

## 基带信号
x = 7*np.cos(2*np.pi*f1*t + np.pi/4) + 5*np.cos(2*np.pi*f2*t + np.pi/2) + 3*np.cos(2*np.pi*f3*t + np.pi/3) #+ 4.5 # (4.5是直流)
## 载波
s = np.exp(1j * 2 * np.pi * fs * t)
# s = np.cos(2 * np.pi * fs * t)
h = x * s

#%%======================= x ==============================
# 对时域采样信号, 执行快速傅里叶变换 FFT
X = scipy.fftpack.fft(x)
# X = FFT(x)  # 或者用自己编写的，与 fft 一致

#  IFFT
IX = scipy.fftpack.ifft(X)
# IX = IFFT(X)*N # 自己写的，和 ifft 一样

# 消除相位混乱
X[np.abs(X) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
X = X/N               # 将频域序列 X 除以序列的长度 N

#==================================================
# 半谱图
#==================================================

# 提取 X 里正频率的部分, 并且将 X 里负频率的部分合并到正频率
if N%2 == 0:
     Y = X[0 : int(N/2)+1].copy()                 # 提取 X 里正频率的部分,N为偶数
     Y[1 : int(N/2)] = 2*Y[1 : int(N/2)].copy()   # 将 X 里负频率的部分合并到正频率,N为偶数

# 计算频域序列 Y 的幅值和相角
A = abs(Y)                        # 计算频域序列 Y 的幅值
Pha = np.angle(Y, deg = 1)        # 计算频域序列 Y 的相角 (弧度制)
R = np.real(Y)                    # 计算频域序列 Y 的实部
I = np.imag(Y)                    # 计算频域序列 Y 的虚部

#  定义序列 Y 对应的频率刻度
df = Fs/N                         # 频率间隔
if N%2 == 0:
     f = np.arange(0, int(N/2)+1)*df      # 频率刻度,N为偶数
      # f = scipy.fftpack.fftfreq(N, d=1/Fs)[0:int(N/2)+1]

#==================================================
# 全谱图
#==================================================
# # 对时域采样信号, 执行快速傅里叶变换 FFT
# X1 = scipy.fftpack.fft(x)

# # 消除相位混乱
# X1[np.abs(X1)<1e-8] = 0   # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
# X1 = X1/N                 # 将频域序列 X 除以序列的长度 N

#  方法一，二：将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
Y1 = scipy.fftpack.fftshift(X,)      # 新的频域序列 Y

# 计算频域序列 Y 的幅值和相角
A1 = abs(Y1);                       # 计算频域序列 Y 的幅值
Pha1 = np.angle(Y1, deg = True)        # 计算频域序列 Y 的相角 (弧度制)
R1 = np.real(Y1)                    # 计算频域序列 Y 的实部
I1 = np.imag(Y1)                    # 计算频域序列 Y 的虚部

###  定义序列 Y 对应的频率刻度
df = Fs/N                           # 频率间隔
if N%2 == 0:
    # 方法一
    # f1 = np.arange(-int(N/2),int(N/2))*df      # 频率刻度,N为偶数
    #或者如下， 方法二：
    f1 = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(N, 1/Fs))

#%%======================= s ==============================
# 对时域采样信号, 执行快速傅里叶变换 FFT
S = scipy.fftpack.fft(s)
# X = FFT(x)  # 或者用自己编写的，与 fft 一致

#  IFFT
IS = scipy.fftpack.ifft(s)
# IX = IFFT(X)*N # 自己写的，和 ifft 一样
# 消除相位混乱
S[np.abs(S) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
S = S/N               # 将频域序列 X 除以序列的长度 N

#==================================================
# 半谱图
#==================================================
# 提取 X 里正频率的部分, 并且将 X 里负频率的部分合并到正频率
if N%2 == 0:
     SY = S[0 : int(N/2)+1].copy()                 # 提取 X 里正频率的部分,N为偶数
     # SY[1 : int(N/2)] = 2*SY[1 : int(N/2)].copy()   # 将 X 里负频率的部分合并到正频率,N为偶数

# 计算频域序列 Y 的幅值和相角
AS = abs(SY)                        # 计算频域序列 Y 的幅值
PhaS = np.angle(SY, deg=1)        # 计算频域序列 Y 的相角 (弧度制)
RS = np.real(SY)                    # 计算频域序列 Y 的实部
IS = np.imag(SY)                    # 计算频域序列 Y 的虚部

#  定义序列 Y 对应的频率刻度
df = Fs/N                         # 频率间隔
if N%2==0:
     f_s = np.arange(0, int(N/2)+1)*df      # 频率刻度,N为偶数
      # f = scipy.fftpack.fftfreq(N, d=1/Fs)[0:int(N/2)+1]

#==================================================
# 全谱图
#==================================================

#  方法一，二：将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
SY1 = scipy.fftpack.fftshift(S,)      # 新的频域序列 Y
#Y1=X1

# 计算频域序列 Y 的幅值和相角
AS1 = abs(SY1);                       # 计算频域序列 Y 的幅值
PhaS1 = np.angle(SY1, deg=True)        # 计算频域序列 Y 的相角 (弧度制)
RS1 = np.real(SY1)                    # 计算频域序列 Y 的实部
IS1 = np.imag(SY1)                    # 计算频域序列 Y 的虚部

###  定义序列 Y 对应的频率刻度
df = Fs/N                           # 频率间隔
if N%2 == 0:
    # 方法一
    # f1 = np.arange(-int(N/2),int(N/2))*df      # 频率刻度,N为偶数
    #或者如下， 方法二：
    f_s1 = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(N, 1/Fs))

#%%======================= h ==============================
# 对时域采样信号, 执行快速傅里叶变换 FFT
H = scipy.fftpack.fft(h)
# X = FFT(x)  # 或者用自己编写的，与 fft 一致

#  IFFT
IH = scipy.fftpack.ifft(H)
# IX = IFFT(X)*N # 自己写的，和 ifft 一样

# 消除相位混乱
H[np.abs(H) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
H = H/N               # 将频域序列 X 除以序列的长度 N

#==================================================
# 半谱图
#==================================================

# 提取 X 里正频率的部分, 并且将 X 里负频率的部分合并到正频率
if N%2 == 0:
     HY = H[0 : int(N/2)+1].copy()                 # 提取 X 里正频率的部分,N为偶数
     # SY[1 : int(N/2)] = 2*SY[1 : int(N/2)].copy()   # 将 X 里负频率的部分合并到正频率,N为偶数

# 计算频域序列 Y 的幅值和相角
AH = abs(HY)                        # 计算频域序列 Y 的幅值
PhaH = np.angle(HY, deg=1)        # 计算频域序列 Y 的相角 (弧度制)
RH = np.real(HY)                    # 计算频域序列 Y 的实部
IH = np.imag(HY)                    # 计算频域序列 Y 的虚部

#  定义序列 Y 对应的频率刻度
df = Fs/N                         # 频率间隔
if N%2==0:
     f_h = np.arange(0, int(N/2)+1)*df      # 频率刻度,N为偶数
      # f = scipy.fftpack.fftfreq(N, d=1/Fs)[0:int(N/2)+1]

#==================================================
# 全谱图
#==================================================
# # 对时域采样信号, 执行快速傅里叶变换 FFT
# H1 = scipy.fftpack.fft(h)
# # X1 = FFT(x) # 或者用自己编写的

# # 消除相位混乱
# H1[np.abs(H1)<1e-8] = 0   # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
# H1 = H1/N                 # 将频域序列 X 除以序列的长度 N

#  方法一，二：将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
HY1 = scipy.fftpack.fftshift(H,)      # 新的频域序列 Y
#Y1=X1

# 计算频域序列 Y 的幅值和相角
AH1 = abs(HY1);                       # 计算频域序列 Y 的幅值
PhaH1 = np.angle(HY1, deg=True)        # 计算频域序列 Y 的相角 (弧度制)
RH1 = np.real(HY1)                    # 计算频域序列 Y 的实部
IH1 = np.imag(HY1)                    # 计算频域序列 Y 的虚部

###  定义序列 Y 对应的频率刻度
df = Fs/N                           # 频率间隔
if N%2 == 0:
    # 方法一
    # f1 = np.arange(-int(N/2),int(N/2))*df      # 频率刻度,N为偶数
    #或者如下， 方法二：
    f_h1 = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(N, 1/Fs))


#%%====================================== 开始画图 ===============================================
width = 4
high = 3
horvizen = 3
vertical = 2
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
legend1 = axs[0,0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,0].get_xticklabels() + axs[0,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 0,1 =========================================
axs[1,0].plot(f1, A1, color='r', linestyle='-', label='x全谱',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1,0].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[1,0].set_ylabel(r'幅度', fontproperties=font3)
axs[1,0].set_title('x全谱', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[1,0].legend(loc='best', borderaxespad=0,  edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[1,0].get_xticklabels() + axs[1,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

# #======================================= 0,2 =========================================
# axs[0,2].plot(f, Pha, color='g', linestyle='-', label='相位',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[0,2].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[0,2].set_ylabel(r'相位', fontproperties=font3)
# axs[0,2].set_title('x半谱', fontproperties=font3)

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
# axs[0,3].plot(f1, A1, color='r', linestyle='-', label='幅度',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[0,3].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[0,3].set_ylabel(r'幅度', fontproperties=font3)
# axs[0,3].set_title('x全谱', fontproperties=font3)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[0,3].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs[0,3].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
# labels = axs[0,3].get_xticklabels() + axs[0,3].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


# #======================================= 1,2 =========================================
# axs[0,4].plot(f1, Pha1, color='g', linestyle='-', label='相位',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[0,4].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[0,4].set_ylabel(r'相位', fontproperties=font3)
# axs[0,4].set_title('x全谱', fontproperties=font3)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[0,4].legend(loc='best', borderaxespad=0,  edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs[0,4].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
# labels = axs[0,4].get_xticklabels() + axs[0,4].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


#%% s
#======================================= 1,0 =========================================

axs[0,1].plot(t, s, color='b', linestyle='-', label='载波',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[0,1].set_xlabel(r'时间(s)', fontproperties=font3)
axs[0,1].set_ylabel(r'载波', fontproperties=font3)
# axs[1,0].set_title('s信号值', fontproperties=font3)

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


#======================================= 1,1 =========================================
axs[1,1].plot(f_s1, AS1, color='r', linestyle='-', label='s全谱',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1,1].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[1,1].set_ylabel(r'幅度', fontproperties=font3)
axs[1,1].set_title('s全谱', fontproperties=font3)

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
# axs[1,2].plot(f_s, PhaS, color='g', linestyle='-', label='相位',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[1,2].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[1,2].set_ylabel(r'相位', fontproperties=font3)
# axs[1,2].set_title('s半谱', fontproperties=font3)

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
# axs[1,3].plot(f_s1, AS1, color='cyan', linestyle='-', label='实部',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[1,3].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[1,3].set_ylabel(r'实部', fontproperties=font3)
# axs[1,3].set_title('s全谱', fontproperties=font3)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[1,3].legend(loc='best', borderaxespad=0,  edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs[1,3].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
# labels = axs[1,3].get_xticklabels() + axs[1,3].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

# #======================================= 1,4 =========================================
# axs[1,4].plot(f_s1, PhaS1, color='#FF8C00', linestyle='-', label='虚部',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[1,4].set_xlabel(r'频率(Hz)', fontproperties=font3)
# axs[1,4].set_ylabel(r'虚部', fontproperties=font3)
# axs[1,4].set_title('s全谱', fontproperties=font3)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[1,4].legend(loc='best', borderaxespad=0,  edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs[1,4].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
# labels = axs[1,4].get_xticklabels() + axs[1,4].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


#%% h
#======================================= 2,0 =========================================
axs[0,2].plot(t, h, color='b', linestyle='-', label='已调信号',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[0,2].set_xlabel(r'时间(s)', fontproperties=font3)
axs[0,2].set_ylabel(r'已调信号', fontproperties=font3)
# axs[2,0].set_title('已调信号', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[0,2].legend(loc='best', borderaxespad=0,  edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[0,2].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,2].get_xticklabels() + axs[0,2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


#======================================= 2,1 =========================================
axs[1,2].plot(f_h1, AH1, color='r', linestyle='-', label='已调信号全谱',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1,2].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[1,2].set_ylabel(r'已调信号全谱', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[1,2].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[1,2].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[1,2].get_xticklabels() + axs[1,2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


# #======================================= 2,2 =========================================
# axs[2,2].plot(f_h, PhaH, color='g', linestyle='-', label='相位',)

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
# axs[2,3].plot(f_h1, AH1, color='cyan', linestyle='-', label='实部',)

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
# axs[2,4].plot(f_h1, PhaH1, color='#FF8C00', linestyle='-', label='虚部',)

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
# out_fig.savefig('fft_cosshift.eps',  bbox_inches = 'tight')
out_fig.savefig('fft_expshift.eps',  bbox_inches = 'tight')
plt.show()
