#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 21:59:54 2022

@author: jack

https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.fft.html
https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html
https://vimsky.com/examples/usage/python-numpy.fft.fftshift.html
https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html
https://zhuanlan.zhihu.com/p/559711158
https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html

"""

import matplotlib.pyplot as plt
import numpy as np
# import math
import matplotlib
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
# import copy
# from matplotlib.pyplot import MultipleLocator
import scipy

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
# FFT变换，慢的版本
def FFT(xx):
     N = len(xx)
     X = np.zeros(N, dtype = complex) # 频域频谱
     # DTF变换
     for k in range(N):
         for n in range(N):
             X[k] = X[k] + xx[n]*np.exp(-1j*2*np.pi*n*k/N)
     return X

# IFFT变换，慢的版本
def IFFT(XX):
     N = len(XX)
     # IDFT变换
     x_p = np.zeros(N, dtype = complex)
     for n in range(N):
          for k in range(N):
               x_p[n] = x_p[n] + 1/N*X[k]*np.exp(1j*2*np.pi*n*k/N)
     return x_p

#======================================================
# ===========  定义时域采样信号 cos(x)
#======================================================

# Fs = 10                          # 信号采样频率
# Ts = 1/Fs                        # 采样时间间隔
# N = 100                           # 采样信号的长度
# t = np.linspace(0, N-1, N)*Ts    # 定义信号采样的时间点 t

# f1 = 2                             # 第一个余弦信号的频率
# x =  np.cos(2*np.pi*f1*t)

#======================================================
# ===========  定义时域采样信号 cos(x + theta)
#======================================================

# Fs = 10                          # 信号采样频率
# Ts = 1/Fs                        # 采样时间间隔
# N = 100                           # 采样信号的长度
# t = np.linspace(0, N-1, N)*Ts    # 定义信号采样的时间点 t

# f1 = 2                             # 第一个余弦信号的频率
# x =  np.cos(2*np.pi*f1*t+np.pi/4)

# ======================================================
# ===========  定义时域采样信号 sin(x) = cos(pi/2 - x)
# ======================================================

# Fs = 10                          # 信号采样频率
# Ts = 1/Fs                        # 采样时间间隔
# N = 100                           # 采样信号的长度
# t = np.linspace(0, N-1, N)*Ts    # 定义信号采样的时间点 t

# f1 = 2                             # 第一个余弦信号的频率
# x =  np.sin(2*np.pi*f1*t )

## ======================================================
## ===========  定义时域采样信号 sin(x + np.pi/4)
## ======================================================

Fs = 20                          # 信号采样频率
Ts = 1/Fs                        # 采样时间间隔
N = 1024                           # 采样信号的长度
t = np.linspace(0, N-1, N)*Ts    # 定义信号采样的时间点 t

# f1 = 2                             # 第一个余弦信号的频率
# x =  4 * np.cos(2*np.pi*f1*t + np.pi/4) # = cos(x) = sin(pi/2 - x)

f1 = 2
f2 = 4
f3 = 6
x =  7*np.cos(2*np.pi*f1*t + np.pi/4) + 5*np.cos(2*np.pi*f2*t + np.pi/2) + 3*np.cos(2*np.pi*f3*t + np.pi/3) + 4.5 # (4.5是直流)


#=====================================================
# 对时域采样信号, 执行快速傅里叶变换 FFT
FFTN = 1024        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细
X = scipy.fftpack.fft(x, n = FFTN)
# X = FFT(x, n = FFTN)  # 或者用自己编写的，与 fft 一致

#%% IFFT
IX = scipy.fftpack.ifft(scipy.fftpack.fft(x))
# IX = IFFT(X) # 自己写的，和 ifft 一样

#%%==================================================
# 半谱图
#==================================================

# 消除相位混乱
threshold = np.max(np.abs(X))/10000
X[np.abs(X) < threshold] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
X = X/x.size               # 将频域序列 X 除以序列的长度 N

# 提取 X 里正频率的部分, 并且将 X 里负频率的部分合并到正频率
if FFTN%2 == 0:
     Y = X[0 : int(FFTN/2)+1]                # 提取 X 里正频率的部分,N为偶数
     Y[1 : int(FFTN/2)] = 2*Y[1 : int(FFTN/2)]   # 将 X 里负频率的部分合并到正频率,N为偶数
else: #奇数时下面的有问题
     Y = X[0 : int(FFTN/2)+1]                # 提取 X 里正频率的部分,N为奇数
     Y[1 : int(FFTN/2)+1] = 2*Y[1:int(FFTN/2)+1]   # 将 X 里负频率的部分合并到正频率,N为奇数

# 计算频域序列 Y 的幅值和相角
A = abs(Y);                       # 计算频域序列 Y 的幅值
Pha = np.angle(Y,deg=True);       # 计算频域序列 Y 的相角 (弧度制)
R = np.real(Y)                    # 计算频域序列 Y 的实部
I = np.imag(Y)                    # 计算频域序列 Y 的虚部

#  定义序列 Y 对应的频率刻度
df = Fs/FFTN                           # 频率间隔
if N%2==0:
       f = np.arange(0, int(FFTN/2)+1)*df      # 频率刻度,N为偶数
      # f = scipy.fftpack.fftfreq(FFTN, d=1/Fs)[0:int(FFTN/2)+1]
else:#奇数时下面的有问题
     f = np.arange(0, int(FFTN/2)+1)*df     # 频率刻度,N为奇数

#%%==================================================
# 全谱图
#==================================================
# 对时域采样信号, 执行快速傅里叶变换 FFT
X1 = scipy.fftpack.fft(x, n = FFTN)
# X1 = FFT(x) # 或者用自己编写的

# 消除相位混乱
X1[np.abs(X1)<1e-8] = 0;   # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
X1 = X1/x.size             # 将频域序列 X 除以序列的长度 N

### 方法一，二：将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
Y1 = scipy.fftpack.fftshift(X1,)      # 新的频域序列 Y
#Y1=X1

# 计算频域序列 Y 的幅值和相角
A1 = abs(Y1);                       # 计算频域序列 Y 的幅值
Pha1 = np.angle(Y1,deg=True)        # 计算频域序列 Y 的相角 (弧度制)
R1 = np.real(Y1)                    # 计算频域序列 Y 的实部
I1 = np.imag(Y1)                    # 计算频域序列 Y 的虚部

###  定义序列 Y 对应的频率刻度
df = Fs/FFTN                           # 频率间隔
if FFTN%2 == 0:
    # 方法一
    f1 = np.arange(-int(FFTN/2),int(FFTN/2))*df      # 频率刻度,N为偶数
    #或者如下， 方法二：
    # f1 = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/Fs))
else:#奇数时下面的有问题
    f1 = np.arange(-int(FFTN/2),int(FFTN/2)+1)*df      # 频率刻度,N为奇数

# #%% 方法三
# # 将 X 不重新排列,
# Y1 = X1

# # 计算频域序列 Y 的幅值和相角
# A1 = abs(Y1);                       # 计算频域序列 Y 的幅值
# Pha1 = np.angle(Y1,deg=True);       # 计算频域序列 Y 的相角 (弧度制)
# R1 = np.real(Y1);	                # 计算频域序列 Y 的实部
# I1 = np.imag(Y1);	                # 计算频域序列 Y 的虚部

# # 定义序列 Y 对应的频率刻度
# f1 =  scipy.fftpack.fftfreq(FFTN, 1/Fs)    # 频率刻度

#%%==================================================
#     频率刻度错位
#==================================================
X2 = scipy.fftpack.fft(x, n = FFTN)

# 消除相位混乱
X2[np.abs(X2)<1e-8] = 0        # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
X2 = X2/N             # 将频域序列 X 除以序列的长度 N

# 计算频域序列 Y 的幅值和相角
A2 = abs(X2);                       # 计算频域序列 Y 的幅值
Pha2 = np.angle(X2,deg=True)       # 计算频域序列 Y 的相角 (弧度制)
R2 = np.real(X2)                   # 计算频域序列 Y 的实部
I2 = np.imag(X2)                   # 计算频域序列 Y 的虚部

df = Fs/FFTN                           # 频率间隔
if N%2 == 0:
    # 方法一
    f2 = np.arange(0, FFTN)*df      # 频率刻度,N为偶数

#%%====================================== 开始画图 ===============================================
width = 4
high = 3
horvizen = 5
vertical = 3
fig, axs = plt.subplots(vertical, horvizen, figsize=(horvizen*width, vertical*high), constrained_layout=True)
labelsize = 20

#%% 半谱图
#======================================= 0,0 =========================================
axs[0,0].plot(t, x, color='b', linestyle='-', label='原始信号值',)

axs[0,0].set_xlabel(r'时间(s)', )
axs[0,0].set_ylabel(r'原始信号值', )

legend1 = axs[0,0].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 0,1 =========================================
axs[0,1].plot(f, A, color='r', linestyle='-', label='幅度',)

axs[0,1].set_xlabel(r'频率(Hz)', )
axs[0,1].set_ylabel(r'幅度', )

legend1 = axs[0,1].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 0,2 =========================================
axs[0,2].plot(f, Pha, color='g', linestyle='-', label='相位',)

axs[0,2].set_xlabel(r'频率(Hz)', )
axs[0,2].set_ylabel(r'相位', )

legend1 = axs[0,2].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 0,3 =========================================
axs[0,3].plot(f, R, color='cyan', linestyle='-', label='实部',)

axs[0,3].set_xlabel(r'频率(Hz)', )
axs[0,3].set_ylabel(r'实部', )

legend1 = axs[0,3].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 0,4 =========================================
axs[0,4].plot(f, I, color='#FF8C00', linestyle='-', label='虚部',)

axs[0,4].set_xlabel(r'频率(Hz)', )
axs[0,4].set_ylabel(r'虚部', )

legend1 = axs[0,4].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#%% 全谱图
#======================================= 1,0 =========================================
axs[1,0].plot(t, IX, color='b', linestyle='-', label='恢复的信号值',)

axs[1,0].set_xlabel(r'时间(s)', )
axs[1,0].set_ylabel(r'逆傅里叶变换信号值', )
#axs[0,0].set_title('信号值', fontproperties=font3)

legend1 = axs[1,0].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 1,1 =========================================
axs[1,1].plot(f1, A1, color='r', linestyle='-', label='幅度',)

axs[1,1].set_xlabel(r'频率(Hz)', )
axs[1,1].set_ylabel(r'幅度', )

legend1 = axs[1,1].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 1,2 =========================================
axs[1,2].plot(f1, Pha1, color='g', linestyle='-', label='相位',)

axs[1,2].set_xlabel(r'频率(Hz)', )
axs[1,2].set_ylabel(r'相位', )

legend1 = axs[1,2].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)

#======================================= 1,3 =========================================
axs[1,3].plot(f1, R1, color='cyan', linestyle='-', label='实部',)

axs[1,3].set_xlabel(r'频率(Hz)', )
axs[1,3].set_ylabel(r'实部', )

legend1 = axs[1,3].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 1,4 =========================================
axs[1,4].plot(f1, I1, color='#FF8C00', linestyle='-', label='虚部',)

axs[1,4].set_xlabel(r'频率(Hz)', )
axs[1,4].set_ylabel(r'虚部', )

legend1 = axs[1,4].legend(loc='best', borderaxespad=0, edgecolor='black' )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#%% 频率刻度错位
#======================================= 2,0 =========================================

#======================================= 2,1 =========================================
axs[2,1].plot(f2, A2, color='r', linestyle='-', label='幅度',)

axs[2,1].set_xlabel(r'频率(Hz)', )
axs[2,1].set_ylabel(r'幅度', )

legend1 = axs[2,1].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 2,2 =========================================
axs[2,2].plot(f2, Pha2, color='g', linestyle='-', label='相位',)

axs[2,2].set_xlabel(r'频率(Hz)', )
axs[2,2].set_ylabel(r'相位', )

legend1 = axs[2,2].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 2,3 =========================================
axs[2,3].plot(f2, R2, color='cyan', linestyle='-', label='实部',)

axs[2,3].set_xlabel(r'频率(Hz)', )
axs[2,3].set_ylabel(r'实部', )

legend1 = axs[2,3].legend(loc='best', borderaxespad=0,  edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#======================================= 2,4 =========================================
axs[2,4].plot(f2, I2, color='#FF8C00', linestyle='-', label='虚部',)

axs[2,4].set_xlabel(r'频率(Hz)', )
axs[2,4].set_ylabel(r'虚部', )

legend1 = axs[2,4].legend(loc='best', borderaxespad=0, edgecolor='black', )
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

#================================= super ===============================================
out_fig = plt.gcf()
#out_fig.savefig(filepath2+'hh.eps',  bbox_inches='tight')
plt.show()
plt.close()





















