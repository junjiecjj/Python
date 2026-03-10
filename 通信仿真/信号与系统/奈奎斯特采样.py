#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:50:05 2026

@author: jack
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import scipy
import math

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 20               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 20          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 20          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 20         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 20         # 设置 y 轴刻度字体大小
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
plt.rcParams['legend.fontsize'] = 12

np.random.seed(42)

#%%
def freqDomainView(x, Fs, FFTN, type = 'double'): # N为偶数
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
        A = abs(Y)                        # 计算频域序列 Y 的幅值
        Pha = np.angle(Y, deg=1)          # 计算频域序列 Y 的相角 (弧度制)
        R = np.real(Y)                    # 计算频域序列 Y 的实部
        I = np.imag(Y)                    # 计算频域序列 Y 的虚部
    elif type == 'double':
        f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/Fs))
        Y = scipy.fftpack.fftshift(X, )
        # 计算频域序列 Y 的幅值和相角
        A = abs(Y)                        # 计算频域序列 Y 的幅值
        Pha = np.angle(Y, deg=1)          # 计算频域序列 Y 的相角 (弧度制)
        R = np.real(Y)                    # 计算频域序列 Y 的实部
        I = np.imag(Y)                    # 计算频域序列 Y 的虚部

    return f, Y, A, Pha, R, I


#%%  展示sinc函数
ts = 1                          # x(t) = sinc(t/ts), T = 0.1, f = 10 Hz
B  = 1/(2*ts)                     # Hz
# f_max = 2*np.pi*B               # 角频率rad/s,
f_max = B                         # 画图用的时间频率 Hz
Fs = 40                           # 信号采样频率
Ts = 1/Fs                         # 采样时间间隔
# N = 100                         # 采样信号的长度

m = 5
t = np.arange(-m*ts, m*ts, Ts)    # 定义信号采样的时间点 t
x = np.sinc(t/ts)
x_len = x.size

width = 6
high = 4
horvizen = 2
vertical = 1
fig, axs = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
labelsize = 12

axs.plot(t, x, color='b', linestyle='-', label='原始信号值',)

axs.set_xlabel(r't', )
axs.set_ylabel(r'x(t)', )

axs.tick_params(direction='in', axis='both',top=False,right=False, labelsize=labelsize, width=3,)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号
axs.set_xlim(-ts*4, ts*4)  #拉开坐标轴范围显示投影
axs.set_xticks([-5, -4, -3, -2, -1, 0, 1,2,3,4,5])

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.spines['bottom'].set_position(('data',0.0))
axs.spines['left'].set_position(('data',0.0))

out_fig = plt.gcf()
out_fig.savefig('sinc.pdf', )
plt.show()
plt.close()



#%% 展示sinc函数+FFT
## ======================================================
## ===========  定义时域采样信号
## ======================================================
ts = 0.1                          # x(t) = sinc(t/ts), T = 0.1, f = 10 Hz
B  = 1/(2*ts)                     # Hz
# f_max = 2*np.pi*B               # 角频率rad/s,
f_max = B                         # 画图用的时间频率 Hz
Fs = 200                           # 信号采样频率
Ts = 1/Fs                         # 采样时间间隔
# N = 100                         # 采样信号的长度

m = 20
t = np.arange(-m*ts, m*ts, Ts)    # 定义信号采样的时间点 t
x = np.sinc(t/ts)
x_len = x.size
#=====================================================
# 对时域采样信号, 执行快速傅里叶变换 FFT
FFTN = 1024 # 2**math.ceil(math.log(x.size, 2))        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细

fx, Yx, Ax, Phax, Rx, Ix = freqDomainView(x, Fs, FFTN, type = 'double')

###
width = 6
high = 4
horvizen = 3
vertical = 1
fig, axs = plt.subplots(vertical, horvizen, figsize=(horvizen*width, vertical*high), constrained_layout=True)
labelsize = 12

axs[0].plot(t, x, color='b', linestyle='-', label='原始信号值',)

axs[0].set_xlabel(r't', )
axs[0].set_ylabel(r'x(t)', )

# legend1 = axs[0,0].legend(loc='best', borderaxespad=0,  edgecolor='black', )
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0].tick_params(direction='in', axis='both',top=False,right=False, labelsize=labelsize, width=3,)
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号
axs[0].set_xlim(-ts*4, ts*4)  #拉开坐标轴范围显示投影

axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['bottom'].set_position(('data',0.0))
axs[0].spines['left'].set_position(('data',0.0))
# 去除轴标签
# axs[0,0].set_xticklabels([])
# axs[0,0].set_yticklabels([])
# 去除轴刻度
# axs[0,0].set_xticks([])
# axs[0,0].set_yticks([])

axs[1].plot(fx, Ax, color='r', linestyle='-', label='幅度',)
axs[1].set_xlabel(r'f(Hz)', )
axs[1].set_ylabel(r'X(f)', )

axs[1].tick_params(direction='in', axis='both',top=False,right=False, labelsize=labelsize, width=3,)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号
# axs[1].set_xlim(-2*B, 2*B)  #拉开坐标轴范围显示投影

axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].spines['bottom'].set_position(('data',0.0))
axs[1].spines['left'].set_position(('data',0.0))

axs[2].plot(fx, Ax, color='r', linestyle='-', label='幅度',)
axs[2].set_xlabel(r'f(Hz)', )
axs[2].set_ylabel(r'X(f)', )

axs[2].tick_params(direction='in', axis='both',top=False,right=False, labelsize=labelsize, width=3,)
labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号
axs[2].set_xlim(-2*B, 2*B)  #拉开坐标轴范围显示投影

axs[2].spines['right'].set_visible(False)
axs[2].spines['top'].set_visible(False)
axs[2].spines['bottom'].set_position(('data',0.0))
axs[2].spines['left'].set_position(('data',0.0))

#================================= super ===============================================
out_fig = plt.gcf()
out_fig.savefig('sinc_fft.pdf', )
plt.show()
plt.close()


#%% 利用sinc函数展示采样定理, 并还原信号
# https://blog.csdn.net/weixin_50345615/article/details/127416572
## ======================================================
## ===========  定义时域采样信号
## ======================================================
ts = 0.1                          # x(t) = sinc(t/ts), T = 0.1, f = 10 Hz
B  = 1/(2*ts)                     # Hz
# f_max = 2*np.pi*B               # 角频率rad/s,
f_max = B                         # 画图用的时间频率 Hz
Fs = 400                          # 信号采样频率
Ts = 1/Fs                         # 采样时间间隔
# N = 100                         # 采样信号的长度

m = 10
t = np.arange(-m*ts, m*ts, Ts)    # 定义信号采样的时间点 t
x = np.sinc(t/ts)
x_len = x.size

## 采样脉冲序列
fs = 10                          # 冲击采样脉冲的频率
p = int(Fs/fs)
bplus = [0]*p
bplus[0] = 1
plus = np.array(bplus*int(x.size/p))

## 采样后的信号
x_sample = x * plus

#%%============================== 时域低通滤波，信号恢复 ======================================
#%%================================ IIR -巴特沃兹 带阻 滤波器  =====================================
lf = fs/2    # 通带截止频率200Hz
hf = fs
Fc = 200   # 阻带截止频率1000Hz
Rp = 1      # 通带波纹最大衰减为1dB
Rs = 40     # 阻带衰减为40dB
#-----------------------计算最小滤波器阶数-----------------------------
na = np.sqrt(10**(0.1*Rp)-1)
ea = np.sqrt(10**(0.1*Rs)-1)
order  = np.ceil(np.log10(ea/na)/np.log10(Fc/lf))  #巴特沃兹阶数

# order = 4
Wn = lf / Fs
[Bb, Ba] = scipy.signal.butter(order, Wn, 'low')
# ## [BW, BH] = scipy.signal.freqz(Bb, Ba)
# x_recov = scipy.signal.lfilter(Bb, Ba, x) # 进行滤波
x_recov = scipy.signal.lfilter(Bb, Ba, x_sample) # 进行滤波
x_recov = x_recov/np.max(np.abs(x_recov))


#%%==========================================================================================
# 对时域采样信号, 执行快速傅里叶变换 FFT
FFTN = 1024 # 2**math.ceil(math.log(x.size, 2))        ## 执行FFT的点数，可以比N_sample大很多，越大频谱越精细

fx, Yx, Ax, Phax, Rx, Ix = freqDomainView(x, Fs, FFTN, type = 'double')

f_plus, Y_plus, A_plus, Pha_plus, R_plus, I_plus = freqDomainView(plus, Fs, FFTN, type = 'double')

f_xsam, Y_xsam, A_xsam, Pha_xsam, R_xsam, I_xsam = freqDomainView(x_sample, Fs, FFTN, type = 'double')

f_xrec, Y_xrec, A_xrec, Pha_xrec, R_xrec, I_xrec = freqDomainView(x_recov, Fs, FFTN, type = 'double')


#%%
width = 5
high = 3
horvizen = 4
vertical = 2
fig, axs = plt.subplots(vertical, horvizen, figsize=(horvizen*width, vertical*high), constrained_layout=True)
labelsize = 12

axs[0,0].plot(t, x, color='b', linestyle='-', label='原始信号值',)

axs[0,0].set_xlabel(r't', )
axs[0,0].set_title(r'x(t)', )

# legend1 = axs[0,0].legend(loc='best', borderaxespad=0,  edgecolor='black', )
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0,0].tick_params(direction='in', axis='both',top=False,right=False, labelsize=labelsize, width=3,)
labels = axs[0,0].get_xticklabels() + axs[0,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号
axs[0,0].set_xlim(-ts*4, ts*4)  #拉开坐标轴范围显示投影

axs[0,0].spines['right'].set_visible(False)
axs[0,0].spines['top'].set_visible(False)
axs[0,0].spines['bottom'].set_position(('data',0.0))
axs[0,0].spines['left'].set_position(('data',0.0))
# 去除轴标签
# axs[0,0].set_xticklabels([])
# axs[0,0].set_yticklabels([])
# 去除轴刻度
# axs[0,0].set_xticks([])
# axs[0,0].set_yticks([])


axs[0,1].stem(t[::p], plus[::p], linefmt='r--', markerfmt='gD', basefmt='b--', bottom=0)
axs[0,1].set_xlabel(r't', )
axs[0,1].set_title(r'p(t)', )

axs[0,1].tick_params(direction='in', axis='both',top=False,right=False, labelsize=labelsize, width=3,)
labels = axs[0,1].get_xticklabels() +axs[0,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

axs[0,1].spines['right'].set_visible(False)
axs[0,1].spines['top'].set_visible(False)
axs[0,1].spines['bottom'].set_position(('data',0.0))
axs[0,1].spines['left'].set_position(('data',0.0))
axs[0,1].set_xlim(-ts, ts)  #拉开坐标轴范围显示投影


axs[0,2].plot(t[::p], x_sample[::p], marker = 'o', ms = 2, mec = 'r', ls = '--',  lw = 1, c='b')
axs[0,2].set_xlabel(r't', )
axs[0,2].set_title(r'$x_s(t)$', )

axs[0,2].tick_params(direction='in', axis='both',top=False,right=False, labelsize=labelsize, width=3,)
labels = axs[0,2].get_xticklabels() +axs[0,2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号
# axs[0,1].set_xlim(-ts*4, ts*4)  #拉开坐标轴范围显示投影

axs[0,2].spines['right'].set_visible(False)
axs[0,2].spines['top'].set_visible(False)
axs[0,2].spines['bottom'].set_position(('data',0.0))
axs[0,2].spines['left'].set_position(('data',0.0))
axs[0,2].set_xlim(-ts*4, ts*4)  #拉开坐标轴范围显示投影

axs[0,3].plot(t, x_recov, color='r', linestyle='-', label='幅度',)
axs[0,3].set_xlabel(r't', )
axs[0,3].set_title(r'$\hat{x}(t)$', )

axs[0,3].tick_params(direction='in', axis='both',top=False,right=False, labelsize=labelsize, width=3,)
labels = axs[0,3].get_xticklabels() + axs[0,3].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号
# axs[0,1].set_xlim(-ts*4, ts*4)  #拉开坐标轴范围显示投影

axs[0,3].spines['right'].set_visible(False)
axs[0,3].spines['top'].set_visible(False)
axs[0,3].spines['bottom'].set_position(('data',0.0))
axs[0,3].spines['left'].set_position(('data',0.0))
axs[0,3].set_xlim(-ts*4, ts*4)  #拉开坐标轴范围显示投影

axs[1,0].plot(fx, Ax, color='b', linestyle='-', )
axs[1,0].set_xlabel(r'f(Hz)', )
axs[1,0].set_title(r'X(f)', )
#axs[0,0].set_title('信号值', fontproperties=font3)

axs[1,0].tick_params(direction='in', axis='both',top=False,right=False, labelsize=labelsize, width=3,)
labels = axs[1,0].get_xticklabels() + axs[1,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

axs[1,0].spines['right'].set_visible(False)
axs[1,0].spines['top'].set_visible(False)
axs[1,0].spines['bottom'].set_position(('data',0.0))
axs[1,0].spines['left'].set_position(('data',0.0))
axs[1,0].set_xlim(-2*B, 2*B)  #拉开坐标轴范围显示投影


axs[1,1].plot(f_plus, A_plus, color='r', linestyle='-', )
axs[1,1].set_xlabel(r'f(Hz)', )
axs[1,1].set_title(r'P(f)', )
#axs[0,0].set_title('信号值', fontproperties=font3)

axs[1,1].tick_params(direction='in', axis='both',top=False,right=False, labelsize=labelsize, width=3,)
labels = axs[1,1].get_xticklabels() + axs[1,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

axs[1,1].spines['right'].set_visible(False)
axs[1,1].spines['top'].set_visible(False)
axs[1,1].spines['bottom'].set_position(('data',0.0))
axs[1,1].spines['left'].set_position(('data',0.0))
# axs[1,1].set_xlim(-10*B, 10*B)  #拉开坐标轴范围显示投影

axs[1,2].plot(f_xsam, A_xsam, color='b', linestyle='-', )
axs[1,2].set_xlabel(r'f(Hz)', )
axs[1,2].set_title(r'$X_s(f)$', )
#axs[0,0].set_title('信号值', fontproperties=font3)

axs[1,2].tick_params(direction='in', axis='both',top=False,right=False, labelsize=labelsize, width=3,)
labels = axs[1,2].get_xticklabels() + axs[1,2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

axs[1,2].spines['right'].set_visible(False)
axs[1,2].spines['top'].set_visible(False)
axs[1,2].spines['bottom'].set_position(('data',0.0))
axs[1,2].spines['left'].set_position(('data',0.0))
# axs[1,1].set_xlim(-10*B, 10*B)  #拉开坐标轴范围显示投影



axs[1,3].plot(f_xrec, A_xrec, color='b', linestyle='-', )
axs[1,3].set_xlabel(r'f(Hz)', )
axs[1,3].set_title(r'$\hat{X}(f)$', )
#axs[0,0].set_title('信号值', fontproperties=font3)

axs[1,3].tick_params(direction='in', axis='both',top=False,right=False, labelsize=labelsize, width=3,)
labels = axs[1,3].get_xticklabels() + axs[1,3].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

axs[1,3].spines['right'].set_visible(False)
axs[1,3].spines['top'].set_visible(False)
axs[1,3].spines['bottom'].set_position(('data',0.0))
axs[1,3].spines['left'].set_position(('data',0.0))
# axs[1,3].set_xlim(-100, 100)  #拉开坐标轴范围显示投影



#================================= super ===============================================
out_fig = plt.gcf()
out_fig.savefig(f'sinc_recv_{fs}.pdf', )
plt.show()
plt.close()





























































































