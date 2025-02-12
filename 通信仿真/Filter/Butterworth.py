#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:53:16 2024

@author: jack
https://blog.csdn.net/qq_36002089/article/details/127793378
https://blog.csdn.net/KPer_Yang/article/details/127895985

https://blog.csdn.net/qq_30759585/article/details/112056566
https://blog.csdn.net/Simon223/article/details/118758793
"""

# https://blog.csdn.net/u014033218/article/details/97004609
#
from   scipy  import   signal

# 1).低通滤波
# 这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除400hz以上频率成分，即截至频率为400hz,则wn=2*400/1000=0.8。Wn=0.8
# b, a  =   signal.butter( 8 ,  0.8 ,  'lowpass' )    #配置滤波器 8 表示滤波器的阶数
# filtedData  =   signal.filtfilt(b, a, data)   #data为要过滤的信号


# 2).高通滤波
# 这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除100hz以下频率成分，即截至频率为100hz,则wn=2*100/1000=0.2。Wn=0.2
# b, a = signal.butter(8, 0.2, 'highpass')   #配置滤波器 8 表示滤波器的阶数
# filtedData = signal.filtfilt(b, a, data)  #data为要过滤的信号

# 3).带通滤波
# 这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除100hz以下，400hz以上频率成分，即截至频率为100，400hz,则wn1=2*100/1000=0.2，Wn1=0.2； wn2=2*400/1000=0.8，Wn2=0.8。Wn=[0.02,0.8]
# b, a  =   signal.butter( 8 , [ 0.2 , 0.8 ],  'bandpass' )    #配置滤波器 8 表示滤波器的阶数
# filtedData  =   signal.filtfilt(b, a, data)   #data为要过滤的信号

# 4).带阻滤波
# 这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除100hz以上，400hz以下频率成分，即截至频率为100，400hz,则wn1=2*100/1000=0.2，Wn1=0.2； wn2=2*400/1000=0.8，Wn2=0.8。Wn=[0.02,0.8]，和带通相似，但是带通是保留中间，而带阻是去除。
# b, a  =   signal.butter( 8 , [ 0.2 , 0.8 ],  'bandstop' )    #配置滤波器 8 表示滤波器的阶数
# filtedData  =   signal.filtfilt(b, a, data)   #data为要过滤的信号


#%%===================================================================================
# https://www.delftstack.com/zh/howto/python/low-pass-filter-python/
import numpy as np
from scipy.signal import butter, lfilter,filtfilt, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog = False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data) # or
    y1 = lfilter(b, a, data)
    return y, y1


# Setting standard filter requirements.
order = 10
fs = 30.0
cutoff = 3.667

b, a = butter_lowpass(cutoff, fs, order)

fig = plt.figure(figsize=(16,10), constrained_layout = True)
# Plotting the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5 * fs * w / np.pi, np.abs(h), "b")
plt.plot(cutoff, 0.5 * np.sqrt(2), "ko")
plt.axvline(cutoff, color="k")
plt.xlim(0, 0.5 * fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel("Frequency [Hz]")
plt.grid()


# Creating the data for filteration
T = 5.0  # value taken in seconds
n = int(T * fs)  # indicates total samples
t = np.linspace(0, T, n, endpoint=False)

data = ( np.sin(1.2 * 2 * np.pi * t) + 1.5 * np.cos(9 * 2 * np.pi * t) + 0.5 * np.sin(12.0 * 2 * np.pi * t))

# Filtering and plotting
y, y1 = butter_lowpass_filter(data, cutoff, fs, order)

plt.subplot(2, 1, 2)
plt.plot(t, data, "b-", label="data")
plt.plot(t, y, "g-", linewidth=2, label="filtfilt data")
plt.plot(t, y1, "r--", linewidth=2, label="lfilter data")
plt.xlabel("Time [sec]")
plt.grid()
plt.legend()

# plt.subplots_adjust(hspace=0.35)
plt.show()


#%%===================================================================================
# https://zhuanlan.zhihu.com/p/657425129
# python中常用的巴特沃斯低通滤波器实现有两种方式，一种是filtfilt，一种是lfilter，两种区别很大。

# filtfilt 函数不适用于实时滤波，它用于离线信号处理，需要使用整个信号的历史数据进行滤波计算；
# 对于实时应用，更适合使用 lfilter 函数来逐个样本地滤波数据。lfilter函数是一个递归滤波器，可以用于实时滤波；


from scipy.signal import butter, filtfilt, lfilter, lfilter_zi
import matplotlib
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

#%%======================================================
## ===========  定义时域采样信号 x
##======================================================
## 定义时域采样信号 x
Fs = 20                     # 信号采样频率
Ts = 1/Fs                     # 采样时间间隔
N = 200                      # 采样信号的长度
t = np.linspace(0, N-1, N)*Ts    # 定义信号采样的时间点 t

f1 = 2
f2 = 4
f3 = 6
imu =  7*np.sin(2*np.pi*f1*t - np.pi/4) + 5*np.sin(2*np.pi*f2*t - np.pi/6) + 3*np.sin(2*np.pi*f3*t - np.pi/3) + 4.5 # (4.5是直流)

# 定义低通滤波器参数
order = 2  # 滤波器阶数
sampling_freq = Fs  # 采样频率（Hz）
cutoff_freq = 3  # 截止频率（Hz）

# 计算归一化截止频率
nyquist_freq = 0.5 * sampling_freq
normalized_cutoff_freq = cutoff_freq / nyquist_freq

# 使用巴特沃斯滤波器设计滤波器系数
b, a = butter(order, normalized_cutoff_freq, btype='low', analog=False)

###用 filtfilt 滤波
acc_y_filtered = filtfilt(b, a, imu )
###

###用 lfilter 滤波
# 初始化滤波器状态
zi = lfilter_zi(b, a)
acc_y_filtered_2 = np.zeros_like(imu )
for index in range(len(imu)):
    raw_value = imu[index ]
    filtered, zi = lfilter(b, a, [raw_value], zi = zi)
    acc_y_filtered_2[index] = filtered
###

width = 10
high = 6
horvizen = 1
vertical = 1
fig, axs = plt.subplots(1, 1, figsize = (horvizen*width, vertical*high), constrained_layout = True)
labelsize = 20

axs.plot(t, imu, label = 'raw')
axs.plot(t, acc_y_filtered, label = 'filtfilt')
axs.plot(t, acc_y_filtered_2, label = 'lfilter')

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs.set_xlabel('time',fontproperties=font)
axs.set_ylabel('value',fontproperties=font)
#font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
#  edgecolor='black',
# facecolor = 'y', # none设置图例legend背景透明
legend1 = axs.legend(loc='best',  prop=font1, bbox_to_anchor=(0.5, -0.2), ncol = 3,  borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs.tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号

plt.show()


#%%===================================================================================
# https://blog.csdn.net/dss875914213/article/details/90085199


from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


Fs = 100                       # 信号采样频率
Ts = 1/Fs                     # 采样时间间隔
# N = 200                       # 采样信号的长度
t = np.arange(-1, 1, Ts)     # 定义信号采样的时间点 t

f1 = 0.75
f2 = 1.25
f3 = 3.85

x = (np.sin(2*np.pi*f1*t*(1-t) + 2.1) + 0.1*np.sin(2*np.pi*f2*t + 1) +  0.18*np.cos(2*np.pi*f3*t))
xn = x + np.random.randn(len(t)) * 0.08

normalized_cutoff_freq = 2 * 1 / Fs
b, a = signal.butter(3, 0.05, btype="low",)
zi = signal.lfilter_zi(b, a)  # 初始化滤波器状态
dss = zi
data = []
data3, _ = signal.lfilter(b, a, xn, zi = dss, )
print(dss)
for i in xn:
    z, dss = signal.lfilter(b, a, [i], zi = dss)
    data.append(z)
data2 = signal.filtfilt(b, a, xn)

fig, axs = plt.subplots(1, 1, figsize=(10, 8))
axs.plot(t, xn, 'b', linewidth = 2, label = 'raw',  alpha=0.75)
axs.plot(t, data, 'r--', linewidth = 2, label = 'lfilter, 1by1', )
axs.plot(t, data2,'g',   linewidth = 2, label = 'filtfilt',)
axs.plot(t, data3, color = 'cyan', marker ='d', ms = 12, linewidth = 2, label = 'lfilter, all', alpha=0.3)

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs.set_xlabel('time',fontproperties=font)
axs.set_ylabel('value',fontproperties=font)
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 26)
# font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
#  edgecolor='black',
# facecolor = 'y', # none设置图例legend背景透明
legend1 = axs.legend(loc='best',  prop=font1,   borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs.tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号

# lfilter 滤波后的波形有偏移，filtfilt滤波后的没有偏移。
axs.grid(True)
plt.show()
































































































































































































































































































































































































































































































































































































































































































































