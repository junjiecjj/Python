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
from scipy.signal import butter, lfilter,filtfilt, freqz, freqs
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
plt.plot(0.5 * fs * w / np.pi, 20 * np.log10(abs(h)), "b")
# plt.plot(w, 20 * np.log10(abs(h)), "b")  ## plt.xlabel("Frequency in rad/sample")
plt.plot(cutoff, 0.5 * np.sqrt(2), "ko")
plt.axvline(cutoff, color="k")
plt.xlim(0, 0.5 * fs)
plt.title("Lowpass Filter Frequency Response (dB)")
plt.xlabel("Frequency [Hz]")
plt.grid()
# plt.show()

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter
# fig = plt.figure(figsize=(8,6), constrained_layout = True)
# # Plotting the frequency response.
# w, h =  freqs(b, a)
# plt.subplot(1, 1, 1)
# plt.semilogx(w, 20 * np.log10(abs(h)), "b")
# # plt.plot(cutoff, 0.5 * np.sqrt(2), "ko")
# # plt.axvline(cutoff, color="k")
# # plt.xlim(0, 0.5 * fs)
# plt.title("Lowpass Filter Frequency Response")
# plt.xlabel("Frequency [Hz]")
# plt.grid()
# plt.show()


# Creating the data for filteration
T = 5.0  # value taken in seconds
n = int(T * fs)  # indicates total samples
t = np.linspace(0, T, n, endpoint=False)
data = np.sin(1.2 * 2 * np.pi * t) + 1.5 * np.cos(9 * 2 * np.pi * t) + 0.5 * np.sin(12.0 * 2 * np.pi * t)

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


#%% https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter
from scipy import signal


t = np.linspace(0, 1, 1000, False)  # 1 second
sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, sig)
ax1.set_title('10 Hz and 20 Hz sinusoids')
ax1.axis([0, 1, -2, 2])

sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
filtered = signal.sosfilt(sos, sig)
ax2.plot(t, filtered)
ax2.set_title('After 15 Hz high-pass filter')
ax2.axis([0, 1, -2, 2])
ax2.set_xlabel('Time [s]')
plt.tight_layout()
plt.show()

#%%===================================================================================
# https://zhuanlan.zhihu.com/p/657425129
# python中常用的巴特沃斯低通滤波器实现有两种方式，一种是 filtfilt ，一种是 lfilter ，两种区别很大。

# filtfilt 函数不适用于实时滤波，它用于离线信号处理，需要使用整个信号的历史数据进行滤波计算；
# 对于实时应用，更适合使用 lfilter 函数来逐个样本地滤波数据。lfilter函数是一个递归滤波器，可以用于实时滤波；
# lfilter 滤波后的波形有偏移，filtfilt滤波后的没有偏移。

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
# sampling_freq = Fs  # 采样频率（Hz）
cutoff_freq = 3  # 截止频率（Hz）

# 计算归一化截止频率
nyquist_freq = 0.5 * Fs
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
# lfilter 滤波后的波形有偏移，filtfilt滤波后的没有偏移。

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

x = np.sin(2*np.pi*f1*t*(1-t) + np.pi/4) + 0.1*np.sin(2*np.pi*f2*t + np.pi/2) +  0.18*np.cos(2*np.pi*f3*t)
xn = x + np.random.randn(len(t)) * 0.08

# normalized_cutoff_freq = 2 * 1 / Fs
b, a = signal.butter(3, 0.05, btype="low",)
zi = signal.lfilter_zi(b, a)  # 初始化滤波器状态
dss = zi
print(dss)

## 3: lfilter 波形，所有点一起滤波
data, _ = signal.lfilter(b, a, xn, zi = dss, )

## 1: 是lfilter 波形，一个一个点滤波
data1 = []
for i in xn:
    z, dss = signal.lfilter(b, a, [i], zi = dss)
    data1.append(z)

##  2: 是filtfilt波形。好像只能多余12个点才能滤波。
data2 = signal.filtfilt(b, a, xn)


fig, axs = plt.subplots(1, 1, figsize=(10, 8))
axs.plot(t, xn, color ='b', linewidth = 2, label = 'raw',  alpha=0.75)
axs.plot(t, data, color = 'cyan', ls = '--', linewidth = 2,  marker ='d', ms = 12, label = 'lfilter, all', )
axs.plot(t, data1, color = 'r',   linewidth = 2, label = 'lfilter, 1by1',)
axs.plot(t, data2, color = 'g', linewidth = 2, label = 'filtfilt', )

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


axs.grid(True)
plt.show()



#%% lfilter
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html#scipy.signal.lfilter
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
rng = np.random.default_rng()
t = np.linspace(-1, 1, 201)
x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) + 0.1*np.sin(2*np.pi*1.25*t + 1) + 0.18*np.cos(2*np.pi*3.85*t))
xn = x + rng.standard_normal(len(t)) * 0.08

b, a = signal.butter(3, 0.05)

zi = signal.lfilter_zi(b, a)
z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])

z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])

y = signal.filtfilt(b, a, xn)

plt.figure
plt.plot(t, xn, 'b', alpha=0.75)
plt.plot(t, z, 'r--', t, z2, 'r', t, y, 'k')
plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice', 'filtfilt'), loc='best')
plt.grid(True)
plt.show()

#%% lfilter_zi:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter_zi.html#scipy.signal.lfilter_zi

from numpy import array, ones
from scipy.signal import lfilter, lfilter_zi, butter
b, a = butter(5, 0.25)
zi = lfilter_zi(b, a)
y, zo = lfilter(b, a, ones(10), zi=zi)
print(y)


from numpy import array, ones
from scipy.signal import lfilter, lfilter_zi, butter
b, a = butter(5, 0.25)
zi = lfilter_zi(b, a)
y = lfilter(b, a, ones(10),  )
print(y)

x = array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
y, zf = lfilter(b, a, x, zi=zi*x[0])
print(y)




#%% scipy.signal.filtfilt
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html#scipy.signal.filtfilt

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

###
t = np.linspace(0, 1.0, 2001)
xlow = np.sin(2 * np.pi * 5 * t)
xhigh = np.sin(2 * np.pi * 250 * t)
x = xlow + xhigh
b, a = signal.butter(8, 0.125)
y = signal.filtfilt(b, a, x, padlen=150)
np.abs(y - xlow).max()


###
b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied.

rng = np.random.default_rng()
n = 60
sig = rng.standard_normal(n)**3 + 3*rng.standard_normal(n).cumsum()

fgust = signal.filtfilt(b, a, sig, method="gust")
fpad = signal.filtfilt(b, a, sig, padlen=50)
plt.plot(sig, 'k-', label='input')
plt.plot(fgust, 'b-', linewidth=4, label='gust')
plt.plot(fpad, 'c-', linewidth=1.5, label='pad')
plt.legend(loc='best')
plt.show()



#%% scipy.signal.sosfiltfilt
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html#scipy.signal.sosfiltfilt

import numpy as np
from scipy.signal import sosfiltfilt, butter
import matplotlib.pyplot as plt
from scipy.signal import sosfilt, sosfilt_zi
rng = np.random.default_rng()

# Create an interesting signal to filter.
n = 201
t = np.linspace(0, 1, n)
x = 1 + (t < 0.5) - 0.25*t**2 + 0.05*rng.standard_normal(n)

# Create a lowpass Butterworth filter, and use it to filter x.
sos = butter(4, 0.125, output='sos')
y = sosfiltfilt(sos, x)

# For comparison, apply an 8th order filter using sosfilt. The filter is initialized using the mean of the first four values of x.
sos8 = butter(8, 0.125, output='sos')
zi = x[:4].mean() * sosfilt_zi(sos8)
y2, zo = sosfilt(sos8, x, zi=zi)

plt.plot(t, x, alpha=0.5, label='x(t)')
plt.plot(t, y, label='y(t)')
plt.plot(t, y2, label='y2(t)')
plt.legend(framealpha=1, shadow=True)
plt.grid(alpha=0.25)
plt.xlabel('t')
plt.show()



#%% scipy.signal.freqz
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html#scipy.signal.freqz
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

taps, f_c = 80, 1.0  # number of taps and cut-off frequency
b = signal.firwin(taps, f_c, window = ('kaiser', 8), fs = 2*np.pi)
w, h = signal.freqz(b)

fig, ax1 = plt.subplots(tight_layout = True)
ax1.set_title(f"Frequency Response of {taps} tap FIR Filter" + f"($f_c={f_c}$ rad/sample)")
ax1.axvline(f_c, color = 'black', linestyle = ':', linewidth = 0.8)
ax1.plot(w, 20 * np.log10(abs(h)), 'C0')
ax1.set_ylabel("Amplitude in dB", color = 'C0')
ax1.set(xlabel = "Frequency in rad/sample", xlim = (0, np.pi))

ax2 = ax1.twinx()
phase = np.unwrap(np.angle(h))
ax2.plot(w, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color = 'C1')
ax2.grid(True)
ax2.axis('tight')
plt.show()

fig, ax1 = plt.subplots(tight_layout = True)
ax1.set_title(f"Frequency Response of {taps} tap FIR Filter" + f"($f_c={f_c}$ rad/sample)")
# ax1.axvline(f_c, color = 'black', linestyle = ':', linewidth = 0.8)
# ax1.plot(w, 20 * np.log10(abs(h)), 'C0')
ax1.plot(0.5 * fs * w / np.pi, 20 * np.log10(abs(h)), "b")

ax1.set_ylabel("Amplitude in dB", color = 'C0')
plt.xlabel("Frequency [Hz]")
plt.show()





















































































































































































































































































































































































































































































































































































































































