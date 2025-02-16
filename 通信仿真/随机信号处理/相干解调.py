#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 01:10:22 2025

@author: jack

相干解调的关键是锁相环，这里的代码只是粗浅的仿真，只是表面的结果是对的，至于是否真的是按正确的理论实现的还得进一步学习以确认。搞清楚相干解调的前提是吃透数字锁相环。

"""
import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

## my lib
from Xcorrs import xcorr, correlate_maxlag, correlate_template, get_lags

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12

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

#%% 例2：信号调制与解调, 幅度调制（AM）: 调幅信号的解调，
# 幅度调制（AM）:
# 相干解调

fs = 500
dt = 1/fs
T = 2
fm = 3
fc = 40
t = np.arange(0, T, 1/fs)
Am = 1
at = Am * np.cos(2 * np.pi * fm * t )
# ct = scipy.signal.chirp(t, 20, t[-1], 80)
ct = np.cos(2 * np.pi * fc * t)            # 载波
x = at * ct + 0.01 * np.random.randn(t.size)        # 信号

s = x * ct
#================================ IIR -巴特沃兹低通滤波器  =====================================
lf = 6    # 通带截止频率200Hz
Fc = 400   # 阻带截止频率1000Hz
Rp = 1      # 通带波纹最大衰减为1dB
Rs = 40     # 阻带衰减为40dB
#-----------------------计算最小滤波器阶数-----------------------------
na = np.sqrt(10**(0.1*Rp)-1)
ea = np.sqrt(10**(0.1*Rs)-1)
order  = np.ceil(np.log10(ea/na)/np.log10(Fc/lf))  #巴特沃兹阶数
Wn = lf*2/fs
#---------------------- 低通滤波  -----------------------------
### 方法1
[Bb, Ba] = scipy.signal.butter(order, Wn, 'low')
# y = scipy.signal.lfilter(Bb, Ba, s) # 进行滤波
y = scipy.signal.filtfilt(Bb, Ba, s) * 2

##### plot
fig, axs = plt.subplots(4, 1, figsize = (8, 8), constrained_layout = True)

# x
axs[0].plot(t, at, color = 'b', lw = 1, label = '原始波形')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("信息信号")
axs[0].legend()

axs[1].plot(t, ct, color = 'b', lw = 1, label = '载波信号')
axs[1].set_xlabel('时间 (s)',)
axs[1].set_ylabel('幅度',)
axs[1].set_title("载波信号")
axs[1].legend()

axs[2].plot(t, x, color = 'b', lw = 1, label = '幅度调制信号')
# axs[2].plot(t, np.real(z), color = 'g', ls = '--', lw = 0.5, label = 'Real(z[n])')
# axs[2].plot(t, np.imag(z), color = 'r', ls = '--', lw = 0.5, label = 'Imag(z[n])')
axs[2].set_xlabel('时间 (s)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("幅度调制信号")
axs[2].legend()

axs[3].plot(t, y, color = 'r', lw = 2, label = '解调信号')
axs[3].plot(t, at, color = 'b', lw = 1, label = '原始波形')
axs[3].set_xlabel('时间 (s)',)
axs[3].set_ylabel('幅度',)
axs[3].set_title("解调信号")
axs[3].legend()

plt.show()
plt.close()


#%% 相位调制 (PM), 相干解调,
fc = 40         # 载波频率 (Hz)
fm = 3          # 调制信号频率 (Hz)
Ac = 1          # 载波幅度
alpha = 1       # 信号幅度
theta = 0       # 信号初始相位
beta = 0        # 载波初始相位
fs = 500        # 采样频率 (Hz)
receiverKnowsCarrier = False
T = 3
t = np.arange(0, T, 1/fs)

mt = alpha * np.cos(2 * np.pi * fm * t + theta) # 信息承载信号
ct = Ac * np.cos(2 * np.pi * fc * t + beta)
x = Ac * np.cos(2 * np.pi * fc * t + beta + mt)  # 已调信号

nMean = 0
nSigma = 0.01
n = nMean + nSigma * np.random.randn(t.size)
r = x + n

## PLL 锁相环相干解调, 相干解调法(同步解调),用锁相环同步
L = t.size
vco_phase = np.zeros(L) # 初始化vco相位
rt = np.zeros(L)        # 初始化压控振荡器vco输出
et = np.zeros(L)        # 初始化乘法鉴相器pd输出
vt = np.zeros(L)        # 初始化环路滤波器lf输出
Av = 1                  # vco输出幅度
kv = 100                # vco频率灵敏度
km = 1                  # 鉴相器增益pd增益
k0 = 1                  # lf增益
rt[0] = Av * np.cos(vco_phase[0])  #
et[0] = km * r[0] * rt [0]         #

#================================ IIR -巴特沃兹低通滤波器  =====================================
lf = 10     # 通带截止频率200Hz
Fc = 200    # 阻带截止频率1000Hz
Rp = 1      # 通带波纹最大衰减为1dB
Rs = 40     # 阻带衰减为40dB
#-----------------------计算最小滤波器阶数-----------------------------
na = np.sqrt(10**(0.1 * Rp) - 1)
ea = np.sqrt(10**(0.1 * Rs) - 1)
order = np.ceil(np.log10(ea/na)/np.log10(Fc/lf))  #巴特沃兹阶数
Wn = lf * 2 / fs
#------- 低通滤波 ---------
[Bb, Ba] = scipy.signal.butter(order, Wn, 'low')
vt = scipy.signal.filtfilt(Bb, Ba, et)

# vt[0] = k0 * b0 * et[0]
for i in range(1, L):
    vco_phase_change = 2 * np.pi * fc * dt + 2 * np.pi * kv * vt[i-1] * dt
    vco_phase[i] = vco_phase[i-1] + vco_phase_change

    rt[i] = Av * np.cos(vco_phase[i]) # vco输出（会跟踪st的相位）
    et[i] = km * rt[i] * r[i]         # 乘法鉴相器输出，式(16)

    # vt = scipy.signal.lfilter(Bb, Ba, et) * 2 # 进行滤波
    vt = scipy.signal.filtfilt(Bb, Ba, et)
    # vt[i] = k0 * (b0 * et[i] + b1 * et[i-1] - a1 * vt[i-1])

## 接下来根据VCO锁定的信号恢复出信息承载信号，这是PLL在这和FM中的不同之处，这里多个这个步骤
z = scipy.signal.hilbert(rt)
inst_amplitude = np.abs(z) # instantaneous amplitude
inst_phase = np.unwrap(np.angle(z)) # instantaneous phase, \phi(t)

if receiverKnowsCarrier:
    offsetTerm = 2 * np.pi * fc * t + beta
else:
    p = np.polyfit(t, inst_phase, 1)
    offsetTerm = np.polyval(p, t)

demodulated = inst_phase - offsetTerm  # alpha* sin(2*pi*fm*t+theta) = \phi(t) - 2*pi*fc*t - beta

##### plot
fig, axs = plt.subplots(6, 1, figsize = (8, 12), constrained_layout = True)
labelsize = 20

axs[0].plot(t, mt, color = 'b', lw = 2, label = '原始波形 (时域)')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("原始波形 (时域)")
axs[0].legend()

axs[1].plot(t , x, color = 'b', lw = 0.5, label = '相位调制PM信号')
axs[1].set_xlabel('时间 (s)',)
axs[1].set_ylabel('幅度',)
axs[1].set_title("相位调制PM信号")
axs[1].legend()

axs[2].plot(t, r, color = 'b', lw = 0.2, label = '接收信号(时域)')
axs[2].set_xlabel('时间 (s)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("接收信号(时域)")
axs[2].legend()

axs[3].plot(t, demodulated, color = 'r', label = '解调信号 (时域)')
axs[3].plot(t, mt, color = 'b', lw = 2, ls='--', label = '原始波形 (时域)')
axs[3].set_xlabel('时间 (s)',)
axs[3].set_ylabel('幅度',)
axs[3].set_title("解调信号(时域)")
axs[3].legend()

axs[4].plot(t, inst_amplitude, color = 'b', lw = 0.5, label = '提取的包络')
axs[4].set_xlabel('时间 (s)',)
axs[4].set_ylabel('幅度',)
axs[4].set_title("提取的包络")
axs[4].legend()

axs[5].plot(t, rt, color = 'b', lw = 0.2, label = 'vco信号 (时域)')
axs[5].set_xlabel('时间 (s)',)
axs[5].set_ylabel('幅度',)
axs[5].set_title("vco信号 (时域)")
axs[5].legend()

plt.show()
plt.close()



#%% 频率调制（FM）是一种广泛应用于广播和通信系统的调制方式。其基本概念是通过改变信号的频率来传递信息。
# 相干解调
import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import hilbert

# 参数设置
fs = 500  # 采样频率
dt = 1/fs
T  = 3
t  = np.arange(0, T, 1/fs)  # 时间向量
fc = 40  # 载波频率
Ac = 1    # 载波幅度
kf = 20  # 频率偏移常数, 这个参数相当重要，直接决定解调的效果，需要学习一下确定这个参数的方法
Am = 1  # 调制信号幅度
fm = 3   # 调制信号频率

# 调制信号（假设为正弦波）
mt = Am * np.cos(2 * np.pi * fm * t)

# 频率调制
ct = Ac * np.cos(2 * np.pi * fc * t)  # 载波信号
x = Ac * np.cos(2 * np.pi * fc * t +  2 * np.pi * kf * np.cumsum(mt) * dt )

## PLL 锁相环相干解调, 相干解调法(同步解调),用锁相环同步
L = t.size
vco_phase = np.zeros(L) # 初始化vco相位
rt = np.zeros(L)        # 初始化压控振荡器vco输出
et = np.zeros(L)        # 初始化乘法鉴相器pd输出
vt = np.zeros(L)        # 初始化环路滤波器lf输出
Av = 1                  # vco输出幅度
kv = 100                # vco频率灵敏度
km = 1                  # 鉴相器增益pd增益
k0 = 1                  # lf增益
rt[0] = Av * np.cos(vco_phase[0])  #
et[0] = km * x[0] * rt [0]         #

# [Bb, Ba] = scipy.signal.butter(1, 2 * 4 * fm / fs, 'low')
# b0 = Bb[0]
# b1 = Bb[1]
# a1 = Ba[1]
## b0 = 0.07295965726826667;       # Fs = 40000, fcut = 1000的1阶巴特沃斯低通滤波器系数, 由FDA生成
## b1 = 0.07295965726826667;
## a1 = -0.8540806854634666;
#================================ IIR -巴特沃兹低通滤波器  =====================================
lf = 10     # 通带截止频率200Hz
Fc = 200    # 阻带截止频率1000Hz
Rp = 1      # 通带波纹最大衰减为1dB
Rs = 40     # 阻带衰减为40dB
#-----------------------计算最小滤波器阶数-----------------------------
na = np.sqrt(10**(0.1 * Rp) - 1)
ea = np.sqrt(10**(0.1 * Rs) - 1)
order = np.ceil(np.log10(ea/na)/np.log10(Fc/lf))  #巴特沃兹阶数
Wn = lf * 2 / fs
#------- 低通滤波 ---------
[Bb, Ba] = scipy.signal.butter(order, Wn, 'low')
vt = scipy.signal.filtfilt(Bb, Ba, et)

# vt[0] = k0 * b0 * et[0]
for i in range(1, L):
    vco_phase_change = 2 * np.pi * fc * dt + 2 * np.pi * kv * vt[i-1] * dt
    vco_phase[i] = vco_phase[i-1] + vco_phase_change

    rt[i] = Av * np.cos(vco_phase[i]) # vco输出（会跟踪st的相位）
    et[i] = km * rt[i] * x[i]         # 乘法鉴相器输出，式(16)

    # vt = scipy.signal.lfilter(Bb, Ba, et) * 2 # 进行滤波
    vt = scipy.signal.filtfilt(Bb, Ba, et)

    # vt[i] = k0 * (b0 * et[i] + b1 * et[i-1] - a1 * vt[i-1])

# 绘制调制信号和FM信号
fig, axs = plt.subplots(5, 1, figsize = (8, 10), constrained_layout = True)

axs[0].plot(t, mt, color = 'b', lw = 2, label = '原始波形 (时域)')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("原始波形 (时域)")
axs[0].legend()

axs[1].plot(t, ct, color = 'b', lw = 0.5, label = '载波信号')
axs[1].set_xlabel('时间 (s)',)
axs[1].set_ylabel('幅度',)
axs[1].set_title("载波信号 (时域)")
axs[1].legend()

axs[2].plot(t, x, color = 'b', lw = 0.2, label = '频率调制信号')
axs[2].set_xlabel('时间 (s)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("频率调制信号")
axs[2].legend()

axs[3].plot(t, rt, color = 'b', lw = 0.2, label = 'vco信号 (时域)')
axs[3].set_xlabel('时间 (s)',)
axs[3].set_ylabel('幅度',)
axs[3].set_title("vco信号 (时域)")
axs[3].legend()

axs[4].plot(t, vt * 2, color = 'b', label = '解调信号(时域)')
axs[4].plot(t, mt, color = 'r', label = '原始信号 (时域)')
axs[4].set_xlabel('时间 (s)',)
axs[4].set_ylabel('幅度',)
axs[4].set_title("解调信号(时域)")
axs[4].legend()

plt.show()
plt.close()

# https://www.cnblogs.com/gjblog/p/13494103.html#
# https://blog.csdn.net/weixin_42553916/article/details/122225988
#%% 频率调制（FM）是一种广泛应用于广播和通信系统的调制方式。其基本概念是通过改变信号的频率来传递信息。
# 相干解调
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
fs = 500  # 采样频率
dt = 1/fs
T  = 3
t  = np.arange(0, T, dt)  # 时间向量
fc = 40    # 载波频率
Ac = 1     # 载波幅度
kf = 20    # 频率偏移常数,频偏常数, 表示调频器的调频灵敏度. 这个参数相当重要，直接决定解调的效果，需要学习一下确定这个参数的方法
Am = 1     # 调制信号幅度
fm = 3     # 调制信号频率

# 调制信号（假设为正弦波）
mt = Am * np.cos(2 * np.pi * fm * t)
# 频率调制
ct = Ac * np.cos(2 * np.pi * fc * t)  # 载波信号
x = Ac * np.cos(2 * np.pi * fc * t +   kf * Am * np.sin(2 * np.pi * fm * t) / (fm) ) # + 0.01 * np.random.randn(t.size)

## PLL 锁相环相干解调, 相干解调法(同步解调),用锁相环同步
L = t.size
vco_phase = np.zeros(L) # 初始化vco相位
rt = np.zeros(L)        # 初始化压控振荡器vco输出
et = np.zeros(L)        # 初始化乘法鉴相器pd输出
vt = np.zeros(L)        # 初始化环路滤波器lf输出
Av = 1                  # vco输出幅度
kv = 100                # vco频率灵敏度
km = 1                  # 鉴相器增益pd增益
k0 = 1                  # lf增益
rt[0] = Av * np.cos(vco_phase[0])  #
et[0] = km * x[0] * rt [0]         #

# [Bb, Ba] = scipy.signal.butter(1, 2 * 4 * fm / fs, 'low')
# b0 = Bb[0]
# b1 = Bb[1]
# a1 = Ba[1]
## b0 = 0.07295965726826667;       # Fs = 40000, fcut = 1000的1阶巴特沃斯低通滤波器系数, 由FDA生成
## b1 = 0.07295965726826667;
## a1 = -0.8540806854634666;
#================================ IIR -巴特沃兹低通滤波器  =====================================
lf = 10     # 通带截止频率200Hz
Fc = 200    # 阻带截止频率1000Hz
Rp = 1      # 通带波纹最大衰减为1dB
Rs = 40     # 阻带衰减为40dB
#-----------------------计算最小滤波器阶数-----------------------------
na = np.sqrt(10**(0.1 * Rp) - 1)
ea = np.sqrt(10**(0.1 * Rs) - 1)
order = np.ceil(np.log10(ea/na)/np.log10(Fc/lf))  #巴特沃兹阶数
Wn = lf * 2 / fs
#------- 低通滤波 ---------
[Bb, Ba] = scipy.signal.butter(order, Wn, 'low')
vt = scipy.signal.filtfilt(Bb, Ba, et)

# vt[0] = k0 * b0 * et[0]
for i in range(1, L):
    vco_phase_change = 2 * np.pi * fc * dt + 2 * np.pi * kv * vt[i-1] * dt
    vco_phase[i] = vco_phase[i-1] + vco_phase_change

    rt[i] = Av * np.cos(vco_phase[i]) # vco输出（会跟踪st的相位）
    et[i] = km * rt[i] * x[i]         # 乘法鉴相器输出，式(16)

    # vt = scipy.signal.lfilter(Bb, Ba, et) * 2 # 进行滤波
    vt = scipy.signal.filtfilt(Bb, Ba, et)

    # vt[i] = k0 * (b0 * et[i] + b1 * et[i-1] - a1 * vt[i-1])

# 绘制调制信号和FM信号
fig, axs = plt.subplots(5, 1, figsize = (8, 10), constrained_layout = True)

axs[0].plot(t, mt, color = 'b', lw = 2, label = '原始波形 (时域)')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("原始波形 (时域)")
axs[0].legend()

axs[1].plot(t, ct, color = 'b', lw = 0.5, label = '载波信号')
axs[1].set_xlabel('时间 (s)',)
axs[1].set_ylabel('幅度',)
axs[1].set_title("载波信号 (时域)")
axs[1].legend()

axs[2].plot(t, x, color = 'b', lw = 0.2, label = '频率调制信号')
axs[2].set_xlabel('时间 (s)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("频率调制信号")
axs[2].legend()

axs[3].plot(t, rt, color = 'b', lw = 0.2, label = 'vco信号 (时域)')
axs[3].set_xlabel('时间 (s)',)
axs[3].set_ylabel('幅度',)
axs[3].set_title("vco信号 (时域)")
axs[3].legend()

axs[4].plot(t, vt * 2, color = 'b', label = '解调信号(时域)')
axs[4].plot(t, mt, color = 'r', label = '原始信号 (时域)')
axs[4].set_xlabel('时间 (s)',)
axs[4].set_ylabel('幅度',)
axs[4].set_title("解调信号(时域)")
axs[4].legend()

plt.show()
plt.close()















































