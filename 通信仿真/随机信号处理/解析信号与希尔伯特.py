#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 01:10:22 2025

@author: jack
实值信号的傅里叶变换是复对称的。这意味着负频率的内容相对于正频率是冗余的。在Gabor[12]和Ville[13]的工作中，旨在通过去除傅立叶变换产生的冗余负频率内容来创建一个分析信号。
解析信号是复值信号，但其频谱是单侧的（只有正频率），保留了原始实值信号的频谱内容。用解析信号代替原来的实值信号，已被证明是有用的.

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


#%% Check and investigate components of an analytic signal
def analytic_signal(x):
    # Generate analytic signal using frequency domain approach
    x = x[:]
    N = x.size
    X = np.fft.fft(x, n = N)
    spectrum = np.hstack((X[0], 2*X[1:int(N/2)],X[int(N/2)+1], np.zeros(int(N/2)-1)))
    z = np.fft.ifft(spectrum, n = N)
    return z

fs = 100
T = 1
t = np.arange(0, T, 1/fs)
f0 = 10
x = 2 * np.sin(2 * np.pi * f0 * t)
# z = analytic_signal(x)
## 或者使用自带的库，返回x的解析信号，完全等价
z = scipy.signal.hilbert(x)
x_hilbert = np.imag(z) ## 解析信号的虚部才是x的hilbert变换结果

## 验证解析信号的频谱只是对应实值信号的正半部分. 解析信号是复值信号，但其频谱是单侧的（只有正频率），保留了原始实值信号的频谱内容
FFTN = x.size
fx, Yx, Ax, Phax, Rx, Ix = freqDomainView(x, fs, FFTN, type = 'double')
fz, Yz, Az, Phaz, Rz, Iz = freqDomainView(z, fs, FFTN, type = 'double')


##### plot
fig, axs = plt.subplots(4, 1, figsize = (8, 10), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, x, color = 'b', lw = 2, label = 'x[n]')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("x[n]")
axs[0].legend()

axs[1].plot(t, np.real(z), color = 'k', label = 'Real(z[n])')
axs[1].plot(t, np.imag(z), color = 'r', label = 'Imag(z[n]):x的hilbert变换')
axs[1].set_xlabel('时间 (s)',)
axs[1].set_ylabel('幅度',)
axs[1].set_title("z[n]")
axs[1].legend()

axs[2].plot(fx, Ax, color = 'k', label = 'FFT(x)')
# axs[2].plot(t, np.imag(z), color = 'r', label = 'Imag(z[n])')
axs[2].set_xlabel('频率 (Hz)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("原信号x的频谱")
axs[2].legend()

axs[3].plot(fz, Az, color = 'k', label = 'FFT(z)')
# axs[2].plot(t, np.imag(z), color = 'r', label = 'Imag(z[n])')
axs[3].set_xlabel('频率 (Hz)',)
axs[3].set_ylabel('幅度',)
axs[3].set_title("原信号x对应的解析信号z的频谱")
axs[3].legend()

plt.show()
plt.close()


#%%  np.unwrap
import numpy as np

l1 =[1, 2, 3, 4, 5]
print("Result 1: ", np.unwrap(l1))
# Result 1:  [1. 2. 3. 4. 5.]

l2 =[0, 0.78, 5.49, 6.28]
print("Result 2: ", np.unwrap(l2))
# Result 2:  [ 0., 0.78, -0.79318531, -0.00318531]

l1 = [5, 7, 10, 15, 19, 25, 32]
print("Result 1: ", np.unwrap(l1,))
# Result 1:  array([ 5. ,  7. , 10. , 8.71681469, 6.43362939, 6.15044408,  6.86725877])

l = [0, 45, 90, 135, 180, 225, 270, 315, 360, 405]
print(np.unwrap(l, period = 360))

l = [0, 45, 270, 300, 100, -100, -110, -300, -10, 45, -10, 200, 360, 700]
print(np.unwrap(l, period = 360))


#%% Applications of analytic signal: Extracting instantaneous amplitude, phase, frequency
# 幅度调制（AM）
fs = 600
t = np.arange(0, 1, 1/fs)
at = 1 + 0.5 * np.sin(2 * np.pi * 3 * t)
ct = scipy.signal.chirp(t, 20, t[-1], 80)
# ct = np.sin(2 * np.pi * 40 * t)
x = at * ct + 0.01 * np.random.randn(t.size)
z = analytic_signal(x)
# z = scipy.signal.hilbert(x)
inst_amplitude = np.abs(z)
inst_phase = np.unwrap(np.angle(z))
inst_freq = np.diff(inst_phase) / (2 * np.pi) * fs

regenerated_carrier = np.cos(inst_phase)

##### plot
fig, axs = plt.subplots(6, 1, figsize = (8, 12), constrained_layout = True)

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

axs[2].plot(t, x, color = 'b', lw = 1, label = '已调信号')
axs[2].plot(t, np.real(z), color = 'g', ls = '--', lw = 0.5, label = 'Real(z[n])')
axs[2].plot(t, np.imag(z), color = 'r', ls = '--', lw = 0.5, label = 'Imag(z[n])')
axs[2].set_xlabel('时间 (s)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("已调信号")
axs[2].legend()

axs[3].plot(t, inst_amplitude, color = 'r', lw = 2, label = '解调信号')
axs[3].set_xlabel('时间 (s)',)
axs[3].set_ylabel('幅度',)
axs[3].set_title("解调信号")
axs[3].legend()

axs[4].plot(t, regenerated_carrier, color = 'b', label = '载波恢复')
axs[4].set_xlabel('时间 (s)',)
axs[4].set_ylabel('cos[w(t)]',)
axs[4].set_title("载波恢复")
axs[4].legend()

axs[5].plot(t[:-1], inst_freq, color = 'r', label = '瞬时时间频率')
axs[5].set_xlabel('时间 (s)',)
axs[5].set_ylabel('Hz',)
axs[5].set_title("瞬时时间频率")
axs[5].legend()

plt.show()
plt.close()


#%% 例2：信号调制与解调, 幅度调制（AM）: 调幅信号的解调，涉及调幅信号包络的提取

fs = 1000;                   # 采样频率 (Hz)
T = 1
t = np.arange(0, T, 1/fs)    # 时间向量

fc = 100;                   # 载波频率 (Hz)
fm = 10;                    # 调制信号频率 (Hz)
Am = 1                     # 调制信号幅度
Ac = 1                      # 载波信号幅度

m = 1 + Am * np.cos(2 * np.pi * fm * t) # 信号
c = Ac * np.cos(2 * np.pi * fc * t) # 载波
s = m * c + 0.001 * np.random.randn(m.size)

z = scipy.signal.hilbert(s)
s_demod = np.abs(z) # inst_amplitude

inst_phase = np.unwrap(np.angle(z))
inst_freq = np.diff(inst_phase) / (2 * np.pi) * fs
regenerated_carrier = np.cos(inst_phase)

##### plot
fig, axs = plt.subplots(6, 1, figsize = (8, 12), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, m, color = 'b', lw = 2, label = '原始波形 (时域)')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("原始波形 (时域)")
axs[0].legend()

axs[1].plot(t[:500], c[:500], color = 'b', lw = 0.5, label = '载波信号')
axs[1].set_xlabel('时间 (s)',)
axs[1].set_ylabel('幅度',)
axs[1].set_title("载波信号 (时域)")
axs[1].legend()

axs[2].plot(t, s, color = 'b', lw = 0.2, label = '已调信号 (AM, 时域)')
axs[2].set_xlabel('时间 (s)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("已调信号 (幅度调制AM, 时域)")
axs[2].legend()

axs[3].plot(t, s_demod, color = 'b', label = '解调信号 (时域)')
axs[3].set_xlabel('时间 (s)',)
axs[3].set_ylabel('幅度',)
axs[3].set_title("解调信号 (时域)")
axs[3].legend()

axs[4].plot(t[:500], regenerated_carrier[:500], color = 'b', lw = 0.5, label = '载波恢复')
axs[4].set_xlabel('时间 (s)',)
axs[4].set_ylabel('幅度',)
axs[4].set_title("载波恢复")
axs[4].legend()

axs[5].plot(t[:-1], inst_freq, color = 'r', label = '瞬时时间频率')
axs[5].set_xlabel('时间 (s)',)
axs[5].set_ylabel('Hz',)
axs[5].set_title("瞬时时间频率")
axs[5].legend()

plt.show()
plt.close()


#%% Applications of analytic signal: Phase demodulation (PM) using Hilbert transform
fc = 210
fm = 10
A = 1.5
alpha = 2
theta = np.pi/4
beta = np.pi/5
receiverKnowsCarrier = False
fs = 4*fc
T = 1
t = np.arange(0, T, 1/fs)

mt = alpha * np.sin(2 * np.pi * fm * t + theta)
x = A * np.cos(2 * np.pi * fc * t + beta + mt)

nMean = 0
nSigma = 0.01
n = nMean + nSigma * np.random.randn(t.size)
r = x + n

z = scipy.signal.hilbert(r)
inst_amplitude = np.abs(z)

inst_phase = np.unwrap(np.angle(z))

if receiverKnowsCarrier:
    offsetTerm = 2 * np.pi * fc * t + beta
else:
    p = np.polyfit(t, inst_phase, 1)
    offsetTerm = np.polyval(p, t)

demodulated = inst_phase - offsetTerm
mt_hat = np.cos(demodulated)

##### plot
fig, axs = plt.subplots(5, 1, figsize = (8, 12), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, mt, color = 'b', lw = 2, label = '原始波形 (时域)')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("原始波形 (时域)")
axs[0].legend()

axs[1].plot(t , x, color = 'b', lw = 0.5, label = '已调信号 (PM, 时域)')
axs[1].set_xlabel('时间 (s)',)
axs[1].set_ylabel('幅度',)
axs[1].set_title("已调信号 (PM, 时域)")
axs[1].legend()

axs[2].plot(t, r, color = 'b', lw = 0.2, label = '接收信号(时域)')
axs[2].set_xlabel('时间 (s)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("接收信号(时域)")
axs[2].legend()

axs[3].plot(t, demodulated, color = 'b', label = '解调信号 (时域)')
axs[3].set_xlabel('时间 (s)',)
axs[3].set_ylabel('幅度',)
axs[3].set_title("解调信号(时域)")
axs[3].legend()

axs[4].plot(t, inst_amplitude, color = 'b', lw = 0.5, label = '提取的包络')
axs[4].set_xlabel('时间 (s)',)
axs[4].set_ylabel('幅度',)
axs[4].set_title("提取的包络")
axs[4].legend()

plt.show()
plt.close()

#%% 频率调制（FM）是一种广泛应用于广播和通信系统的调制方式。其基本概念是通过改变信号的频率来传递信息。
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
fs = 500         # 采样频率
f_signal = 20    # 基带信号频率
f_carrier = 100  # 载波频率
T = 1            # 信号时长
t = np.arange(0, T, 1/fs)

# 生成基带信号（正弦波）
baseband_signal = np.cos(2 * np.pi * f_signal * t)

# 生成FM信号
kf = 100  # 调频灵敏度
fm_signal = np.cos(2 * np.pi * f_carrier * t + kf * np.cumsum(baseband_signal) / (2 * np.pi *fs))

# 绘制信号
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, baseband_signal)
plt.title('Baseband Signal')
plt.subplot(3, 1, 2)
plt.plot(t, fm_signal)
plt.title('FM Signal')
plt.tight_layout()
plt.show()

# FM解调
def fm_demodulate(fm_signal):
    # 计算信号的瞬时相位
    z = scipy.signal.hilbert(fm_signal)
    instantaneous_phase = np.unwrap(np.angle(z))
    # 计算瞬时频率
    instantaneous_frequency = np.diff(instantaneous_phase) * (fs / (2.0 * np.pi))
    # demodulated_signal = np.concatenate(([0], instantaneous_frequency))  # 补齐长度
    return instantaneous_frequency

# 解调
demodulated_signal = fm_demodulate(fm_signal)

# 绘制解调后的信号
plt.figure(figsize=(12, 6))
plt.plot(t[1:], demodulated_signal)
plt.title('Demodulated Signal')
plt.xlabel('Time (s)')
plt.show()

#%% 频率调制（FM）
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
fs = 500  # 采样频率
t = np.arange(0, 1, 1/fs)  # 时间向量
fc = 100  # 载波频率
kf = 100  # 频率偏移常数
Am = 1  # 调制信号幅度
fm = 10  # 调制信号频率

# 调制信号（假设为正弦波）
modulating_signal = Am * np.sin(2 * np.pi * fm * t)

# 频率调制
carrier = np.cos(2 * np.pi * fc * t)  # 载波信号
fm_signal = np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * np.cumsum(modulating_signal) / fs)

# 绘制调制信号和FM信号
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, modulating_signal)
plt.title('Modulating Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(t, fm_signal)
plt.title('FM Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# 频率解调
# 计算信号的相位变化
phase = np.unwrap(np.angle(np.exp(1j * 2 * np.pi * fc * t + 1j * 2 * np.pi * kf * np.cumsum(modulating_signal) / fs)))

# 计算瞬时频率
instantaneous_frequency = np.diff(phase) * fs / (2 * np.pi)

# 为了对齐时间向量，去掉最后一个点
t_demod = t[:-1]

# 绘制解调信号
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t_demod, instantaneous_frequency)
plt.title('Demodulated Signal (Instantaneous Frequency)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.subplot(2, 1, 2)
plt.plot(t_demod, instantaneous_frequency - fc)  # 减去载波频率，得到原始调制信号
plt.title('Recovered Modulating Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()














































































































