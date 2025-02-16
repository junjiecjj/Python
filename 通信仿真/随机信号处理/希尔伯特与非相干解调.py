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

# 采用非相干解调，也称为包络检测技术。这要求基带信号为非负，可以通过在原基带信号中叠加一个直流信号实现。例如：
# BOOK <wireless communication system in Matlab >
#%% Applications of analytic signal: Extracting instantaneous amplitude, phase, frequency
# 幅度调制（AM）:非相干解调
# 对于AM，解调的关键是提取出包络（包络就是信息信号），而恢复出瞬时相位就可以得到载波信号，瞬时频率就是载波频率
# 对于PM，关键是提取出瞬时相位，瞬时相位减去载波相位就是信息信号，而瞬时幅度就是载波幅度.
fs = 500
dt = 1/fs
T = 1
fm = 3
fc = 40
t = np.arange(0, T, 1/fs)
Am = 0.5
at = 1 + Am * np.sin(2 * np.pi * fm * t + np.pi/4)
# ct = scipy.signal.chirp(t, 20, t[-1], 80)
ct = np.sin(2 * np.pi * fc * t+ np.pi/3)            # 载波
x = at * ct + 0.01 * np.random.randn(t.size)        # 信号
# z = analytic_signal(x)
z = scipy.signal.hilbert(x)
inst_amplitude = np.abs(z)                            # instantaneous amplitude, 取出包络 a(t)
inst_phase = np.unwrap(np.angle(z))                   # instantaneous phase, 恢复出瞬时相位 Phi(t)
# inst_freq = np.diff(inst_phase) * fs / (2 * np.pi)  # instantaneous temporal frequency, 瞬时频率 ~= fc, 这里乘以fs是因为求导数时要把时间的尺度作为分母，而时间尺度就是1/fs,除以1/fs等于乘以fs
inst_freq = np.diff(inst_phase) / dt / (2 * np.pi)    # instantaneous temporal frequency, 瞬时频率 ~= fc,
regenerated_carrier = np.cos(inst_phase)              # 载波恢复

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
# 非相干解调
fs = 500                    # 采样频率 (Hz)
dt = 1/fs
T = 1
t = np.arange(0, T, 1/fs)    # 时间向量

fc = 40                     # 载波频率 (Hz)
Ac = 1                       # 载波信号幅度, 一般为1，如果不为1， 则后面的幅度要注意
fm = 3                      # 调制信号频率 (Hz)
Am = 0.5                     # 调制信号幅度

m = 1 + Am * np.cos(2 * np.pi * fm * t + np.pi/4) # 信号
c = Ac * np.cos(2 * np.pi * fc * t + np.pi/3)     # 载波
s = m * c + 0.001 * np.random.randn(m.size)

z = scipy.signal.hilbert(s)
s_demod = np.abs(z) / Ac # inst_amplitude

inst_phase = np.unwrap(np.angle(z)) # instantaneous phase
# inst_freq = np.diff(inst_phase) / (2 * np.pi) * fs # instantaneous angular, 这里乘以fs是因为求导数时要把时间的尺度作为分母，而时间尺度就是1/fs,除以1/fs等于乘以fs
inst_freq = np.diff(inst_phase) / (2 * np.pi) / dt # instantaneous angular,
regenerated_carrier = np.cos(inst_phase) * Ac

##### plot
fig, axs = plt.subplots(6, 1, figsize = (8, 12), constrained_layout = True)

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

# BOOK <wireless communication system in Matlab >
#%% Applications of analytic signal: Phase demodulation (PM) using Hilbert transform
## 相位调制 (PM), 非相干解调
fc = 40         # 载波频率 (Hz)
fm = 3          # 调制信号频率 (Hz)
Ac = 1.5         # 载波幅度
alpha = 2        # 信号幅度
theta = np.pi/4  # 信号初始相位
beta = np.pi/5   # 载波初始相位
receiverKnowsCarrier = False
fs = 500       # 采样频率 (Hz)
T = 1
t = np.arange(0, T, 1/fs)

mt = alpha * np.sin(2 * np.pi * fm * t + theta) # 信息承载信号
x = Ac * np.cos(2 * np.pi * fc * t + beta + mt)  # 已调信号

nMean = 0
nSigma = 0.01
n = nMean + nSigma * np.random.randn(t.size)
r = x + n

z = scipy.signal.hilbert(r)
inst_amplitude = np.abs(z) # instantaneous amplitude
inst_phase = np.unwrap(np.angle(z)) # instantaneous phase, \phi(t)

if receiverKnowsCarrier:
    offsetTerm = 2 * np.pi * fc * t + beta
else:
    p = np.polyfit(t, inst_phase, 1)
    offsetTerm = np.polyval(p, t)

demodulated = inst_phase - offsetTerm  # alpha* sin(2*pi*fm*t+theta) = \phi(t) - 2*pi*fc*t - beta

##### plot
fig, axs = plt.subplots(5, 1, figsize = (8, 12), constrained_layout = True)
labelsize = 20

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
# 非相干解调
import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import hilbert

# 参数设置
fs = 500  # 采样频率
dt = 1/fs
T  = 1
t  = np.arange(0, T, 1/fs)  # 时间向量
fc = 40  # 载波频率
Ac = 2    # 载波幅度
kf = 10  # 频率偏移常数, 这个参数相当重要，直接决定解调的效果，需要学习一下确定这个参数的方法
Am = 1.5  # 调制信号幅度
fm = 3   # 调制信号频率

# 调制信号（假设为正弦波）
mt = Am * np.cos(2 * np.pi * fm * t)

# 频率调制
ct = Ac * np.cos(2 * np.pi * fc * t)  # 载波信号
x = Ac * np.cos(2 * np.pi * fc * t +  2 * np.pi * kf * np.cumsum(mt) * dt )

# 频率解调
# 使用希尔伯特变换提取瞬时频率
z = scipy.signal.hilbert(x)
inst_amplitude = np.abs(z) # instantaneous amplitude
inst_phase = np.unwrap(np.angle(z)) # instantaneous phase
inst_freq = np.diff(inst_phase) / (2 * np.pi * dt)
mt_hat = (inst_freq - fc) / kf

# 为了对齐时间向量，去掉最后一个点
t_demod = t[:-1]

# 绘制调制信号和FM信号
fig, axs = plt.subplots(5, 1, figsize = (8, 10), constrained_layout = True)

# x
axs[0].plot(t, mt, color = 'b', lw = 2, label = '原始波形 (时域)')
axs[0].set_xlabel('时间 (s)',)
axs[0].set_ylabel('幅度',)
axs[0].set_title("原始波形 (时域)")
axs[0].legend()

axs[1].plot(t[:500], ct[:500], color = 'b', lw = 0.5, label = '载波信号')
axs[1].set_xlabel('时间 (s)',)
axs[1].set_ylabel('幅度',)
axs[1].set_title("载波信号 (时域)")
axs[1].legend()

axs[2].plot(t, x, color = 'b', lw = 0.2, label = '已调信号 (AM, 时域)')
axs[2].set_xlabel('时间 (s)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("已调信号 (幅度调制AM, 时域)")
axs[2].legend()

axs[3].plot(t_demod, mt_hat, color = 'b', label = '解调信号 (时域)')
axs[3].set_xlabel('时间 (s)',)
axs[3].set_ylabel('幅度',)
axs[3].set_title("解调信号 (时域)")
axs[3].legend()

axs[4].plot(t, inst_amplitude, color = 'b', lw = 1, label = '提取的包络')
axs[4].set_xlabel('时间 (s)',)
axs[4].set_ylabel('幅度',)
axs[4].set_title("提取的包络")
axs[4].legend()
print(f"提取的包络 = {np.abs(inst_amplitude).mean()}")
plt.show()
plt.close()

# https://blog.csdn.net/zhouxiangjun11211/article/details/71172164
# https://www.cnblogs.com/gjblog/p/13494103.html#
# https://blog.csdn.net/zhouxiangjun11211/article/details/71172164
# https://blog.51cto.com/u_16213673/7540906
# https://blog.csdn.net/weixin_42553916/article/details/122225988
#%% 频率调制（FM）是一种广泛应用于广播和通信系统的调制方式。其基本概念是通过改变信号的频率来传递信息。
# 非相干解调
import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import hilbert

# 参数设置
fs = 500  # 采样频率
dt = 1/fs
T  = 2
t  = np.arange(0, T, dt)  # 时间向量
fc = 40  # 载波频率
Ac = 2    # 载波幅度
kf = 20  # 频率偏移常数,频偏常数, 表示调频器的调频灵敏度. 这个参数相当重要，直接决定解调的效果，需要学习一下确定这个参数的方法
Am = 1.5  # 调制信号幅度
fm = 3   # 调制信号频率

# 调制信号（假设为正弦波）
mt = Am * np.cos(2 * np.pi * fm * t)
# 频率调制
ct = Ac * np.cos(2 * np.pi * fc * t)  # 载波信号
# https://blog.csdn.net/weixin_42553916/article/details/122225988
x = Ac * np.cos(2 * np.pi * fc * t +   kf * Am * np.sin(2 * np.pi * fm * t) / (fm) ) # + 0.01 * np.random.randn(t.size)

# 非相干解调
x_diff = np.diff(x) / dt
x_diff = np.hstack((np.array([0]), x_diff))
# 使用希尔伯特变换提取瞬时频率
z = scipy.signal.hilbert(x_diff)
inst_amplitude = np.abs(z) # instantaneous amplitude
mt_hat = (inst_amplitude / Ac - 2 * np.pi * fc) / (2 * np.pi * kf)

# 为了对齐时间向量，去掉最后一个点
# t_demod = t[:-1]

# 绘制调制信号和FM信号
fig, axs = plt.subplots(4, 1, figsize = (8, 8), constrained_layout = True)

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

axs[2].plot(t, x, color = 'b', lw = 0.2, label = '已调信号 (AM, 时域)')
axs[2].set_xlabel('时间 (s)',)
axs[2].set_ylabel('幅度',)
axs[2].set_title("已调信号 (幅度调制AM, 时域)")
axs[2].legend()

axs[3].plot(t, mt_hat, color = 'b', label = '解调信号 (时域)')
axs[3].set_xlabel('时间 (s)',)
axs[3].set_ylabel('幅度',)
axs[3].set_title("解调信号 (时域)")
axs[3].legend()

plt.show()
plt.close()



























