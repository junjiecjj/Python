#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 21:30:18 2025

@author: jack

https://blog.csdn.net/qq_43485394/article/details/122655901

"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300      # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22

#%%  https://blog.csdn.net/qq_43485394/article/details/122655901
### 频域实现脉冲压缩
def rectpuls(t, remove, T):
    Ts = t[1] - t[0]
    fs = 1/Ts

    rect = (t >= -T/2) * (t <= T/2)
    # res = np.zeros(rect.size)
    K = int(remove*fs)
    rect = np.roll(rect, K)   # 循环右移

    # t = t + remove
    return rect

### parameters
c = 3e8          # 光速
f0 = 10e9        # 载波
Tp = 10e-6       # 脉冲持续时间
B = 10e6         # 带宽
k = B/Tp         # 调频斜率
fs = 100e6       # 采样频率
R0 = 3000         # 目标距离

# signal generation
N = 1024*4       # 采样点
# n = np.arange(N)
Ts = 1/fs        # 采样间隔
t = np.arange(N)*Ts
f = np.arange(-N/2, N/2) * fs/N
tau_0 = 2*R0/c   # 时延

st = rectpuls(t, Tp/2, Tp) * np.exp(1j * np.pi * k * (t-Tp/2)**2)    #  参考信号
#  回波信号
# secho = rectpuls(t, tau_0+Tp/2, Tp) * np.exp(1j * np.pi * k * (t - tau_0 - Tp/2)**2) * np.exp(-1j * 2 * np.pi * f0 * tau_0)
secho = rectpuls(t, tau_0+Tp/2, Tp) * np.exp(1j * np.pi * (k * (t - tau_0 - Tp/2)**2 + 2 * f0 * tau_0))
#=============== 频域实现脉冲压缩 ================
Xs = scipy.fft.fft(st, N);        # 本地副本的FFT
Xecho = scipy.fft.fft(secho, N);  # 输入信号的FFT
Y = np.conjugate(Xs)*Xecho;       # 乘法器
Y = scipy.fft.fftshift(Y);
y = scipy.fft.ifft(Y, N);          # IFFT
y = np.abs(y)/max(np.abs(y)) + 1e-10;

r = t*c/2;
R0_est = r[np.argmax(y)]

##### plot
fig, axs = plt.subplots(4, 2, figsize = (12, 16), constrained_layout = True)

axs[0,0].plot(t * 1e6, np.real(st), color = 'b', label = '')
axs[0,0].set_xlabel('时间/us',)
axs[0,0].set_ylabel('幅值',)
axs[0,0].set_title("Real Part of Reference Signal")

axs[0,1].plot(t * 1e6, np.imag(st), color = 'b', label = '')
axs[0,1].set_xlabel('时间/us',)
axs[0,1].set_ylabel('幅值',)
axs[0,1].set_title("Imagine Part of Reference Signal" )

axs[1,0].plot(t * 1e6, np.real(secho), color = 'b', label = '')
axs[1,0].set_xlabel('时间/us',)
axs[1,0].set_ylabel('幅值',)
axs[1,0].set_title("Real Part of Echo Signal" )

axs[1,1].plot(t * 1e6, np.imag(secho), color = 'b', label = '')
axs[1,1].set_xlabel('时间/us',)
axs[1,1].set_ylabel('幅值',)
axs[1,1].set_title("Imagine Part of Echo Signal" )


axs[2,0].plot(f/(1e6), np.abs(scipy.fft.fftshift(Xs)), color = 'b', label = '')
axs[2,0].set_xlabel('Frequency/MHz',)
axs[2,0].set_ylabel('幅值',)
axs[2,0].set_title("Spectral of Reference Signal" )

axs[2,1].plot(f/(1e6), np.abs(scipy.fft.fftshift(Xecho)), color = 'b', label = ' ')
axs[2,1].set_xlabel('Frequency/MHz',)
axs[2,1].set_ylabel('幅值',)
axs[2,1].set_title("Spectral of Echo Signal" )

axs[3,0].plot(f/(1e6), np.abs(Y), color = 'b', label = '')
axs[3,0].set_xlabel('Frequency/MHz',)
axs[3,0].set_ylabel('幅值',)
axs[3,0].set_title("Spectral of the Result of Pulse Compression" )

axs[3,1].plot(r, 20*np.log10(y), color = 'b', label = '')
axs[3,1].set_xlabel('Range/m',)
axs[3,1].set_ylabel('幅值',)
axs[3,1].set_title("Result of Pulse Compression" )

print(f"R0 = {R0}, R0_est = {R0_est}")
plt.show()
plt.close()

#=============== 时域实现脉冲压缩 ================
matched_filter = np.conj(st[::-1])  # 发射信号的共轭反转
compressed_signal = np.convolve(secho, matched_filter, mode = 'same')
# 结果归一化
compressed_signal = compressed_signal / np.max(np.abs(compressed_signal) + 1e-10 )
# 结果可视化
fig, axs = plt.subplots(4, 1, figsize = (6, 16), constrained_layout = True)

# 发射信号（实部）
axs[0].plot(t*1e6, np.real(st))
axs[0].set_title("Transmitted LFM Signal (Real Part)")
axs[0].set_xlabel("Time (μs)")
axs[0].set_ylabel("Amplitude")

# 接收回波信号（实部）
axs[1].plot(t*1e6, np.real(secho))
axs[1].set_title("Received Echo Signal (Real Part)")
axs[1].set_xlabel("Time (μs)")
axs[1].set_ylabel("Amplitude")

# matched_filter 信号（实部）
t_match = np.arange(len(matched_filter)) / fs
axs[2].plot(t_match * 1e6, np.real(matched_filter))
axs[2].set_title("matched_filter (Real Part)")
axs[2].set_xlabel("Time (μs)")
axs[2].set_ylabel("Amplitude")

# 脉冲压缩结果（幅度）
tmp = int((min(len(secho), len(matched_filter)) - 1) /2)
t_compressed = (np.arange(len(compressed_signal)) - tmp) / fs
r = t_compressed * c/2;
R0_est1 = r[np.argmax(np.abs(compressed_signal))]
print(f"R0 = {R0}, R0_est1 = {R0_est1}")

axs[3].plot(r, 20 * np.log10(np.abs(compressed_signal) + 1e-10))
axs[3].set_title("Range/m")
axs[3].set_xlabel("Range (m)")
axs[3].set_ylabel("Amplitude (dB)")

plt.show()
plt.close()




#%% 多目标回波信号的匹配滤波输出(含源码)
# https://mp.weixin.qq.com/s?__biz=MzAwMDE1ODE5NA==&mid=2652542571&idx=1&sn=0e0eb494ac7ee19d18227a5e96c2b27e&chksm=80065fae159d3dd84e1d9c3a866f126b4b306ec97b1a427b7ff664c0c311286533a276ab7193&mpshare=1&scene=1&srcid=0329Q8dj1B90QMlepVAj2Um9&sharer_shareinfo=38d19dc84b14ff1c2d3b069947b97c9c&sharer_shareinfo_first=38d19dc84b14ff1c2d3b069947b97c9c&exportkey=n_ChQIAhIQFCYeQ%2B6%2BTwh8yNrYRb5RTBKfAgIE97dBBAEAAAAAAFd1FUcF70gAAAAOpnltbLcz9gKNyK89dVj0FDSmEnzfw8MsNY2waUVVqmm5UxZzyDzF5tbZS7E1FJ8ks%2FFLirUTE1wQ2Xr5RMr0LSsVrqypI%2F2aqly%2Fl4uofOZAPvQQjCb4t1wr1bgr1iGp0%2Fja6EufHwe6%2BOtX8Muca1J8F%2F1mtxqFxdDnfAIGnTm7M%2BC2BumNQg1gfrdTl6iuQghRu9X1fqpoRIHk%2BmYl7dtIDNp40mke%2FmuiC%2Fr9RUITAQQShNsr%2FvVz5QleWdVWLSST1uCtkvEuYdurrGkLJZKHLp9gZyOW95cPiUp8bNB0gtT7SOTvU9UrH8Eedr8sQLQBsqwtiKAVKJjgqUj6RjiH3yJWarkT&acctmode=0&pass_ticket=fh8TkWVQ2FSWTxDQvzOQRMqDWhGDthA7I9lYcXveqOdL%2Bq7ha%2FaWBw%2Fse4F%2BIMDs&wx_header=0#rd

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

### 1，生成多个周期的脉冲信号，脉内调制为线性调频信号。
# 1. 参数设置
T = 20e-6  # 单个脉冲宽度 (s)
B = 20e6  # 调频带宽 (Hz)
fs = 10 * B  # 采样率，奈奎斯特采样，取为 2B
N_pulse = 5  # 脉冲数
DutyCycle = 0.1  # 占空比 (10%)
PRI = T / DutyCycle  # 脉冲重复间隔 (s)
N_total = int(round(PRI * fs * N_pulse))  # 总采样点数
t = np.arange(N_total) / fs  # 总时间序列

# 2. 生成单个 LFM 脉冲信号
alpha = B / T  # 调频斜率
samples_pulse = int(round(T * fs))  # 单个脉冲的采样点数
t_pulse = np.arange(samples_pulse) / fs  # 单个脉冲的时间序列
s_pulse = np.exp(1j * np.pi * alpha * t_pulse**2)  # 单个脉冲信号

# 3. 生成脉冲列
s_tx = np.zeros(N_total, dtype=complex)
samples_gap = int(round(PRI * fs)) - samples_pulse  # 脉冲间隔的采样点数

if samples_gap < 0:
    raise ValueError('占空比过高，导致脉冲间隔为负。请调整占空比或脉冲宽度。')

for n in range(N_pulse):
    start_idx = n * (samples_pulse + samples_gap)
    end_idx = start_idx + samples_pulse
    if end_idx <= N_total:
        s_tx[start_idx:end_idx] = s_pulse
    else:
        print(f'警告: 脉冲 {n+1} 超出总采样点数范围，未完全生成。')

# 可视化发射脉冲列
plt.figure(figsize=(12, 6))
plt.plot(t * 1e6, np.real(s_tx))
plt.title('发射 LFM 脉冲列（实部）')
plt.xlabel('时间 (µs)')
plt.ylabel('幅度')
plt.xlim(0, PRI * 1e6 * N_pulse)
plt.ylim(-1.2, 1.2)
plt.grid(True)
plt.show()

### 2，根据多个目标的延时生成目标的叠加回波，并添加噪声。
### 4. 模拟多个目标的回波
# 定义多个目标，每个目标有时延 Tau 和衰减系数 A
targets = [
    {'Tau': 5e-6, 'A': 1.0},
    {'Tau': 15e-6, 'A': 0.8},
    {'Tau': 25e-6, 'A': 0.6}
]

rcv_sig = np.zeros(N_total, dtype=complex)

for k, target in enumerate(targets):
    Tau = target['Tau']
    A = target['A']
    delay_samples = int(round(Tau * fs))

    if delay_samples >= N_total:
        print(f'警告: 目标 {k+1} 的时延超过总信号长度，忽略该目标。')
        continue

    # 将发射信号延时后叠加到接收信号中
    rcv_sig[delay_samples:] += A * s_tx[:N_total - delay_samples]

# 添加噪声
SNR_dB = 20  # 信噪比
signal_power = np.sqrt(np.mean(np.abs(rcv_sig)**2))  # RMS功率
noise_power = signal_power / (10**(SNR_dB / 10))
noise = np.sqrt(noise_power / 2) * (np.random.randn(N_total) + 1j * np.random.randn(N_total))
rcv_sig_noisy = rcv_sig + noise

# 绘制接收信号
plt.figure(figsize=(12, 6))
plt.plot(t * 1e6, np.abs(rcv_sig_noisy), 'r', label='幅度')
plt.plot(t * 1e6, np.real(rcv_sig_noisy), 'b', alpha=0.7, label='实部')
plt.title('接收回波幅度（含噪声）')
plt.xlabel('时间 (µs)')
plt.ylabel('幅度')
plt.legend()
plt.grid(True)
plt.show()

### 3，进行匹配滤波（快速卷积实现）
# 5. 匹配滤波器
h = np.conj(s_pulse[::-1])  # 匹配滤波器冲激响应

# 6. 快速卷积实现匹配滤波
# 使用FFT进行快速卷积
len_fft = 2**int(np.ceil(np.log2(len(rcv_sig_noisy) + len(h) - 1)))
S_fft = np.fft.fft(rcv_sig_noisy, len_fft)
H_fft = np.fft.fft(h, len_fft)
y_fft = S_fft * H_fft
mf_output = np.fft.ifft(y_fft)

# 截取有效长度
mf_output = mf_output[:len(rcv_sig_noisy)]

### 7. 可视化匹配滤波输出
plt.figure(figsize=(12, 6))
plt.plot(t * 1e6, 20 * np.log10(np.abs(mf_output) + 1e-10))  # 加小值避免log(0)
plt.title('匹配滤波输出幅度')
plt.xlabel('时间 (µs)')
plt.ylabel('幅度 (dB)')
plt.grid(True)
plt.show()

# 可选：显示目标位置
print("目标位置信息:")
for i, target in enumerate(targets):
    delay_samples = int(round(target['Tau'] * fs))
    if delay_samples < len(mf_output):
        peak_value = 20 * np.log10(np.abs(mf_output[delay_samples]) + 1e-10)
        print(f"目标 {i+1}: 时延 = {target['Tau']*1e6:.1f} µs, 峰值 = {peak_value:.2f} dB")




#%% 三种不同类型信号的脉冲压缩（一）--------线性调频脉冲信号的压缩处理
# https://mp.weixin.qq.com/s?__biz=Mzg3ODkwOTgyMw==&mid=2247485021&idx=3&sn=742ef5748dce8629ebc43d99ad06befe&chksm=ce5103d1e0308bf218466c6e248c75b18e5064424c93d98e916591d1398c5811881ff4131ad9&mpshare=1&scene=1&srcid=0329wtlZc1KNnFCZ6iw1tk7G&sharer_shareinfo=48caf2d5f9cf6976c915c311eee94f2b&sharer_shareinfo_first=48caf2d5f9cf6976c915c311eee94f2b&exportkey=n_ChQIAhIQYIFUhJ1ixZB7LHc8116UpxKfAgIE97dBBAEAAAAAAFs7BFMneDoAAAAOpnltbLcz9gKNyK89dVj08JjHPehNSxSotXXsU001an68bbK6IqjQ60hNFBjrROO1ZNChcAUoUGNBOq%2BD7vVzTk4zhjDQfHgsd36CwGvEP9cuCpcaF0b84K1woLB5BqZlBpBKeciOu%2FhYsfoYtoJR9v241Kspkw9ouDuSLwYzBApbL88wLKd6vgimG5ZaCZq28gVyWgQiuYepcUZBThnU%2BhV%2FjawaczWvNkPrJ0B0EOkq8aIACuOXLWHj3itH3W%2B%2F6W9ebuggdjgZDtiLb4wjDcgPBWYK8ugu3jGyrVnRKNP0r4QHnH%2F9%2F2EQfqdYWFg8z%2FfxBUOJAlVFN7PKRJwY0AgdHdiwFHCY&acctmode=0&pass_ticket=UOuTwL5JezorkCrj%2BTMjx7yzKbpTU8fEVTb6keEK8pXFIgyOFbr4GvLLR%2F4CSq25&wx_header=0#rd

import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
###**************参数配置*********************
Tp = 200e-6;                # 发射脉冲宽度s
B = 1e6;                    # 调频带宽Hz
Ts = 0.5e-6;                # 采样时钟s
R0 = np.array([80e3, 85e3])         # 目标的距离m
vr = np.array([0, 0])               # 目标速度
SNR = np.array([20, 10])            # 信噪比
Rmin = 20e3;                # 采样的最小距离
Rrec = 150e3;               # 接收距离窗的大小
bos = 2 * np.pi / 0.03;     # 波数2*pi/λ。

# *********************************************
mu = B/Tp;                    # 调频率
c = 3e8;                      # 光速m/s
M = int(np.round(Tp/Ts))
t1 = np.arange(-M/2+0.5, M/2+0.5) * Ts;     # 时间矢量
NR0 = np.ceil(np.log2(2 * Rrec / c / Ts));
NR1 = int(2**NR0)                             # 计算FFT的点数
lfm = np.exp(1j * np.pi * mu * t1**2);        # 信号复包络
lfm_w = lfm * scipy.signal.windows.hann(M)
gama = (1+2*vr/c)**2
sp = 0.707 * (np.random.randn(NR1) + 1j * np.random.randn(NR1));                                  # 噪声
for k in range(len(R0)):
    NR = math.trunc(2*(R0[k] - Rmin) / c / Ts)
    print(f"NR = {NR}")
    spt = (10**(SNR[k]/20)) * np.exp(-1j * bos * R0[k]) * np.exp(1j * np.pi * mu * gama[k] * t1**2);    # 信号
    sp[NR - 1: NR + M - 1] = sp[NR - 1: NR + M - 1] + spt

spf = scipy.fft.fft(sp, NR1);
lfmf = scipy.fft.fft(lfm, NR1);                                    # 未加窗
lfmf_w = scipy.fft.fft(lfm_w, NR1);                                # 加窗
y = np.abs(scipy.fft.ifft(spf * np.conjugate(lfmf), NR1)/NR0);
y1 = np.abs(scipy.fft.ifft(spf * np.conjugate(lfmf_w), NR1)/NR0)   # 加窗

fig, axs = plt.subplots(4, 1, figsize = (6, 16), constrained_layout = True)

axs[0].plot(np.real(sp))
# axs[0].set_title("Range/m")
axs[0].set_xlabel("时域采样点")
axs[0].set_ylabel("Amplitude ")

axs[1].plot(t1*1e6, np.real(lfm))
axs[1].set_title("匹配滤波系数实部")
axs[1].set_xlabel("时间/us")
axs[1].set_ylabel("匹配滤波系数实部")

axs[2].plot(np.arange(NR1)/10, 20 * np.log10(y))
axs[2].set_title("脉冲压缩结果（未加窗）")
axs[2].set_xlabel("距离/km")
axs[2].set_ylabel("脉压输出/dB")

axs[3].plot(np.arange(NR1)/10, 20 * np.log10(y1))
axs[3].set_title("脉冲压缩结果（加窗）")
axs[3].set_xlabel("距离/km")
axs[3].set_ylabel("脉压输出/dB")

plt.show()
plt.close()


#%%
def xcorr(x, y, normed = True, detrend = True, maxlags = 10):
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed = True
    # Returns inner products when normed = False
    # Usage: lags, c = xcorr(x, y, maxlags = len(x)-1)
    # Optional detrending e.g. mlab.detrend_mean

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')

    if detrend:
        import matplotlib.mlab as mlab
        x = mlab.detrend_mean(np.asarray(x)) # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))
    c = np.correlate(x, y, mode='full')
    if normed:
        # n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        n = np.sqrt(np.linalg.norm(x)**2 * np.linalg.norm(y)**2)
        c = np.true_divide(c,n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags : Nx + maxlags]
    return c, lags

nscat = 2             # 接收窗内的点散射体数
rrec = 50             # 接收窗的大小m
taup = 10e-6          # 未压缩的脉冲宽度s
b = 50.0e6            # 信号带宽Hz
scat_range = np.array([15, 25])        # 散射体的相对距离矢量（在接收窗内）
scat_rcs = np.array([1, 2])            # 散射体的RCS
winid = 0                     # 窗函数，0表示无窗函数
eps = 1.0e-16                 # 定义一个很小的常量，用于处理数值中计算舍入误差
time_B_product = b * taup    # 时宽带宽积
c = 3.0e8;                    # 光速
#  在匹配滤波器的应用中，时间带宽积和采样点数之间存在一定的关系。通常情况下，为了准确地捕捉信号
#  特征并避免信息的丢失，采样点数应该足够多，以确保在时间域内有足够的采样点来表示信号的特征。
#  一般而言，采样点数应该大于等于时间带宽积，以确保恢复出精确的信号特征。
n = math.trunc(5 * time_B_product)   # 乘以5的目的是为了提供一定荣誉，来防止信号特征在时间域上的模糊化
x = np.zeros((nscat, n), dtype = complex)
y = np.zeros(n, dtype = complex)
replica = np.zeros(n, dtype = complex)

t = np.linspace(-taup/2, taup/2, n);
replica = np.exp(1j * np.pi * (b/taup) * t**2);

sampling_interval = taup / n;       #  采样间隔
freqlimit = 0.5/ sampling_interval; #  通过将0.5除以采样间隔，可以计算出信号的最高频率，在这个频率以下的信号可以被准确地表示和恢复。
freq = np.linspace(-freqlimit, freqlimit, n);

#  对于每个散射体，计算其距离range对应的散射信号，并将其与输出向量y相加。
for j in range(nscat):
    Range = scat_range[j]
    x[j,:] = scat_rcs[j] * np.exp(1j * np.pi * (b/taup) * (t + (2*Range/c))**2)   #  回波信号
    y = x[j,:] + y     #  回波信号相加

out, _ = xcorr(replica, y, maxlags = replica.size-1)                # 计算发射信号和回波信号的相关性
out = out / n                             # 归一化
s = taup * c / 2                          # 计算脉冲宽度taup对应的距离步长s
Npoints = int(np.ceil(rrec * n /s))       # LFM 的距离步长为 s 对应 n 个点，则 rrec 对应的点数
dist = np.linspace(0, rrec, Npoints)      # 基于接收窗口的范围rrec计算距离向量dist
delr = c/2/b

fig, axs = plt.subplots(4, 1, figsize = (6, 16), constrained_layout = True)

axs[0].plot(t, np.real(replica))
axs[0].set_title("匹配滤波系数实部")
axs[0].set_xlabel("时间/s")
axs[0].set_ylabel("匹配滤波系数实部")

axs[1].plot(freq, scipy.fft.fftshift(np.abs(scipy.fft.fft(replica))))
# axs[1].set_title("脉冲压缩结果（未加窗）")
axs[1].set_xlabel("频率/Hz")
axs[1].set_ylabel("频谱")

axs[2].plot(t, np.real(y),)
axs[2].set_title("脉冲压缩结果（加窗）")
axs[2].set_xlabel("Relative delay / s")
axs[2].set_ylabel("未进行脉压")

axs[3].plot(dist, np.abs(out[n-1 : n+Npoints-1]))
axs[3].set_title("Range/m")
axs[3].set_xlabel("目标位置/m")
axs[3].set_ylabel("脉压输出")

plt.show()
plt.close()


#%% https://mp.weixin.qq.com/s?__biz=MzUxNTY5NzYzMA==&mid=2247553394&idx=1&sn=85255267d109644bbd54a80a8d161d98&chksm=f82261285e85b08e2ed81badfe157aed88a7c5fa599ad64148d3fe4ddfe2e1cdb2b95c395778&mpshare=1&scene=1&srcid=0329S589unC9aTqoMGAP80Ip&sharer_shareinfo=1239f9b97eb0d5376843535656576bc0&sharer_shareinfo_first=1239f9b97eb0d5376843535656576bc0&exportkey=n_ChQIAhIQrXuc3vlmReLg2VrWk3pxgxKfAgIE97dBBAEAAAAAAH%2FJCqibM60AAAAOpnltbLcz9gKNyK89dVj0qU8QDSNbqKY3HMOHwcWHRbw3xUWZ1kd2zoydLPbQBKZRVcpqSq8JQrP14GYQd53PJjdAvDePUFL6Lj3FcUTrOk39woXEQX%2FX8iLqPvs7a54T2BoA79vTgvhnfkKX9FqmBIUm8hNdQPAcqNfAWcwaObXb4bX4gp9RyMEBbuO2cKCGEzoAL5WvBC0n3EVnE4isF7%2B2O3jEOCToSjCZaEO%2BAyGrQM3QPEwQDn%2BXhkLkqOIs1Y9A0QHvcMykYwhW2A7xFWrnc4IOvCqsZclNnKGsld%2BPEhTp8AKKQtM574RxLhfAdBVNVhUX%2BSel%2F5pezmlpgWSV%2FDm1zJ45&acctmode=0&pass_ticket=juUJ8JHuA70tTcAQyaFf2ZDKkTnOdyVFeAMOFBjosljVKpPPqm9P1olPjK8m7M%2Bg&wx_header=0#rd



#%%



#%%


























































