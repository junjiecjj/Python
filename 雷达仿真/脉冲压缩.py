#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:03:56 2022

@author: jack

关于脉冲压缩的几点说明:
(1) 雷达几乎都是数字域进行脉压处理的，脉冲压缩本身就是实现信号的匹配滤波，只是在模拟域一般称为匹配滤波，而在数字域中称为脉冲压缩.
(2) 距离分辨率要求B越大越好，雷达最大探测距离要求B越小越好（其他变量恒定的情况下）,即提高最大探测距离就要减小距离分辨率, 脉冲压缩的目的是在不减小最大探测距离的情况下提高雷达的距离分辨率.
(3) 脉冲压缩可以同时提高工作距离和距离分辨率。
(4) 雷达系统想要同时满足高距离分辨率和高速度分辨率，就必须采用大时宽带宽积信号。对于一个普通信号，其时宽带宽积为一个常数，即窄脉冲具有宽频带，宽脉冲具有窄频带。而脉冲压缩处理可以将发射的宽脉冲信号压缩成窄脉冲信号,使信号既可以发射宽脉冲以提高平均功率和雷达的速度分辨率,又能保持窄脉冲的距离分辨率。而脉冲压缩处理本身就是信号的匹配滤波，只是在模拟域一般称为匹配滤波，在数字域称为脉冲压缩。

https://blog.csdn.net/jiangwenqixd/article/details/109521694?spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-109521694-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-109521694-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=10

https://blog.csdn.net/RICEresearchNOTE/article/details/140855697

https://blog.51cto.com/u_16213651/8904378

https://blog.csdn.net/qq_44648285/article/details/143471871

https://zhuanlan.zhihu.com/p/692354746

https://blog.csdn.net/qq_43485394/article/details/122655901

https://blog.51cto.com/u_16213651/8904378

"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False        # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]           # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300                # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2               # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6              # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22

def freqDomainView(x, Fs, FFTN = None, type = 'double'): # N为偶数
    if FFTN == None:
        FFTN = 2**int(np.ceil(np.log2(x.size)))
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

#%% https://blog.csdn.net/innovationy/article/details/121572508?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121572508-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121572508-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=6
# 线性调频信号经过匹配滤波
T = 10e-6                   # 脉冲持续时间10us
B = 30e6                    # 线性调频信号的带宽30MHz
K = B/T                     # 调频斜率
Fs = 10*B
Ts = 1/Fs                   # 采样频率和采样间隔
N = int(T/Ts)
t = np.linspace(-T/2, T/2, int(N))
St = np.exp(1j * np.pi * K * t**2);       # 线性调频信号
Ht = np.exp(-1j * np.pi * K * t**2)       # 匹配滤波
Sot = np.convolve(St, Ht);                # 线性调频信号经过匹配滤波
Z = np.abs(Sot)
Z = Z/max(Z)                     # 归一化
Z = 20 * np.log10(Z + 1e-6)

##### plot
fig, axs = plt.subplots(4, 1, figsize = (12, 12), constrained_layout = True)

L = 2 * N - 1
t1 = np.linspace(-T, T, int(L))
Z1 = np.abs(np.sinc(B * t1))     # 辛格函数
Z1 = 20 * np.log10(Z1 + 1e-6)
t1 = t1*B

##
axs[0].plot(t1, Z, color = 'b', label = '匹配滤波')
axs[0].plot(t1, Z1, color = 'r', ls = 'none', marker = 'o', label = '辛格')
axs[0].set_xlabel('时间/s',)
axs[0].set_ylabel('幅值/dB',)
axs[0].set_title("线性调频信号经过匹配滤波" )
axs[0].set_xlim(-15, 15)  # 拉开坐标轴范围显示投影
axs[0].set_ylim(-50, 1)   # 拉开坐标轴范围显示投影
axs[0].legend()

##
N0 = int(3*Fs/B)
t2 = np.arange(-N0 * Ts, N0 * Ts + Ts, Ts)
t2 = B * t2
axs[1].plot(t2, Z[N-N0-1 : N+N0], color = 'b', label = '')
axs[1].plot(t2, Z1[N-N0-1 : N+N0], color = 'r', ls = 'none', marker = 'o', label = '')
axs[1].set_xlabel('时间/s',)
axs[1].set_ylabel('幅度/dB',)
axs[1].set_title("线性调频信号经过匹配滤波(补零展开之后)")
axs[1].set_xticks([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])
axs[1].set_yticks([-13.4, -4, 0])

##
Z_value = Z[2950 - 1 : 3050]
axs[2].plot(Z_value, color = 'b', label = '')
axs[2].set_xlabel('时间/s',)
axs[2].set_ylabel('幅度/dB',)

##
axs[3].plot(Z, color = 'b', label = '')
axs[3].set_xlabel('时间/s',)
axs[3].set_ylabel('幅度/dB',)

plt.show()
plt.close()

##### plot
fig, axs = plt.subplots(2, 1, figsize = (8, 8), constrained_layout = True)

axs[0].plot(t, np.real(St), color = 'b', label = ' ')
axs[0].set_xlabel('时间/s',)
axs[0].set_ylabel('值',)
axs[0].set_title(" " )
# axs[0].set_xlim(-15, 15)  # 拉开坐标轴范围显示投影
# axs[0].set_ylim(-50, 1)   # 拉开坐标轴范围显示投影
axs[0].legend()

f, Y, A, Pha, R, I = freqDomainView(St, Fs, FFTN = None, type = 'double')
axs[1].plot(f, A, color = 'b', label = '')
# axs[1].set_xlim(-20e6, 20e6)  # 拉开坐标轴范围显示投影
axs[1].set_xlabel('f/Hz',)
axs[1].set_ylabel('幅度 ',)
axs[1].set_title(" ")

plt.show()
plt.close()

#%% 时域脉冲压缩
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
T = 10e-6        # 脉冲宽度 10 μs
B = 10e6         # 带宽 30 MHz
f0 = 1e6         # 起始频率 1 MHz
fs = (f0 + B) * 3       # 采样频率 100 MHz
SNR_dB = 20      # 信噪比 (dB)
delay = 5e-6     # 目标时延 5 μs

## 生成线性调频信号(LFM)
t = np.arange(0, T, 1/fs)                  # 时间向量
N = len(t)
chirp_signal = np.exp(1j * np.pi * (B/T) * t**2 + 1j * 2 * np.pi * f0 * t)  # 复数LFM信号

## 模拟回波信号（添加时延和多普勒频移）
delay_samples = int(delay * fs)             # 时延对应的采样点数
echo_signal = np.zeros(N + delay_samples, dtype = complex)
echo_signal[delay_samples:delay_samples + N] = chirp_signal  # 添加时延

## 添加高斯白噪声
noise_power = 10**(-SNR_dB/10) * np.mean(np.abs(echo_signal)**2)
noise = np.sqrt(noise_power/2) * (np.random.randn(len(echo_signal)) + 1j * np.random.randn(len(echo_signal)))
echo_signal += noise

mode = 'same'
## 脉冲压缩（匹配滤波）
matched_filter = np.conj(chirp_signal[::-1])  # 匹配滤波器：发射信号的共轭反转
compressed_signal = np.convolve(echo_signal, matched_filter, mode = mode)
compressed_signal = compressed_signal / np.max(np.abs(compressed_signal))
# 结果可视化
plt.figure(figsize=(12, 8))

## 发射信号（实部）
plt.subplot(3, 1, 1)
plt.plot(t*1e6, np.real(chirp_signal))
plt.title("Transmitted LFM Signal (Real Part)")
plt.xlabel("Time (μs)")
plt.ylabel("Amplitude")

# 接收回波信号（实部）
t_echo = np.arange(len(echo_signal)) / fs * 1e6
plt.subplot(3, 1, 2)
plt.plot(t_echo, np.real(echo_signal))
plt.title("Received Echo Signal with Noise (Real Part)")
plt.xlabel("Time (μs)")
plt.ylabel("Amplitude")

# 脉冲压缩结果（幅度）
if mode == 'full':
    t_compressed = (np.arange(len(compressed_signal)) - len(chirp_signal)) / fs * 1e6
    time_delay = t_compressed[np.argmax(np.abs(compressed_signal))] / 1e6
elif mode == 'same':
    tmp = int((min(len(echo_signal), len(matched_filter)) - 1) /2)
    t_compressed = (np.arange(len(compressed_signal)) - tmp) / fs * 1e6
    time_delay = t_compressed[np.argmax(np.abs(compressed_signal))] / 1e6
elif mode == 'valid':
    t_compressed = np.arange(len(compressed_signal)) / fs * 1e6
    time_delay = t_compressed[np.argmax(np.abs(compressed_signal))] / 1e6

print(f"Estimated time delay = {time_delay}")
plt.subplot(3, 1, 3)
plt.plot(t_compressed, 20 * np.log10(np.abs(compressed_signal)))
plt.title("Pulse Compression Output (dB)")
plt.xlabel("Time (μs)")
plt.ylabel("Amplitude (dB)")
plt.tight_layout()
plt.show()
plt.close()



#%% DeepSeek
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
T = 10e-6          # 脉冲宽度 10μs
B = 10e6           # 带宽 30MHz
f0 = 1e6           # 起始频率 1 MHz
fs = 4 * (f0 + B)  # 采样率(满足 Nyquist 定理)
# t = np.linspace(-T/2, T/2, int(fs * T))  # 时间序列
# t = np.linspace(0, T, int(fs * T))         # 时间序列

t = np.arange(-T/2, T/2, 1/fs)
# t = np.arange(0, T, 1/fs)

# 生成线性调频信号(LFM)
K = B / T  # 调频斜率
s_t = np.exp(1j * np.pi * K * t**2 + 1j * 2 * np.pi * f0 * t)  # 发射信号(复数形式)
# s_t = np.exp(1j * np.pi * K * t**2 )  # 发射信号(复数形式)

# 添加噪声模拟接收信号
noise = 0.1 * (np.random.randn(len(s_t)) + 1j * np.random.randn(len(s_t)))
received_signal = s_t + noise

# 匹配滤波器(脉冲压缩)
matched_filter = np.conj(s_t[::-1])  # 发射信号的共轭反转
compressed_signal = np.convolve(received_signal, matched_filter, mode = 'same')

# 结果归一化
compressed_signal = compressed_signal / np.max(np.abs(compressed_signal))

# 绘图
plt.figure(figsize = (12, 10))

# 发射信号时频图
plt.subplot(3, 1, 1)
plt.plot(t, np.real(s_t))
plt.title("Transmitted LFM Signal (Real Part)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# 接收信号（含噪声）
plt.subplot(3, 1, 2)
plt.plot(t, np.real(received_signal))
plt.title("Received Signal with Noise (Real Part)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# 脉冲压缩结果
plt.subplot(3, 1, 3)
plt.plot(t, np.abs(compressed_signal))
plt.title("Pulse Compression Result")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (Normalized)")
plt.tight_layout()
plt.show()
plt.close()


#%%



#%%




#%%




#%%












































































