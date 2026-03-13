#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:09:26 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift, ifft
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


# 设置随机种子以保证结果可重现
np.random.seed(42)

# 生成测试信号 - 一个包含多个频率成分的随机信号
N = 1024  # 信号长度
t = np.arange(N)

# 生成包含多个频率成分的信号
freqs = [0.1, 0.25, 0.4]  # 归一化频率
amplitudes = [1.0, 0.8, 0.6]
phases = [0, np.pi/3, np.pi/4]

# 创建多频信号
x = np.zeros(N)
for freq, amp, phase in zip(freqs, amplitudes, phases):
    x += amp * np.cos(2 * np.pi * freq * t + phase)

# 添加一些高斯噪声
noise_std = 0.2
x += noise_std * np.random.randn(N)

# 方法1: 直接计算功率谱密度 (PSD)
psd_direct = np.abs(fft(x)) ** 2 / N
freqs_fft = np.fft.fftfreq(N)

# 方法2: 通过自相关函数计算PSD (验证Wiener-Khinchin定理)
# 计算自相关函数
def autocorrelation(x):
    """计算信号的自相关函数"""
    n = len(x)
    result = np.correlate(x, x, mode='full')
    return result[n-1:] / np.max(result[n-1:])

# 计算自相关函数
r_xx = autocorrelation(x)
m = np.arange(len(r_xx))  # 滞后值

# 对自相关函数进行DTFT得到PSD
psd_from_acf = np.abs(fft(r_xx, n=N))

# 由于自相关函数是对称的，我们只需要正频率部分
psd_from_acf = psd_from_acf[:N//2]
psd_direct = psd_direct[:N//2]
freqs_positive = freqs_fft[:N//2]

# 可视化结果
plt.figure(figsize=(15, 10))

# 绘制原始信号
plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.title('原始信号')
plt.xlabel('样本索引')
plt.ylabel('幅度')
plt.grid(True)

# 绘制自相关函数
plt.subplot(3, 1, 2)
plt.plot(m, r_xx)
plt.title('自相关函数 $R_{xx}[m]$')
plt.xlabel('滞后 m')
plt.ylabel('自相关')
plt.grid(True)
plt.xlim(0, 100)  # 只显示前100个滞后值

# 绘制功率谱密度比较
plt.subplot(3, 1, 3)
plt.plot(freqs_positive, 10 * np.log10(psd_direct + 1e-10),
         label='直接FFT计算的PSD', alpha=0.8)
plt.plot(freqs_positive, 10 * np.log10(psd_from_acf + 1e-10),
         'r--', label='自相关函数DTFT计算的PSD', alpha=0.8)

# 标记原始频率成分
for freq in freqs:
    plt.axvline(x=freq, color='green', linestyle=':', alpha=0.7,
                label=f'原始频率 {freq}' if freq == freqs[0] else "")

plt.title('功率谱密度比较 (验证Wiener-Khinchin定理)')
plt.xlabel('归一化频率')
plt.ylabel('功率谱密度 (dB)')
plt.legend()
plt.grid(True)
plt.xlim(0, 0.5)

plt.tight_layout()
plt.show()

# 定量验证定理
print("=" * 50)
print("Wiener-Khinchin定理验证结果")
print("=" * 50)

# 计算两种方法得到的PSD之间的相关系数
correlation = np.corrcoef(psd_direct, psd_from_acf)[0, 1]
print(f"两种PSD计算方法的相关性系数: {correlation:.6f}")

# 计算相对误差
relative_error = np.mean(np.abs(psd_direct - psd_from_acf) / (psd_direct + 1e-10))
print(f"平均相对误差: {relative_error:.6f}")

# 检查频率峰值位置是否一致
def find_peak_frequencies(psd, freqs, n_peaks=3):
    """找出PSD中的峰值频率"""
    peaks, _ = signal.find_peaks(psd)
    # 按峰值高度排序
    peak_indices = peaks[np.argsort(psd[peaks])[-n_peaks:]]
    return sorted(freqs[peak_indices])

peak_freqs_direct = find_peak_frequencies(psd_direct, freqs_positive, len(freqs))
peak_freqs_acf = find_peak_frequencies(psd_from_acf, freqs_positive, len(freqs))

print(f"\n直接FFT检测到的峰值频率: {peak_freqs_direct}")
print(f"自相关DTFT检测到的峰值频率: {peak_freqs_acf}")
print(f"原始信号频率成分: {sorted(freqs)}")

# 验证逆变换：从PSD恢复自相关函数
r_xx_recovered = np.real(ifft(psd_from_acf, n=len(r_xx))[:len(r_xx)])

# 计算恢复的自相关函数与原始自相关函数的误差
recovery_error = np.mean(np.abs(r_xx - r_xx_recovered[:len(r_xx)]))
print(f"\n自相关函数恢复误差: {recovery_error:.6f}")

print("\n" + "=" * 50)
if correlation > 0.95 and relative_error < 0.1:
    print("✅ Wiener-Khinchin定理验证成功！")
    print("自相关函数的DTFT等于功率谱密度")
else:
    print("❌ 验证结果不理想，可能存在数值误差")
print("=" * 50)


#%%
# 生成 循环矩阵
def CirculantMatric(gen, row):
     if type(gen) == list:
          col = len(gen)
     elif type(gen) == np.ndarray:
          col = gen.size
     row = col

     mat = np.zeros((row, col), dtype = gen.dtype)
     mat[:, 0] = gen
     for i in range(1, row):
          mat[:,i] = np.roll(gen, i)
     return mat

def circularConvolve(h, s, N):
    if h.size < N:
        h = np.hstack((h, np.zeros(N-h.size)))
    col = N
    row = s.size
    H = np.zeros((row, col), dtype = s.dtype)
    H[:, 0] = h
    for i in range(1, row):
          H[:,i] = np.roll(h, i)
    res = H @ s
    return res

# generateVec =  [1 , 2  , 3 , 4  ]
# X = np.array(generateVec)
# L = len(generateVec)
# A = CirculantMatric(X, L)
# A1 = scipy.linalg.circulant(X)


N = 8
h = np.array([-0.4878, -1.5351, 0.2355])
s = np.array([-0.0155, 2.5770, 1.9238, -0.0629, -0.8105, 0.6727, -1.5924, -0.8007])

lin_s_h = scipy.signal.convolve(h, s)
cir_s_h = circularConvolve(h, s, N)

Ncp = 2
s_cp = np.hstack((s[-Ncp:], s))
lin_scp = scipy.signal.convolve(h, s_cp)
r = lin_scp[Ncp:Ncp+N]
print(f"lin_scp = \n{lin_scp[Ncp:Ncp+N]}\ncir_s_h = \n{cir_s_h}")


R = scipy.fft.fft(r, N)
H = scipy.fft.fft(h, N)
S = scipy.fft.fft(s, N)

r1 = scipy.fft.ifft(S*H)
print(f"r1 = \n{r1}\ncir_s_h = \n{cir_s_h}")

#%% <A Dual-Functional Sensing-Communication Waveform Design Based on OFDM, Guanding Yu>



#%%
# 实现与MATLAB cconv完全一致的圆卷积
def cconv(a, b, n=None):
    """
    实现与MATLAB cconv完全一致的圆卷积
    参数:
        a, b: 输入复数数组
        n: 输出长度 (None表示默认长度len(a)+len(b)-1)
    返回:
        圆卷积结果 (复数数组)
    """
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    # 默认输出长度
    if n is None:
        n = len(a) + len(b) - 1
    # 线性卷积
    linear_conv = np.convolve(a, b, mode='full')
    # 处理不同n的情况
    if n <= 0:
        return np.array([], dtype=complex)
    result = np.zeros(n, dtype=complex)
    if n <= len(linear_conv):
        # n <= M+N-1: 重叠相加
        for k in range(n):
            # 收集所有k + m*n位置的元素
            idx = np.arange(k, len(linear_conv), n)
            result[k] = np.sum(linear_conv[idx])
    else:
        # n > M+N-1: 补零
        result[:len(linear_conv)] = linear_conv

    return result


def convMatrix(h, N):  #
    """
    Construct the convolution matrix of size (L+N-1)x N from the
    input matrix h of size L. (see chapter 1)
    Parameters:
        h : numpy vector of length L
        N : scalar value
    Returns:
        H : convolution matrix of size (L+N-1)xN
    """
    col = np.hstack((h, np.zeros(N-1)))
    row = np.hstack((h[0], np.zeros(N-1)))

    from scipy.linalg import toeplitz
    H = toeplitz(col, row)
    return H


# 下面是OFDM中IFFT -> +cp -> H -> -cp -> FFT的等效过程
h = np.array([-0.4878, -1.5351, 0.2355])
S = np.array([-0.0155, 2.5770, 1.9238, -0.0629, ])
s = np.fft.ifft(S) # IFFT
N = s.size
L = h.size

H = convMatrix(h, N)
y = H @ s

cir_s_h = cconv(h, s, N)

lenCP = L - 1
Acp = np.block([[np.zeros((lenCP, N-lenCP)), np.eye(lenCP)], [np.eye(N)]])

s_cp = Acp @ s                    # add CP

H_cp = convMatrix(h, s_cp.size)
y_cp = H_cp @ s_cp                #  pass freq selected channel

y_remo_cp = y_cp[lenCP:lenCP + N] # receiver, remove cp

H_cp1 = convMatrix(h, s_cp.size)[lenCP:lenCP + N, :]
y_remo_cp1 = H_cp1 @ s_cp        #  pass freq selected channel + remove cp

F = scipy.linalg.dft(N)/np.sqrt(N)
FH = F.conj().T

Diag = F @ H_cp1 @ Acp @ FH  # F@T(h)@A@FH is diagonal such that the data is parallelly transmitted over different subcarriers, and thus the ISI is avoided.

CirH = H_cp1 @ Acp
print(f"h = {h}\nCirH = \n{CirH}") # H --> CirH, 将拓普利兹矩阵变为循环阵, 到这里，从离散信号角度完美的对应OFDM的理论









