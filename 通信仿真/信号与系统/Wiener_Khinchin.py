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
