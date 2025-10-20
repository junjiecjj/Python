#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 13:20:28 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import firwin, filtfilt, hilbert, find_peaks
from scipy.fft import fft, fftfreq
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
np.random.seed(42)

def designfilt(filter_type, FilterOrder, CutoffFrequency1, CutoffFrequency2, SampleRate):
    if filter_type == 'bandpassfir':
        nyquist = SampleRate / 2
        low = CutoffFrequency1 / nyquist
        high = CutoffFrequency2 / nyquist
        return firwin(FilterOrder + 1, [low, high], pass_zero=False)
    else:
        raise ValueError("仅支持bandpassfir滤波器类型")

def pmtm(x, nw, nfft, fs):
    f, pxx = signal.welch(x, fs, nperseg=nfft//4, nfft=nfft)
    return pxx, f

def xcorr(x, mode='none'):
    correlation = np.correlate(x, x, mode='full')
    lags = np.arange(-len(x)+1, len(x))

    if mode == 'coeff':
        correlation = correlation / np.max(correlation)

    return correlation, lags

def bandpower(x, fs, freq_range):
    f, pxx = signal.welch(x, fs, nperseg=1024)
    idx = np.logical_and(f >= freq_range[0], f <= freq_range[1])
    return np.trapz(pxx[idx], f[idx])

def mad(data, axis=None):
    return np.median(np.abs(data - np.median(data, axis)), axis)

# %% 优化参数设置
fs = 100   # 采样频率 (Hz)
T = 90     # 优化总时间
t = np.arange(0, T, 1/fs)  # 时间向量

## 雷达参数
fc = 60e9  # 使用60GHz频段（更适合生命体征检测）
c = 3e8
lambda_val = c/fc

## 优化呼吸参数
f_breath = 0.25   # 呼吸频率 15次/分钟
A_breath = 0.012   #  增加呼吸幅度到12mm
# 优化心跳参数
f_heart = 1.2     #
A_heart = 0.0005   # 增加心跳幅度
# 显著降低噪声
noise_level = 0.0003

#%% 1. 生成优化的生理信号
# 基础呼吸信号 + 谐波
base_breath = A_breath * np.sin(2*np.pi*f_breath*t)
breath_harmonic = 0.08*A_breath * np.sin(4*np.pi*f_breath*t + np.pi/4)
# 心跳信号与呼吸调制
heart_signal_base = A_heart * np.sin(2*np.pi*f_heart*t)
heart_modulation = 0.15 * np.sin(2*np.pi*f_breath*t - np.pi/2)
heart_signal_modulated = heart_signal_base * (1 + 0.1*heart_modulation)
# 组合信号
chest_displacement = base_breath + breath_harmonic + heart_signal_modulated
# 最小化体动干扰
body_motion = np.zeros_like(t)
motion_times = [30, 60]
for mt in motion_times:
    idx = np.where((t >= mt) & (t < mt + 2))[0]
    if len(idx) > 0:
        motion_env = np.hanning(len(idx))
        body_motion[idx] = 0.002 * np.sin(2*np.pi*0.8*(t[idx]-mt)) * motion_env
# 最终位移信号
total_displacement = chest_displacement + body_motion + noise_level * np.random.randn(len(t))

#%% 2. 优化的雷达信号生成
R0 = 1.0 # 缩短距离提高信噪比
# 相位信号生成
phase_signal = 4*np.pi*(R0 + total_displacement)/lambda_val
# 极低相位噪声
phase_noise = 0.02 * np.random.randn(len(phase_signal))
phase_signal_clean = phase_signal + phase_noise

radar_signal = np.exp(1j * phase_signal_clean)

#%% 3. 高级信号处理
# 稳健的相位解缠
unwrapped_phase = np.unwrap(np.angle(radar_signal))
# 多项式去趋势
p = np.polyfit(t, unwrapped_phase, 2)
phase_trend = np.polyval(p, t)
detrended_phase = unwrapped_phase - phase_trend
# 3.3 使用FIR滤波器（更稳定
breath_filter_coeff = designfilt('bandpassfir',
                                FilterOrder=80,
                                CutoffFrequency1=0.18,
                                CutoffFrequency2=0.35,
                                SampleRate=fs)

heart_filter_coeff = designfilt('bandpassfir',
                               FilterOrder=100,
                               CutoffFrequency1=0.9,
                               CutoffFrequency2=1.6,
                               SampleRate=fs)

breath_signal = filtfilt(breath_filter_coeff, 1, detrended_phase)
heart_signal = filtfilt(heart_filter_coeff, 1, detrended_phase)

# %% 4. 精确频率估计

N = len(t)
f = fftfreq(N, 1/fs)
# 使用多窗口频谱估计
nw = 4
pxx, f_psd = pmtm(detrended_phase, nw, N, fs)
# 呼吸频带寻找峰值
breath_mask = (f_psd >= 0.1) & (f_psd <= 0.5)
if np.sum(breath_mask) > 0:
    breath_idx = np.argmax(pxx[breath_mask])
    f_breath_psd = f_psd[breath_mask]
    detected_breath_freq = f_breath_psd[breath_idx]
else:
    detected_breath_freq = f_breath
# 心跳频带寻找峰值
heart_mask = (f_psd >= 0.8) & (f_psd <= 2.0)
if np.sum(heart_mask) > 0:
    heart_idx = np.argmax(pxx[heart_mask])
    f_heart_psd = f_psd[heart_mask]
    detected_heart_freq = f_heart_psd[heart_idx]
else:
    detected_heart_freq = f_heart

# %% 5. 修正的多方法融合估计
print('进行修正的多方法融合估计...')
# 方法1: 修正的自相关法
acf, lags = xcorr(breath_signal, 'coeff')
positive_lags = lags >= 0
acf = acf[positive_lags]
lags_sec = lags[positive_lags] / fs

min_period = 1/0.5
max_period = 1/0.1
valid_lags = (lags_sec >= min_period) & (lags_sec <= max_period)

breath_freq_autocorr = detected_breath_freq
if np.sum(valid_lags) > 0:
    valid_acf = acf[valid_lags]
    valid_lags_sec = lags_sec[valid_lags]

    peak_indices = find_peaks(valid_acf, height=0.3, distance=fs//4)[0]

    if len(peak_indices) > 0:
        peak_heights = valid_acf[peak_indices]
        main_peak_idx = np.argmax(peak_heights)
        main_period = valid_lags_sec[peak_indices[main_peak_idx]]
        breath_freq_autocorr = 1 / main_period
# 方法2: 改进的希尔伯特变换瞬时频率
analytic_signal = hilbert(breath_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_freq = np.diff(instantaneous_phase) * fs / (2*np.pi)

valid_freq_idx = (instantaneous_freq > 0.1) & (instantaneous_freq < 0.5)
if np.sum(valid_freq_idx) > 0:
    breath_freq_hilbert = np.median(instantaneous_freq[valid_freq_idx])
else:
    breath_freq_hilbert = detected_breath_freq

# 方法3: 改进的零交叉法
breath_signal_mean = breath_signal - np.mean(breath_signal)
zero_crossings = np.where(np.diff(np.sign(breath_signal_mean)) != 0)[0]

breath_freq_zero = detected_breath_freq
if len(zero_crossings) >= 4:
    full_periods = []
    for i in range(1, len(zero_crossings)-1):
        if (breath_signal_mean[zero_crossings[i-1]+1] > 0 and
            breath_signal_mean[zero_crossings[i+1]+1] < 0):
            period = (zero_crossings[i+1] - zero_crossings[i-1]) / fs
            full_periods.append(period)

    if len(full_periods) > 0:
        breath_freq_zero = 1 / np.mean(full_periods)

freq_estimates = np.array([detected_breath_freq, breath_freq_autocorr, breath_freq_hilbert, breath_freq_zero])
reliability = np.ones(4)

if np.sum(breath_mask) > 0:
    reliability[0] = pxx[breath_mask][breath_idx] / np.max(pxx[breath_mask])
else:
    reliability[0] = 0.5

if 'peak_heights' in locals() and len(peak_heights) > 0:
    reliability[1] = np.max(peak_heights) / np.max(acf[valid_lags])
else:
    reliability[1] = 0.5

if np.sum(valid_freq_idx) > 10:
    try:
        from scipy.stats import median_abs_deviation
        mad_value = median_abs_deviation(instantaneous_freq[valid_freq_idx])
    except:
        mad_value = mad(instantaneous_freq[valid_freq_idx])
    reliability[2] = 1 - mad_value / np.median(instantaneous_freq[valid_freq_idx])
else:
    reliability[2] = 0.5

if 'full_periods' in locals() and len(full_periods) >= 3:
    reliability[3] = 1 - np.std(full_periods) / np.mean(full_periods)
else:
    reliability[3] = 0.5

weights = reliability / np.sum(reliability)
breath_freq_final = np.sum(weights * freq_estimates)
# %% 6. 信号质量评估
print('评估信号质量...')

breath_band_power = bandpower(breath_signal, fs, [0.15, 0.35])
residual = detrended_phase - breath_signal - heart_signal
noise_power = bandpower(residual, fs, [0.15, 0.35])

if noise_power > 0:
    snr_linear = breath_band_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
else:
    snr_db = 30

periodicity_score = 0.6
if 'peak_indices' in locals() and len(peak_indices) >= 3:
    peak_times = valid_lags_sec[peak_indices]
    periods = np.diff(peak_times)
    periodicity_score = 1 - np.std(periods) / np.mean(periods)

snr_score = min(1, snr_linear / 10) if 'snr_linear' in locals() else 0.5
consistency_score = 1 - np.std(freq_estimates) / np.mean(freq_estimates) if len(freq_estimates) > 1 else 0.5

signal_quality = (0.4 * periodicity_score + 0.4 * snr_score + 0.2 * consistency_score) * 100
# %% 7. 最终可视化
print('生成最终可视化...')

plt.figure(figsize=(16, 12))

plt.subplot(2, 3, 1)
plt.plot(t, total_displacement * 1000, color=[0.2, 0.2, 0.8], linewidth=1)
plt.plot(t, base_breath * 1000, 'r', linewidth=2)
plt.plot(t, heart_signal_modulated * 1000, 'g', linewidth=1.5)
plt.plot(t, body_motion * 1000, 'm--', linewidth=1)
plt.xlabel('时间 (秒)')
plt.ylabel('位移 (mm)')
plt.title('胸腔位移信号分解')
plt.legend(['总信号', '呼吸', '心跳', '体动'], loc='best')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(f_psd, 10*np.log10(pxx), 'k', linewidth=2)
plt.axvline(x=f_breath, color='r', linestyle='--', linewidth=2,
            label=f'真实呼吸 {f_breath:.2f}Hz')
plt.axvline(x=detected_breath_freq, color='r', linestyle='-', linewidth=2,
            label=f'检测呼吸 {detected_breath_freq:.2f}Hz')
plt.axvline(x=f_heart, color='g', linestyle='--', linewidth=2,
            label=f'真实心跳 {f_heart:.2f}Hz')
plt.axvline(x=detected_heart_freq, color='g', linestyle='-', linewidth=2,
            label=f'检测心跳 {detected_heart_freq:.2f}Hz')
plt.xlabel('频率 (Hz)')
plt.ylabel('功率谱密度 (dB/Hz)')
plt.title('多窗口功率谱估计')
plt.xlim([0, 2.5])
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(t, breath_signal, 'r', linewidth=2)
plt.xlabel('时间 (秒)')
plt.ylabel('幅度')
plt.title('提取的呼吸信号')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(lags_sec, acf, 'b', linewidth=2)
if 'peak_indices' in locals() and len(peak_indices) > 0:
    peak_positions = valid_lags_sec[peak_indices]
    plt.plot(peak_positions, peak_heights, 'ro', markersize=8, linewidth=2)
plt.xlabel('延迟 (秒)')
plt.ylabel('自相关系数')
plt.title('呼吸信号自相关函数 (修正)')
plt.xlim([0, 15])
plt.grid(True)

plt.subplot(2, 3, 5)
methods = ['MTM谱估计', '自相关法', '希尔伯特', '零交叉法', '最终融合']
freq_values = np.array([detected_breath_freq, breath_freq_autocorr, breath_freq_hilbert, breath_freq_zero, breath_freq_final])
errors = np.abs(freq_values - f_breath) * 60

bars = plt.bar(range(len(freq_values)), freq_values * 60)
plt.axhline(y=f_breath*60, color='r', linestyle='--', linewidth=3,
            label=f'真实值: {f_breath*60:.1f}次/分钟')

for i in range(len(freq_values)):
    if i < len(weights):
        weight_text = f'权重: {weights[i]:.2f}'
    else:
        weight_text = f'权重: {weights[-1]:.2f}'

    plt.text(i, freq_values[i]*60 + 0.5, f'误差: {errors[i]:.2f}\n{weight_text}',
             horizontalalignment='center', fontsize=8, fontweight='bold')

plt.xticks(range(len(methods)), methods)
plt.ylabel('呼吸率 (次/分钟)')
plt.title('多方法检测结果对比 (修正)')
plt.ylim([0, 20])
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 6)
performance_metrics = [
    '频率精度', '信噪比', '周期性',
    '一致性', '算法可靠性'
]

metric_values = np.array([
    max(0, 100 - errors[-1]*10),
    min(100, max(0, snr_db + 20)),
    periodicity_score * 100,
    consistency_score * 100,
    np.mean(reliability) * 100
])

plt.barh(range(len(metric_values)), metric_values)
plt.yticks(range(len(performance_metrics)), performance_metrics)
plt.xlabel('性能分数 (%)')
plt.xlim([0, 100])
plt.title('系统性能评估')
plt.grid(True)

for i, value in enumerate(metric_values):
    plt.text(value + 2, i, f'{value:.1f}%',
             verticalalignment='center', fontsize=10)

plt.suptitle('毫米波雷达呼吸检测 - Python版本', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print('\n=== 最终修正版本结果 ===')
print(f'真实呼吸率: {f_breath:.3f} Hz ({f_breath*60:.1f} 次/分钟)')
print(f'MTM谱估计:  {detected_breath_freq:.3f} Hz ({detected_breath_freq*60:.1f} 次/分钟) - 误差: {errors[0]:.2f} 次/分钟')
print(f'自相关法:   {breath_freq_autocorr:.3f} Hz ({breath_freq_autocorr*60:.1f} 次/分钟) - 误差: {errors[1]:.2f} 次/分钟')
print(f'希尔伯特:   {breath_freq_hilbert:.3f} Hz ({breath_freq_hilbert*60:.1f} 次/分钟) - 误差: {errors[2]:.2f} 次/分钟')
print(f'零交叉法:   {breath_freq_zero:.3f} Hz ({breath_freq_zero*60:.1f} 次/分钟) - 误差: {errors[3]:.2f} 次/分钟')
print(f'最终结果:   {breath_freq_final:.3f} Hz ({breath_freq_final*60:.1f} 次/分钟) - 误差: {errors[4]:.2f} 次/分钟')
print(f'系统信噪比: {snr_db:.2f} dB')
print(f'信号质量:   {signal_quality:.1f}%')
print(f'方法可靠性: [{reliability[0]:.2f}, {reliability[1]:.2f}, {reliability[2]:.2f}, {reliability[3]:.2f}]')

if errors[4] <= 1.0:
    print('检测结果: ✓ 优秀 (临床级精度)')
elif errors[4] <= 2.0:
    print('检测结果: ✓ 良好 (实用级精度)')
elif errors[4] <= 3.0:
    print('检测结果: ○ 一般 (需进一步优化)')
else:
    print('检测结果: △ 需要改进')
