#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 13:29:48 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzE5ODQ2NTg0NA==&mid=2247483950&idx=1&sn=8c3ef3f43e14e1c2ca7d5e755dab0752&chksm=97152431f5de62ad4481c912e5b0e1ea7b3aaa66393240cf10996b5fbefd8abc5ad7b07d9a81&mpshare=1&scene=1&srcid=1013NiErdgX4tUejQzIN63eX&sharer_shareinfo=76a8c2d378e6cac6d27a63f78aef2d4b&sharer_shareinfo_first=76a8c2d378e6cac6d27a63f78aef2d4b&exportkey=n_ChQIAhIQ1tSUT82pZvRkwL%2B7yVp1lhKfAgIE97dBBAEAAAAAACzSB5uw0dcAAAAOpnltbLcz9gKNyK89dVj0DGN2lWoTghECYjddoV7fGkrXZXDEDSh28hb1C2zjaJilhRKM5v4O377FkrHFQLwxq834ksjtR755rFNAcWzq%2BfXQysX2Kh9ioOtr4IlYMg9JL7xIaJ364SqfnJBmkzXJbwY7LPl1B0u1LLVOu51hcY2Y%2F90oc%2BvmzY1fNIFI3Q7dvxQatCXLSCdEwjTAFDN4k%2BZWNqOxkIZwIkwe3L6nc7Mdhmyq8H4%2Bfvw35uZVpifIS%2B1FKDYJ5Wmic9y2c2zqZn5QlCCnK%2Bk3O5ifTeYznJggS%2Bm3aNuH5yYeZ70kqzroHQtFuYmyBwKNbgZ3cxN37Pt4FonnZYdV&acctmode=0&pass_ticket=QG1YqDDkdHYeaquYFYcruJKXk6KxJjz%2BvOC7fzcoI5HtG90jCHQFa6V2pba8dy5S&wx_header=0#rd

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.stats import kurtosis
import warnings
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

# 雷达参数设置
c = 3e8           # 光速 (m/s)
fc = 77e9         # 载波频率 (Hz)
lambda_ = c / fc  # 波长 (m)

# 雷达参数
PRF = 2000        # 脉冲重复频率 (Hz)
T = 1 / PRF       # 脉冲重复间隔 (s)
N = 512           # 每个脉冲的采样点数
CPI = 0.5         # 相干处理间隔 (s)
M = round(CPI / T) # 脉冲数

# 目标参数
target_types = ['行人', '自行车', '汽车']
num_targets = len(target_types)

# 生成微多普勒特征
# 初始化微多普勒特征矩阵
microdoppler_features = [None] * num_targets
time_axis = np.arange(M) * T
frequency_axis = np.fft.fftshift(np.fft.fftfreq(M, 1/PRF))

# 添加AWGN噪声的函数
def awgn(signal, snr_db):
    """
    添加加性高斯白噪声
    signal: 输入信号
    snr_db: 信噪比 (dB)
    """
    # 计算信号功率
    if np.iscomplexobj(signal):
        signal_power = np.mean(np.abs(signal) ** 2)
    else:
        signal_power = np.mean(signal ** 2)

    # 计算噪声功率
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # 生成复高斯噪声
    if np.iscomplexobj(signal):
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) +
                                          1j * np.random.randn(*signal.shape))
    else:
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)

    return signal + noise

# 对于每种目标类型生成微多普勒特征
for target_idx in range(num_targets):
    target_type = target_types[target_idx]

    # 根据目标类型设置参数
    if target_type == '行人':
        # 行人参数
        v_main = 1.5           # 主体速度 (m/s)
        v_arm = 0.8            # 手臂摆动速度 (m/s)
        v_leg = 1.2            # 腿部摆动速度 (m/s)
        f_arm = 1.2            # 手臂摆动频率 (Hz)
        f_leg = 1.8            # 腿部摆动频率 (Hz)
        arm_amp = 0.3          # 手臂摆动幅度 (m)
        leg_amp = 0.4          # 腿部摆动幅度 (m)

    elif target_type == '自行车':
        # 自行车参数
        v_main = 5             # 主体速度 (m/s)
        v_pedal = 2            # 踏板速度 (m/s)
        wheel_rpm = 200        # 车轮转速 (RPM)
        f_pedal = 1.5          # 踏板频率 (Hz)
        pedal_amp = 0.15       # 踏板幅度 (m)
        wheel_amp = 0.3        # 车轮幅度 (m)

    else:  # 汽车
        # 汽车参数
        v_main = 15            # 主体速度 (m/s)
        wheel_rpm = 300        # 车轮转速 (RPM)
        wheel_amp = 0.35       # 车轮幅度 (m)
        engine_vib_freq = 30   # 发动机振动频率 (Hz)
        engine_vib_amp = 0.02  # 发动机振动幅度 (m)

    # 生成微多普勒信号
    microdoppler_signal = np.zeros((M, N), dtype=complex)

    for m in range(M):
        t = m * T

        # 根据目标类型计算微多普勒频率
        if target_type == '行人':
            # 行人微多普勒模型
            arm_velocity = v_arm * np.sin(2 * np.pi * f_arm * t)
            leg_velocity = v_leg * np.sin(2 * np.pi * f_leg * t + np.pi / 4)
            microdoppler_freq = 2 / lambda_ * (v_main + arm_velocity + leg_velocity)

        elif target_type == '自行车':
            # 自行车微多普勒模型
            pedal_velocity = v_pedal * np.sin(2 * np.pi * f_pedal * t)
            wheel_velocity = (wheel_rpm * 2 * np.pi / 60) * wheel_amp * np.sin(2 * np.pi * (wheel_rpm / 60) * t)
            microdoppler_freq = 2 / lambda_ * (v_main + pedal_velocity + wheel_velocity)

        else:  # 汽车
            # 汽车微多普勒模型
            wheel_velocity = (wheel_rpm * 2 * np.pi / 60) * wheel_amp * np.sin(2 * np.pi * (wheel_rpm / 60) * t)
            engine_vibration = engine_vib_amp * np.sin(2 * np.pi * engine_vib_freq * t)
            microdoppler_freq = 2 / lambda_ * (v_main + wheel_velocity + engine_vibration)

        # 生成信号
        phase = 2 * np.pi * microdoppler_freq * t
        microdoppler_signal[m, :] = np.exp(1j * phase)

    # 添加噪声
    SNR = 20  # 信噪比 (dB)
    microdoppler_signal = awgn(microdoppler_signal, SNR)

    # 存储微多普勒特征
    microdoppler_features[target_idx] = microdoppler_signal

# 时频分析 - 生成微多普勒谱
plt.figure(figsize=(12, 8))
plt.suptitle('不同目标的微多普勒特征', fontsize=16, fontweight='bold')

for target_idx in range(num_targets):
    # 计算短时傅里叶变换(STFT)
    signal = microdoppler_features[target_idx]
    # 对信号进行平均处理
    signal_mean = np.mean(signal, axis=1)

    f, t, Sxx = spectrogram(signal_mean, fs=PRF, nperseg=64, noverlap=60, nfft=1024)

    # 使用imshow绘制微多普勒谱
    plt.subplot(2, 2, target_idx + 1)

    # 准备imshow的数据
    spectrogram_db = 20 * np.log10(np.abs(Sxx))

    # 使用imshow显示
    extent = [t[0], t[-1], f[0]/1e3, f[-1]/1e3]  # 设置坐标范围 [xmin, xmax, ymin, ymax]
    im = plt.imshow(spectrogram_db, extent=extent, aspect='auto', origin='lower',
                   cmap='jet', vmin=-50, vmax=0)

    plt.xlabel('时间 (s)')
    plt.ylabel('多普勒频率 (kHz)')
    plt.title(f'{target_types[target_idx]}的微多普勒谱')
    plt.colorbar(im)

plt.tight_layout()
plt.show()

# 特征提取与分类
# 提取微多普勒特征用于分类
features = []
labels = []

for target_idx in range(num_targets):
    signal = microdoppler_features[target_idx]

    # 计算多普勒频谱
    doppler_spectrum = np.fft.fftshift(np.fft.fft(signal, axis=0), axes=0)

    # 提取特征
    # 1. 频谱中心
    spectrum_center = np.sum(np.abs(doppler_spectrum) * frequency_axis.reshape(-1, 1), axis=0) / np.sum(np.abs(doppler_spectrum), axis=0)

    # 2. 频谱带宽
    spectrum_bandwidth = np.sqrt(
        np.sum(np.abs(doppler_spectrum) * (frequency_axis.reshape(-1, 1) ** 2), axis=0) /
        np.sum(np.abs(doppler_spectrum), axis=0) - spectrum_center ** 2
    )

    # 3. 频谱熵
    norm_spectrum = np.abs(doppler_spectrum) / np.sum(np.abs(doppler_spectrum), axis=0)
    spectrum_entropy = -np.sum(norm_spectrum * np.log(norm_spectrum + np.finfo(float).eps), axis=0)

    # 4. 频谱峰度
    spectrum_kurtosis = kurtosis(np.abs(doppler_spectrum), axis=0, fisher=False)

    # 组合特征
    target_features = np.array([
        np.mean(spectrum_center),
        np.mean(spectrum_bandwidth),
        np.mean(spectrum_entropy),
        np.mean(spectrum_kurtosis)
    ])

    features.append(target_features)
    labels.append(target_idx)

# 转换为numpy数组
features = np.array(features).T

# 显示提取的特征
feature_names = ['频谱中心', '频谱带宽', '频谱熵', '频谱峰度']
plt.figure(figsize=(10, 4))

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.bar(range(num_targets), features[i, :])
    plt.xticks(range(num_targets), target_types)
    plt.title(feature_names[i])
    plt.ylabel('特征值')

plt.suptitle('不同目标的微多普勒特征比较', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 打印特征值
print("提取的特征值:")
print(f"{'目标类型':<8} {'频谱中心':<12} {'频谱带宽':<12} {'频谱熵':<12} {'频谱峰度':<12}")
for i in range(num_targets):
    print(f"{target_types[i]:<8} {features[0, i]:<12.4f} {features[1, i]:<12.4f} "
          f"{features[2, i]:<12.4f} {features[3, i]:<12.4f}")




























