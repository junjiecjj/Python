#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 17:37:58 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzAwMDE1ODE5NA==&mid=2652542571&idx=1&sn=0e0eb494ac7ee19d18227a5e96c2b27e&chksm=80bf6e07fe6c25015962bf84b33dbeb965b1888425ff564db6106ac883dac81060463e60e10d&mpshare=1&scene=1&srcid=0329Q8dj1B90QMlepVAj2Um9&sharer_shareinfo=d81cf5ed53c71bcef317519f9e89a4c6&sharer_shareinfo_first=38d19dc84b14ff1c2d3b069947b97c9c&exportkey=n_ChQIAhIQoqc4TB2d4%2Bl9owvgjWTdPhKfAgIE97dBBAEAAAAAAJWAFhSLX1YAAAAOpnltbLcz9gKNyK89dVj0BVIztoOdHXe1vxTqTJvaLxfHJ6tw115eyMwDESf8NCzBxKT2XQAJ%2BMB4kPlH4oYW0tI4bMH1UPS0MnIbWxqSUerSApop8AdL7PbiviB0NhA%2F05slXiNpfwNkW%2FLmtynGJAkUpkeLvon3tDkz9jNs1VBv6rwbmkKCAjMvdfczj3djD3DTQPHjovK8iUzJM5Qqg4kjqVSlvFGvcpKePEj%2Bhu8OBCCjNC3AixaT7eaXmDTqobPfUzCHTsXiJSoj6BbM%2FzciYjjeI%2BMqFnlMikjW0J9hduxw0zJ%2BonzsJYrQABTvkQZGZyyB2phykV3KyVU2XCcLLGquMNdC&acctmode=0&pass_ticket=aqQR4ewhTuEsf86OFOGd8APM8RmUaaYiSmQJRnuUr9mvWRvnllyDKnrjKQnNwjVh&wx_header=0#rd


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

#%% 1，生成多个周期的脉冲信号，脉内调制为线性调频信号。
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

#%% 2，根据多个目标的延时生成目标的叠加回波，并添加噪声。
# 4. 模拟多个目标的回波
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

#%% 3，进行匹配滤波（快速卷积实现）
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

# 7. 可视化匹配滤波输出
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






















































