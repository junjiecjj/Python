#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 23:58:12 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247489978&idx=1&sn=8ed933fc275af974846a6f3be00f05d8&chksm=c06f60d3542de596d383a4c3322e1e0f33bd6853ad0962f49be14baaff7c1126ed51d0912525&mpshare=1&scene=1&srcid=0323bPMO68rvEFUU8aa5QnQY&sharer_shareinfo=cf22eefea6d212ac1867dcdaee6a8788&sharer_shareinfo_first=cf22eefea6d212ac1867dcdaee6a8788&exportkey=n_ChQIAhIQrzTnyknZZ2pOw7YOXO15HBKfAgIE97dBBAEAAAAAAKc4N%2B%2BIx4AAAAAOpnltbLcz9gKNyK89dVj0bIgzCpeuPa34D1Ov6V3ZVNbFSz830ZSINdOhiMO4Uw3qKUZFF%2FImjJO464ckbuOZkdSe4h1DJcnocX0ZxNrUBDOpDrKjOASUS8g8h3qrKw38eqEqDov7zgh7O9awFsoWefnY9rAKjSSjR2lhrmRH6icJX1x97e90jc%2FWoOgVyyTbCDDG8uDHbot7VmRc572NQq5ztzDZrGerQDeD%2BJ7%2BZrNugOG0ZauOW%2FkfU36c8T7oc3xiHMNI4imMMqMFS7UEPlluvQR%2FQaLpP1%2B9T8dm58YFWYOji4dCBTENOtiiLeOpPF4l71R1NrLA3OBDCfCKsI7%2BGmtxu%2FBD&acctmode=0&pass_ticket=l3Xl3zfrRyJIluhuYJTPnj02ELo%2F%2Fw4SEt9eaw9t0FoT7Ao94AINqNgjZ5nk%2FjXv&wx_header=0#rd

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 代码内容
c = 3e8  # 光速
# FMCW 波形参数
fc = 77.5e9    # Carrier Frequency (Hz)
bw = 250e6     # Bandwidth (Hz)
fr = 1.9531e4  # Chirp Frequency (Hz)
tr = 1 / fr    # Chirp period (Hz)
ns = 512       # 采样点数
tg = np.linspace(0, tr, ns)  # 生成时间向量
"""
FMCW发射信号生成
"""
# 初始化
st = np.zeros(ns, dtype = complex)  # transmitted signal
ft = np.zeros(ns)  # frequency of transmitted signal/sampling freq.

# Sampling Frequency (Hz)
for i in range(len(tg)):
    tg_i = tg[i]  # take the time value of the tg vector
    # for the chirp period after the first period
    # while tg_i > tr:
        # tg_i -= tr
    # tg_i -= tr / 2  # so that the frequency ft becomes the middle frequency

    st[i] = np.cos(2 * np.pi * fc * tg_i + np.pi * bw * fr * (tg_i ** 2))
    ft[i] = fc + bw * fr * tg_i  # 中频频率

"""
FMCW波形u绘制
"""
# FMCW 发射信号
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(tg, st)
plt.title('FMCW Signal at Transmitter')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.grid(True)

# Frequency Variation
plt.subplot(2, 1, 2)
plt.plot(tg, ft / 1000)
plt.title('Frequency Variation of FMCW Signal at Transmitter')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 目标参数
Rt = 30  # 目标参数 (m)
Vt = 0  # 目标径向速度 (m/s)
"""
FMCW Signal Generation at Receiver/Reflected Signal
"""
# initial condition
srx = np.zeros(ns, dtype=complex)  # receiver signal that transmitted
frx = np.zeros(ns)  # frequency of received signal

for i in range(len(tg)):
    tg_i = tg[i]  # take the time value of the tg vector
    # calculate the time delay for receiving signals due to the distance factor
    t_delay = 2 * Rt / c
    # calculates the signal time delay due to target movement
    t_delay += (2 * tg_i * Vt / c)
    tg_i -= t_delay
    # for the chirp period after the first period
    while tg_i > tr:
        tg_i -= tr
    tg_i -= tr / 2  # so that the frequency ft becomes the middle frequency
    srx[i] = np.cos(2 * np.pi * fc * tg_i + np.pi * bw * fr * (tg_i ** 2))
    frx[i] = fc + bw * fr * tg_i
"""
FMCW接收信号
"""
# FMCW接收信号
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(tg, st)
plt.title('FMCW Signal at Receiver')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.grid(True)

# 频率变化
plt.subplot(2, 1, 2)
plt.plot(tg, ft / 1000, 'b')
plt.plot(tg, frx / 1000, 'r')
plt.title('Frequency Variation of FMCW Signal at Receiver')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)')
plt.grid(True)
plt.tight_layout()
plt.show()

"""
添加高斯白噪声
"""
def noise(sig, req_snr):
    sig_energy = np.linalg.norm(sig) ** 2  # energy of the signal
    noise_energy = sig_energy / (10 ** (req_snr / 10))  # energy of noise to be added
    noise_var = noise_energy / (len(sig) - 1)  # variance of noise
    noise_std = np.sqrt(noise_var)  # std. deviation of noise
    noise = noise_std * np.random.randn(*sig.shape)  # noise
    noisy_sig = sig + noise  # noisy signal
    return noisy_sig

srn = noise(srx, 15)
"""
Plot Signals with and without Noise
"""
# w/o Noise
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(tg, srx, 'r')
plt.title('FMCW Signal from Receiver without Noise')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.grid(True)

# w/ Noise
plt.subplot(2, 1, 2)
plt.plot(tg, srn, 'm')
plt.title('FMCW Signal from Receiver with Noise')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 混频
s_demod = srx * st
n_fft = 2 ** int(np.ceil(np.log2(len(s_demod))))
s_fft = np.fft.fftshift(np.fft.fft(s_demod, n_fft) / len(tg))
f_axis = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1 / ns))
# Plot with Demodulation
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(tg, s_demod)
plt.title('Demodulated Received FMCW Signal in t domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(f_axis, np.abs(s_fft))
plt.xlim([-ns / 2, ns / 2])
plt.title('Demodulated Received FMCW Signal in f domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (V)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 通过使用FFT中信号的中值来识别峰值频率
m = len(s_fft) // 2
peak, _ = find_peaks(np.abs(s_fft[m:]), height=0.2)
fd = f_axis[m + peak]
print(f"Peak Frequencies: {fd} Hz")

# Calculate Target Range
rest = c * fd / (2 * bw)
print(f"Estimated target range: {rest} m")
































































