#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 17:36:08 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzk0MDcwNzM2Nw==&mid=2247484214&idx=1&sn=2fdbf25e18e00b58744d9c40d240961c&chksm=c3903a2eccfdc2c3a9a834e0d24a96409ba27e919ee2886d6884ee689da7d8e3629e892518af&mpshare=1&scene=1&srcid=09058K3ChBVMkcOkirtjf3Fv&sharer_shareinfo=347c5441e2f3487e24efc713f1d0f3cd&sharer_shareinfo_first=347c5441e2f3487e24efc713f1d0f3cd&exportkey=n_ChQIAhIQjPWvG%2FdzAvD07KyCBaj6mRKfAgIE97dBBAEAAAAAAFzhAb3E56gAAAAOpnltbLcz9gKNyK89dVj0MQS7Cnm9H6HuD3W7kAcgQTBib571QklIxGjLRjB2s5Ehxr%2B%2Fra7WkX28188tswsiITVoW82BS5I7wE8UNsHIvMCuXry%2B%2F8gIKH1aXF8GB4fhm8ydAPUrnamQjrg8GrgJt7rxiDPIuNkpSk1Gn00I4quVaeuFjbd%2BAeADkK3KLVleOrBsPgREm%2Fc7Euyi%2BhLUe1pic30zUMSV4KUVdoaOmDr0N0QZIQzezeIYYtDUJyq%2Be%2FRqpPigmsuNAHRgB6J0yBpQTtz8Jk7B8yLCdmHcl0BsJU8%2Fb9XcoExgptWmxFpIb%2F9EMiNm9KAp1QWzg%2BLS4IwJfgc8VIPW&acctmode=0&pass_ticket=sDqeVrLzVma7Qe1U2mqsGzZfIHZkY4JSJhMKkF1sueQQLRbQcpTvFm1xD4iZnYSL&wx_header=0#rd


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def computeAmbiguityFunction(signal, fs, maxDoppler, maxDelay):
    """
    Calculating auto-ambiguity function
    Inputs:
    signal - input signal
    fs - sampling frequency
    maxDoppler - maximum Doppler frequency shift (Hz)
    maxDelay - maximum time delay (s)
    Outputs:
    ambiguity - value of ambiguity function
    delay - value of delay axis
    doppler - value of Doppler axis
    """
    N = len(signal)  # signal length
    dopplerShifts = np.linspace(-maxDoppler, maxDoppler, 2 * N)
    delays = np.arange(0, maxDelay, 1/fs)
    ambiguity = np.zeros((len(delays), len(dopplerShifts)), dtype=complex)

    # Calculating the influence of each Doppler shift in advance
    t = np.arange(N) / fs
    dopplerMat = np.exp(1j * 2 * np.pi * t.reshape(-1, 1) @ dopplerShifts.reshape(1, -1))

    # Calculating auto-ambiguity function
    for k in range(len(dopplerShifts)):
        # Doppler frequency shift is applied to the original signal
        dopplerSignal = signal * dopplerMat[:, k]

        for i in range(len(delays)):
            # Generating a delayed signal
            delayIndex = int(round(delays[i] * fs))
            if delayIndex >= N:
                delayedSignal = np.zeros(N)  # If the delay exceeds the signal length, use all-zero vector
            else:
                delayedSignal = np.concatenate([np.zeros(delayIndex), signal[:N - delayIndex]])

            # Calculate the auto-ambiguity value under the current delay and Doppler frequency shift
            ambiguity[i, k] = np.abs(np.sum(delayedSignal[:N] * np.conj(dopplerSignal)))

    # Return delay and Doppler frequency shift
    return ambiguity, delays, dopplerShifts

def computeCrossAF(sig1, sig2, fs, maxDoppler, maxDelay, tstart=0):
    """
    Calculate the cross-ambiguity function between two signals
    Inputs:
    sig1 - emitted signal, sig2 - echo signal
    fs - sampling frequency
    maxDoppler - maximum Doppler frequency shift (Hz)
    maxDelay - maximum time delay (s)
    tstart - starting time (default=0)
    Outputs:
    CAF - value of Cross-Ambiguity Function
    tau - value of delay axis(s)
    fd - value of Doppler axis(Hz)
    """
    N = len(sig2)  # Define N using the length of sig2
    # Ensure that the length of sig1 is at least N
    if len(sig1) < N:
        sig1 = np.concatenate([sig1, np.zeros(N - len(sig1))])
    else:
        sig1 = sig1[:N]  # Cut to match the length

    fd = np.linspace(-maxDoppler, maxDoppler, 2 * N)
    tau = np.arange(tstart, maxDelay, 1/fs)
    CAF = np.zeros((len(tau), len(fd)), dtype=complex)

    # Calculating Doppler frequency shift matrix
    t = np.arange(N) / fs
    dopplerMat = np.exp(-1j * 2 * np.pi * t.reshape(-1, 1) @ fd.reshape(1, -1))

    # Calculate the cross-ambiguity function
    for i in range(len(tau)):
        delayIndex = int(round(tau[i] * fs))
        if delayIndex >= N:
            delayedSig1 = np.zeros(N)  # Delay exceeds signal length, using all zero vector
        else:
            delayedSig1 = np.concatenate([np.zeros(delayIndex), sig1[:N - delayIndex]])

        # Calculate CAF under all Doppler frequency shifts
        for j in range(len(fd)):
            dopplerShiftedSig1 = np.conj(delayedSig1) * dopplerMat[:, j]
            CAF[i, j] = np.abs(sig2 @ dopplerShiftedSig1)

    return CAF, tau, fd

# 主程序
if __name__ == "__main__":
    # 参数设置
    fs = 50e3  # sampling frequency 50kHz
    T = 0.1  # duration
    B = 1/T  # bandwidth
    p_wid = 0.02  # single-period pulse width
    pcw4_t = 0.005  # four-period pulse width
    t = np.arange(0, T, 1/fs)
    t1 = np.arange(0, p_wid, 1/fs)
    t_pcw4 = np.arange(0, pcw4_t, 1/fs)
    prd = 0.025  # pulse repetition interval
    prd_sam = int(0.025 * fs)  # Pulse repetition interval sampling points
    A_pcw = 1  # amplitude
    f_pcw = 7e3  # center frequency 7kHz
    I = 4  # pulse number
    c = 1500  # speed of sound

    # 生成信号
    y_pcw1 = A_pcw * np.cos(2 * np.pi * f_pcw * t1)
    pcw4_1 = A_pcw * np.cos(2 * np.pi * f_pcw * t_pcw4)

    # 生成单周期PCW信号
    y_pcw = np.concatenate([y_pcw1, np.zeros(len(t) - len(t1))])

    # 生成四周期PCW信号
    pcw4 = np.zeros(len(t))
    for i in range(I):
        start_idx = i * prd_sam
        end_idx = start_idx + len(pcw4_1)
        if end_idx <= len(t):
            pcw4[start_idx:end_idx] = pcw4_1

    # 计算频谱
    N_fft = 2**int(np.ceil(np.log2(fs * T)))  # DFT points
    Y = np.fft.fft(y_pcw, N_fft) / N_fft * 2  # single-period PCW signal spectrum
    Y_pcw4 = np.fft.fft(pcw4, N_fft) / N_fft * 2  # four-period PCW signal spectrum
    f = np.arange(N_fft) * fs / N_fft  # Frequency axis
    P = np.abs(Y)  # single-period PCW signal magnitude spectrum
    P_pcw4 = np.abs(Y_pcw4)  # four-period PCW signal magnitude spectrum

    # 计算模糊函数
    maxDelay = 0.1  # maximum delay of calculation
    maxDoppler = 1000  # calculated maximum Doppler frequency shift

    af_pcw, delay, doppler = computeAmbiguityFunction(y_pcw, fs, maxDoppler, maxDelay)
    af_pcw4, delay_4, doppler_4 = computeAmbiguityFunction(pcw4, fs, maxDoppler, maxDelay)

    # 处理数据用于绘图
    af_pcw_d = np.concatenate([np.fliplr(af_pcw.T), af_pcw.T], axis=1)
    delay_d = np.concatenate([-np.flip(delay), delay])
    a_nom_max = np.max(af_pcw_d)
    af_pcw_d_nom = af_pcw_d / a_nom_max  # normalize

    af_pcw4_d = np.concatenate([np.fliplr(af_pcw4.T), af_pcw4.T], axis=1)
    delay_4_d = np.concatenate([-np.flip(delay_4), delay_4])
    a_nom_max4 = np.max(af_pcw4_d)
    af_pcw4_d_nom = af_pcw4_d / a_nom_max4  # normalize

    # 绘制时域和频域波形
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(t, y_pcw)
    plt.xlabel('t/s')
    plt.ylabel('magnitude/v')
    plt.title('Single PCW - Time Domain Waveform')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(f[:N_fft//2+1], P[:N_fft//2+1])
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude')
    plt.title('Single PCW - Frequency Spectrum')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(t, pcw4)
    plt.xlabel('t/s')
    plt.ylabel('magnitude/v')
    plt.title('Four PCW - Time Domain Waveform')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(f[:N_fft//2+1], P_pcw4[:N_fft//2+1])
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude')
    plt.title('Four PCW - Frequency Spectrum')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 绘制模糊函数
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    D, T = np.meshgrid(doppler, delay_d)
    surf1 = ax1.plot_surface(T, D, af_pcw_d_nom.T, cmap=cm.jet, alpha=0.8)
    ax1.set_xlabel('Time Delay (s)')
    ax1.set_ylabel('Doppler Frequency Shift (Hz)')
    ax1.set_zlabel('Normalized Amplitude')
    ax1.set_title('Single PCW Auto-AF')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(122, projection='3d')
    D4, T4 = np.meshgrid(doppler_4, delay_4_d)
    surf2 = ax2.plot_surface(T4, D4, af_pcw4_d_nom.T, cmap=cm.jet, alpha=0.8)
    ax2.set_xlabel('Time Delay (s)')
    ax2.set_ylabel('Doppler Frequency Shift (Hz)')
    ax2.set_zlabel('Normalized Amplitude')
    ax2.set_title('Four PCW Auto-AF (fc=7k, num=4, t=5ms, T=25ms)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    plt.tight_layout()
    plt.show()

    # 打印信息
    print("信号参数:")
    print(f"采样频率: {fs/1000} kHz")
    print(f"信号时长: {T} s")
    print(f"中心频率: {f_pcw/1000} kHz")
    print(f"单周期脉宽: {p_wid*1000} ms")
    print(f"四周期脉宽: {pcw4_t*1000} ms")
    print(f"脉冲重复周期: {prd*1000} ms")
    print(f"脉冲数量: {I}")
