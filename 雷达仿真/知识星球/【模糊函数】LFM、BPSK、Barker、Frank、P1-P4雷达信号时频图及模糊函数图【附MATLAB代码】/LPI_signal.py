#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 20:56:27 2025

@author: jack
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, correlate
from scipy.fft import fft, fftshift
import matplotlib.gridspec as gridspec
from scipy.signal.windows import hamming

def nextpow2(n):
    """计算下一个2的幂"""
    return int(np.ceil(np.log2(n)))

def ambgfun(x, fs, PRF):
    """
    模糊函数计算
    """
    N = len(x)
    # 时延范围
    max_delay = int(N / 2)
    delays = np.arange(-max_delay, max_delay) / fs

    # 多普勒范围
    max_doppler = PRF / 2
    dopplers = np.linspace(-max_doppler, max_doppler, 256)

    # 初始化模糊函数矩阵
    afmag = np.zeros((len(dopplers), len(delays)))

    # 计算模糊函数
    for i, delay in enumerate(delays):
        delay_samples = int(delay * fs)
        for j, doppler in enumerate(dopplers):
            # 时延信号
            if delay_samples >= 0:
                x1 = x[:N-delay_samples]
                x2 = x[delay_samples:] * np.exp(1j * 2 * np.pi * doppler * np.arange(N-delay_samples) / fs)
            else:
                x1 = x[-delay_samples:]
                x2 = x[:N+delay_samples] * np.exp(1j * 2 * np.pi * doppler * np.arange(N+delay_samples) / fs)

            # 计算互相关
            afmag[j, i] = np.abs(np.sum(x1 * np.conj(x2)))

    # 归一化
    afmag = afmag / np.max(afmag)

    return afmag, delays, dopplers

def plot_signal_analysis(signal, fs, prf, signal_name):
    """绘制信号的完整分析图"""
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{signal_name}分析', fontsize=16, fontweight='bold')

    # 时域图
    ax1 = plt.subplot(2, 3, 1)
    t = np.arange(len(signal)) / fs * 1000  # 转换为毫秒
    plt.plot(t, np.real(signal))
    plt.xlabel('时间 (ms)')
    plt.ylabel('幅度')
    plt.title('时域图')
    plt.grid(True)

    # 频域图
    ax2 = plt.subplot(2, 3, 2)
    nfft = max(1024, len(signal))
    Z = fft(signal, nfft)
    fr = np.arange(0, nfft/2) / nfft * fs
    plt.plot(fr / 1000, np.abs(Z[:int(nfft/2)]))
    plt.xlabel('频率 (kHz)')
    plt.ylabel('幅度')
    plt.title('频域图')
    plt.grid(True)

    # 自相关图
    ax3 = plt.subplot(2, 3, 3)
    autocorr = correlate(signal, signal, mode='full')
    plt.plot(np.arange(len(autocorr)), np.abs(autocorr))
    plt.xlabel('采样点')
    plt.ylabel('幅度')
    plt.title('自相关图')
    plt.grid(True)

    # 时频域图
    ax4 = plt.subplot(2, 3, 4)
    nsc = int(len(signal) / 10)
    nov = int(nsc * 0.9)
    nff = max(256, 2**nextpow2(nsc))

    f, t_spec, Sxx = spectrogram(signal, fs=fs, window=hamming(nsc), noverlap=nov, nfft=nff, return_onesided=False)
    Sxx = fftshift(Sxx, axes=0)
    f = fftshift(f)

    plt.pcolormesh(t_spec * 1000, f / 1000, 10 * np.log10(np.abs(Sxx)), shading='gouraud')
    plt.colorbar(label='强度 (dB)')
    plt.ylabel('频率 (kHz)')
    plt.xlabel('时间 (ms)')
    plt.title('时频域图')

    # 模糊函数
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    afmag, delay, doppler = ambgfun(signal, fs, prf)
    X, Y = np.meshgrid(delay * 1e6, doppler / 1e3)
    surf = ax5.plot_surface(X, Y, afmag, cmap='viridis', linewidth=0, antialiased=True)
    ax5.set_xlabel('时延 τ (μs)')
    ax5.set_ylabel('多普勒 f_d (kHz)')
    ax5.set_zlabel('幅度')
    plt.title('模糊函数')

    # 模糊函数等高线
    ax6 = plt.subplot(2, 3, 6)
    contour = plt.contour(delay * 1e6, doppler / 1e3, afmag, levels=10)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel('时延 τ (μs)')
    plt.ylabel('多普勒 f_d (kHz)')
    plt.title('模糊函数等高线')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

###############################################################################
#                           LFM信号实现
###############################################################################
def generate_lfm(fs, sweep_bw, pulse_width, prf):
    """生成LFM信号"""
    t = np.arange(0, pulse_width, 1/fs)
    nsamp = len(t)

    # LFM参数
    k = sweep_bw / pulse_width  # 调频率
    lfm_signal = np.exp(1j * np.pi * k * t**2)

    return lfm_signal

###############################################################################
#                           Frank码实现
###############################################################################
def generate_frank_code(num_chips):
    """生成Frank码相位序列"""
    n = int(np.sqrt(num_chips))
    if n*n != num_chips:
        raise ValueError("Frank码的码片数必须是完全平方数")

    frank_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            frank_matrix[i, j] = 2 * np.pi * i * j / n

    return frank_matrix.flatten()

def generate_frank_signal(num_chips, fs, chip_width, prf):
    """生成Frank码信号"""
    phases = generate_frank_code(num_chips)
    samples_per_chip = int(chip_width * fs)
    signal = np.zeros(num_chips * samples_per_chip, dtype=complex)

    for i in range(num_chips):
        start_idx = i * samples_per_chip
        end_idx = (i + 1) * samples_per_chip
        signal[start_idx:end_idx] = np.exp(1j * phases[i])

    return signal

###############################################################################
#                           Barker码实现
###############################################################################
def generate_barker_code(num_chips):
    """生成Barker码相位序列"""
    barker_codes = {
        2: [0, np.pi],
        3: [0, 0, np.pi],
        4: [0, 0, np.pi, 0],
        5: [0, 0, 0, np.pi, 0],
        7: [0, 0, 0, np.pi, np.pi, 0, np.pi],
        11: [0, 0, 0, np.pi, np.pi, np.pi, 0, np.pi, np.pi, 0, np.pi],
        13: [0, 0, 0, 0, 0, np.pi, np.pi, 0, 0, np.pi, 0, np.pi, 0]
    }

    if num_chips not in barker_codes:
        raise ValueError(f"不支持的Barker码长度: {num_chips}")

    return barker_codes[num_chips]

def generate_barker_signal(num_chips, fs, chip_width, prf):
    """生成Barker码信号"""
    phases = generate_barker_code(num_chips)
    samples_per_chip = int(chip_width * fs)
    signal = np.zeros(num_chips * samples_per_chip, dtype=complex)

    for i in range(num_chips):
        start_idx = i * samples_per_chip
        end_idx = (i + 1) * samples_per_chip
        signal[start_idx:end_idx] = np.exp(1j * phases[i])

    return signal

###############################################################################
#                           P1码实现
###############################################################################
def generate_p1_code(num_chips):
    """生成P1码相位序列"""
    n = int(np.sqrt(num_chips))
    if n*n != num_chips:
        raise ValueError("P1码的码片数必须是完全平方数")

    p1_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            p1_matrix[i, j] = -np.pi / n * (n - (2*j + 1)) * (j * n + i)

    return p1_matrix.flatten()

def generate_p1_signal(num_chips, fs, chip_width, prf):
    """生成P1码信号"""
    phases = generate_p1_code(num_chips)
    samples_per_chip = int(chip_width * fs)
    signal = np.zeros(num_chips * samples_per_chip, dtype=complex)

    for i in range(num_chips):
        start_idx = i * samples_per_chip
        end_idx = (i + 1) * samples_per_chip
        signal[start_idx:end_idx] = np.exp(1j * phases[i])

    return signal

###############################################################################
#                           P2码实现
###############################################################################
def generate_p2_code(num_chips):
    """生成P2码相位序列"""
    n = int(np.sqrt(num_chips))
    if n*n != num_chips:
        raise ValueError("P2码的码片数必须是完全平方数")

    p2_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i <= n/2:
                p2_matrix[i, j] = np.pi / n * (n - (2*j + 1)) * ((n - 1)/2 - i)
            else:
                p2_matrix[i, j] = np.pi / n * (n - (2*j + 1)) * ((n - 1)/2 - i) + np.pi

    return p2_matrix.flatten()

def generate_p2_signal(num_chips, fs, chip_width, prf):
    """生成P2码信号"""
    phases = generate_p2_code(num_chips)
    samples_per_chip = int(chip_width * fs)
    signal = np.zeros(num_chips * samples_per_chip, dtype=complex)

    for i in range(num_chips):
        start_idx = i * samples_per_chip
        end_idx = (i + 1) * samples_per_chip
        signal[start_idx:end_idx] = np.exp(1j * phases[i])

    return signal

###############################################################################
#                           P3码实现
###############################################################################
def generate_p3_code(num_chips):
    """生成P3码相位序列"""
    phases = np.zeros(num_chips)
    for i in range(num_chips):
        phases[i] = np.pi * (i**2) / num_chips

    return phases

def generate_p3_signal(num_chips, fs, chip_width, prf):
    """生成P3码信号"""
    phases = generate_p3_code(num_chips)
    samples_per_chip = int(chip_width * fs)
    signal = np.zeros(num_chips * samples_per_chip, dtype=complex)

    for i in range(num_chips):
        start_idx = i * samples_per_chip
        end_idx = (i + 1) * samples_per_chip
        signal[start_idx:end_idx] = np.exp(1j * phases[i])

    return signal

###############################################################################
#                           P4码实现
###############################################################################
def generate_p4_code(num_chips):
    """生成P4码相位序列"""
    phases = np.zeros(num_chips)
    for i in range(num_chips):
        phases[i] = np.pi * i * (i - num_chips) / num_chips

    return phases

def generate_p4_signal(num_chips, fs, chip_width, prf):
    """生成P4码信号"""
    phases = generate_p4_code(num_chips)
    samples_per_chip = int(chip_width * fs)
    signal = np.zeros(num_chips * samples_per_chip, dtype=complex)

    for i in range(num_chips):
        start_idx = i * samples_per_chip
        end_idx = (i + 1) * samples_per_chip
        signal[start_idx:end_idx] = np.exp(1j * phases[i])

    return signal

###############################################################################
#                           Costas跳频编码实现
###############################################################################
def generate_costas_sequence(length):
    """生成Costas序列"""
    # 已知的Costas序列
    known_sequences = {
        3: [1, 3, 2],
        4: [1, 3, 4, 2],
        5: [1, 3, 5, 2, 4],
        6: [1, 3, 6, 4, 2, 5],
        7: [1, 3, 6, 4, 7, 5, 2],
        8: [1, 3, 6, 8, 4, 7, 5, 2],
        9: [1, 3, 6, 8, 5, 9, 7, 4, 2],
        10: [1, 3, 6, 10, 8, 5, 9, 7, 4, 2],
        11: [1, 3, 6, 10, 8, 11, 5, 9, 7, 4, 2],
        12: [1, 3, 6, 10, 12, 8, 5, 11, 9, 7, 4, 2]
    }

    if length in known_sequences:
        return known_sequences[length]
    else:
        # 对于更大的长度，生成随机排列
        return np.random.permutation(range(1, length+1))

def generate_costas_signal(costas_seq, base_freq, fs, pulse_width, prf):
    """生成Costas跳频信号"""
    num_freqs = len(costas_seq)
    samples_per_pulse = int(pulse_width * fs)

    costas_signal = []
    for freq_idx in costas_seq:
        # 生成单个频率的脉冲
        t = np.linspace(0, pulse_width, samples_per_pulse, endpoint=False)
        freq = base_freq * freq_idx
        pulse = np.exp(1j * 2 * np.pi * freq * t)
        costas_signal.extend(pulse)

    return np.array(costas_signal)

###############################################################################
#                           主程序
###############################################################################
def main():
    # 通用参数
    c = 3e8

    ###########################################################################
    #                           LFM信号
    ###########################################################################
    print("=" * 60)
    print("LFM信号分析")
    print("=" * 60)

    fs_lfm = 500e3
    sweep_bw = 200e3
    pulse_width = 1e-3
    prf_lfm = 1e3

    lfm_signal = generate_lfm(fs_lfm, sweep_bw, pulse_width, prf_lfm)
    plot_signal_analysis(lfm_signal, fs_lfm, prf_lfm, "LFM信号")

    ###########################################################################
    #                           Frank码信号
    ###########################################################################
    print("=" * 60)
    print("Frank码信号分析")
    print("=" * 60)

    Rmax = 200
    Rres = 5
    prf_phase = c / (2 * Rmax)
    bw = c / (2 * Rres)
    fs_phase = 2 * bw

    frank_signal = generate_frank_signal(25, fs_phase, 1/bw, prf_phase)
    plot_signal_analysis(frank_signal, fs_phase, prf_phase, "Frank码")

    ###########################################################################
    #                           Barker码信号
    ###########################################################################
    print("=" * 60)
    print("Barker码信号分析")
    print("=" * 60)

    barker_signal = generate_barker_signal(13, fs_phase, 1/bw, prf_phase)
    plot_signal_analysis(barker_signal, fs_phase, prf_phase, "Barker码")

    ###########################################################################
    #                           P1码信号
    ###########################################################################
    print("=" * 60)
    print("P1码信号分析")
    print("=" * 60)

    p1_signal = generate_p1_signal(25, fs_phase, 1/bw, prf_phase)
    plot_signal_analysis(p1_signal, fs_phase, prf_phase, "P1码")

    ###########################################################################
    #                           P2码信号
    ###########################################################################
    print("=" * 60)
    print("P2码信号分析")
    print("=" * 60)

    p2_signal = generate_p2_signal(16, fs_phase, 1/bw, prf_phase)
    plot_signal_analysis(p2_signal, fs_phase, prf_phase, "P2码")

    ###########################################################################
    #                           P3码信号
    ###########################################################################
    print("=" * 60)
    print("P3码信号分析")
    print("=" * 60)

    p3_signal = generate_p3_signal(25, fs_phase, 1/bw, prf_phase)
    plot_signal_analysis(p3_signal, fs_phase, prf_phase, "P3码")

    ###########################################################################
    #                           P4码信号
    ###########################################################################
    print("=" * 60)
    print("P4码信号分析")
    print("=" * 60)

    p4_signal = generate_p4_signal(25, fs_phase, 1/bw, prf_phase)
    plot_signal_analysis(p4_signal, fs_phase, prf_phase, "P4码")

    ###########################################################################
    #                           Costas跳频信号
    ###########################################################################
    print("=" * 60)
    print("Costas跳频信号分析")
    print("=" * 60)

    costas_seq = generate_costas_sequence(10)
    base_freq = 10e3
    fs_costas = len(costas_seq) * base_freq * 2

    costas_signal = generate_costas_signal(costas_seq, base_freq, fs_costas, 1e-3, 1e3)
    plot_signal_analysis(costas_signal, fs_costas, 1e3, "Costas跳频")

if __name__ == "__main__":
    main()
