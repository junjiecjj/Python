#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 17:42:21 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, lfilter
from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
from DigiCommPy.channels import awgn
from DigiCommPy.errorRates import ser_awgn

import warnings
warnings.filterwarnings('ignore')

# Parameters
N = 64                 # Number of subcarriers
M = 4                  # Modulation order (QPSK)
numSymbols = 1000      # Number of symbols
SNR_dB = np.arange(0, 21, 2)  # SNR range in dB
numIter = 100          # Number of iterations for averaging

# Multipath Channel
channel = np.array([0.8, 0.4, 0.2])  # Simple multipath channel

# Radar Parameters
max_delay = 32         # Maximum delay for ambiguity function
max_doppler = 0.1      # Maximum Doppler frequency (normalized)
delay_bins = np.arange(-max_delay, max_delay + 1)
doppler_bins = np.linspace(-max_doppler, max_doppler, 64)

# Initialize BER arrays
BER_OFDM = np.zeros(len(SNR_dB))
BER_GFDM = np.zeros(len(SNR_dB))
BER_FBMC = np.zeros(len(SNR_dB))

# Initialize Radar Performance arrays
range_resolution_OFDM = np.zeros(len(SNR_dB))
range_resolution_GFDM = np.zeros(len(SNR_dB))
range_resolution_FBMC = np.zeros(len(SNR_dB))

doppler_resolution_OFDM = np.zeros(len(SNR_dB))
doppler_resolution_GFDM = np.zeros(len(SNR_dB))
doppler_resolution_FBMC = np.zeros(len(SNR_dB))

PSL_OFDM = np.zeros(len(SNR_dB))  # Peak Sidelobe Level
PSL_GFDM = np.zeros(len(SNR_dB))
PSL_FBMC = np.zeros(len(SNR_dB))

mod_type = 'PSK'       # Set 'PSK' or 'QAM' or 'PAM' or 'FSK'
M = 4                  # array of M values to simulate
coherence = 'coherent' #'coherent'/'noncoherent'-only for FSK
modem_dict = {'psk': PSKModem, 'qam':QAMModem, 'pam':PAMModem, 'fsk':FSKModem}
modem = modem_dict[mod_type.lower()](M)

# def qpsk_modulate(data, M=4):
#     """QPSK Modulation"""
#     return np.exp(1j * (2 * np.pi * data / M + np.pi/4))

# def qpsk_demodulate(signal, M=4):
#     """QPSK Demodulation"""
#     phase = np.angle(signal)
#     phase = (phase - np.pi/4) % (2 * np.pi)
#     return np.floor(phase * M / (2 * np.pi)).astype(int)

def awgn(signal, snr_dB):
    """Add AWGN noise"""
    snr_linear = 10**(snr_dB / 10)
    power = np.mean(np.abs(signal)**2)
    noise_power = power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

def gfdm_modulate(data, N, M):
    """Improved GFDM modulation with circular filtering"""
    K = 7  # Number of subsymbols
    gfdmSymbols = np.zeros((N, data.shape[1]), dtype=complex)

    for k in range(K):
        # Apply circular shift and filter
        shift_data = np.roll(data, k, axis=0)
        filtered = np.fft.ifft(shift_data, axis=0) * np.exp(1j * 2 * np.pi * k * np.arange(N)[:, np.newaxis] / N)
        gfdmSymbols += filtered

    return gfdmSymbols / np.sqrt(K)

def gfdm_demodulate(gfdmRx, N, M):
    """Improved GFDM demodulation"""
    K = 7  # Number of subsymbols
    gfdmData = np.zeros_like(gfdmRx, dtype=complex)

    for k in range(K):
        filtered = np.fft.fft(gfdmRx * np.exp(-1j * 2 * np.pi * k * np.arange(N)[:, np.newaxis] / N), axis=0)
        gfdmData += np.roll(filtered, -k, axis=0)

    return gfdmData / np.sqrt(K)

def fbmc_modulate(data, N, M):
    """Improved FBMC modulation with prototype filter"""
    fbmcSymbols = np.fft.ifft(data, axis=0)
    # Apply prototype filter (simplified)
    prototype_filter = np.sqrt(2) * np.sin(np.pi * np.arange(N) / (N-1))
    return fbmcSymbols * prototype_filter[:, np.newaxis]

def fbmc_demodulate(fbmcRx, N, M):
    """Improved FBMC demodulation"""
    prototype_filter = np.sqrt(2) * np.sin(np.pi * np.arange(N) / (N-1))
    return np.fft.fft(fbmcRx * prototype_filter[:, np.newaxis], axis=0)

def calculate_ambiguity_function(signal, delay_bins, doppler_bins):
    """Calculate ambiguity function"""
    L = len(signal)
    amb = np.zeros((len(doppler_bins), len(delay_bins)))
    # Ensure signal is column vector
    signal = signal.flatten()
    for i, delay in enumerate(delay_bins):
        for j, doppler in enumerate(doppler_bins):
            # Handle delay
            if delay >= 0:
                # Positive delay: signal1 from start to L-delay, signal2 from delay+1 to end
                if delay < L:
                    signal1 = signal[:L-delay]
                    signal2 = signal[delay:L]
                else:
                    signal1 = np.array([])
                    signal2 = np.array([])
            else:
                # Negative delay: swap signals
                delay = abs(delay)
                if delay < L:
                    signal1 = signal[delay:L]
                    signal2 = signal[:L-delay]
                else:
                    signal1 = np.array([])
                    signal2 = np.array([])

            if len(signal1) > 0 and len(signal2) > 0 and len(signal1) == len(signal2):
                # Apply Doppler shift
                t = np.arange(len(signal1))
                doppler_phase = np.exp(1j * 2 * np.pi * doppler * t)

                # Calculate correlation
                corr_val = np.sum(signal1 * np.conj(signal2) * doppler_phase)
                amb[j, i] = np.abs(corr_val)
            else:
                amb[j, i] = 0

    # Normalize
    if np.max(amb) > 0:
        amb = amb / np.max(amb)

    return amb

def extract_radar_metrics(amb, delay_bins, doppler_bins):
    """Extract radar performance metrics"""
    num_doppler, num_delay = amb.shape

    # Range resolution (使用零多普勒剖面)
    zero_doppler_idx = np.argmin(np.abs(doppler_bins))
    zero_doppler_profile = amb[zero_doppler_idx, :]

    # 找到主瓣峰值
    max_val = np.max(zero_doppler_profile)
    max_idx = np.argmax(zero_doppler_profile)
    # 寻找-3dB点
    threshold = max_val * 10**(-3/20)  # -3dB threshold
    # 向左寻找-3dB点
    left_idx = max_idx
    while left_idx > 0 and zero_doppler_profile[left_idx] > threshold:
        left_idx -= 1
    # 向右寻找-3dB点
    right_idx = max_idx
    while right_idx < num_delay - 1 and zero_doppler_profile[right_idx] > threshold:
        right_idx += 1

    # 计算范围分辨率（延迟方向的-3dB宽度）
    if left_idx >= 0 and right_idx < num_delay:
        range_res = abs(delay_bins[right_idx] - delay_bins[left_idx])
    else:
        range_res = abs(delay_bins[1] - delay_bins[0]) * 4  # Default value

    # Doppler resolution (使用零延迟剖面)
    zero_delay_idx = np.argmin(np.abs(delay_bins))
    zero_delay_profile = amb[:, zero_delay_idx]
    # 找到主瓣峰值
    max_val = np.max(zero_delay_profile)
    max_idx = np.argmax(zero_delay_profile)
    # 寻找-3dB点
    threshold = max_val * 10**(-3/20)  # -3dB threshold
    # 向下寻找-3dB点
    down_idx = max_idx
    while down_idx > 0 and zero_delay_profile[down_idx] > threshold:
        down_idx -= 1
    # 向上寻找-3dB点
    up_idx = max_idx
    while up_idx < num_doppler - 1 and zero_delay_profile[up_idx] > threshold:
        up_idx += 1
    # 计算多普勒分辨率（多普勒方向的-3dB宽度）
    if down_idx >= 0 and up_idx < num_doppler:
        doppler_res = abs(doppler_bins[up_idx] - doppler_bins[down_idx])
    else:
        doppler_res = abs(doppler_bins[1] - doppler_bins[0]) * 4  # Default value

    # Peak Sidelobe Level
    max_val = np.max(amb)
    max_idx_flat = np.argmax(amb)
    max_row, max_col = np.unravel_index(max_idx_flat, amb.shape)

    # 创建主瓣掩码（排除主瓣区域）
    main_lobe_region = 3  # 主瓣区域大小
    row_range = slice(max(0, max_row - main_lobe_region), min(num_doppler, max_row + main_lobe_region + 1))
    col_range = slice(max(0, max_col - main_lobe_region), min(num_delay, max_col + main_lobe_region + 1))

    # 将主瓣区域设置为最小值
    sidelobes = amb.copy()
    sidelobes[row_range, col_range] = 0

    # 找到最大旁瓣
    max_sidelobe = np.max(sidelobes)

    if max_sidelobe > 0:
        psl = 20 * np.log10(max_sidelobe / max_val)
    else:
        psl = -100  # 如果没有旁瓣，设置为很低的值

    return range_res, doppler_res, psl

def plot_ambiguity_functions(ofdmSignal, gfdmSignal, fbmcSignal, delay_bins, doppler_bins):
    """Plot ambiguity functions"""
    sig_len = 256  # Fixed length

    # Ensure signals are long enough
    ofdm_segment = ofdmSignal[:min(sig_len, len(ofdmSignal))]
    gfdm_segment = gfdmSignal[:min(sig_len, len(gfdmSignal))]
    fbmc_segment = fbmcSignal[:min(sig_len, len(fbmcSignal))]

    # Pad if signals are too short
    if len(ofdm_segment) < sig_len:
        ofdm_segment = np.pad(ofdm_segment, (0, sig_len - len(ofdm_segment)))
    if len(gfdm_segment) < sig_len:
        gfdm_segment = np.pad(gfdm_segment, (0, sig_len - len(gfdm_segment)))
    if len(fbmc_segment) < sig_len:
        fbmc_segment = np.pad(fbmc_segment, (0, sig_len - len(fbmc_segment)))

    # Calculate ambiguity functions
    amb_ofdm = calculate_ambiguity_function(ofdm_segment, delay_bins, doppler_bins)
    amb_gfdm = calculate_ambiguity_function(gfdm_segment, delay_bins, doppler_bins)
    amb_fbmc = calculate_ambiguity_function(fbmc_segment, delay_bins, doppler_bins)

    fig = plt.figure(figsize=(15, 5))

    # OFDM Ambiguity Function
    plt.subplot(1, 3, 1)
    plt.imshow(20 * np.log10(np.maximum(amb_ofdm, 1e-6)), extent=[delay_bins[0], delay_bins[-1], doppler_bins[0], doppler_bins[-1]], aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Delay (samples)')
    plt.ylabel('Doppler (normalized)')
    plt.title('OFDM Ambiguity Function')
    plt.clim(-40, 0)

    # GFDM Ambiguity Function
    plt.subplot(1, 3, 2)
    plt.imshow(20 * np.log10(np.maximum(amb_gfdm, 1e-6)), extent=[delay_bins[0], delay_bins[-1], doppler_bins[0], doppler_bins[-1]], aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Delay (samples)')
    plt.ylabel('Doppler (normalized)')
    plt.title('GFDM Ambiguity Function')
    plt.clim(-40, 0)

    # FBMC Ambiguity Function
    plt.subplot(1, 3, 3)
    plt.imshow(20 * np.log10(np.maximum(amb_fbmc, 1e-6)), extent=[delay_bins[0], delay_bins[-1], doppler_bins[0], doppler_bins[-1]], aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Delay (samples)')
    plt.ylabel('Doppler (normalized)')
    plt.title('FBMC Ambiguity Function')
    plt.clim(-40, 0)

    plt.tight_layout()
    plt.show()

# Main loop for SNR
print("Starting simulation...")
for snrIdx, snr in enumerate(SNR_dB):
    print(f"Processing SNR = {snr} dB ({snrIdx+1}/{len(SNR_dB)})")

    ber_ofdm = 0
    ber_gfdm = 0
    ber_fbmc = 0

    # Initialize radar metrics for current SNR
    range_res_ofdm = 0
    range_res_gfdm = 0
    range_res_fbmc = 0

    doppler_res_ofdm = 0
    doppler_res_gfdm = 0
    doppler_res_fbmc = 0

    psl_ofdm = 0
    psl_gfdm = 0
    psl_fbmc = 0

    for iter in range(numIter):
        # Generate random data
        data = np.random.randint(0, M, (N, numSymbols))

        # QPSK Modulation
        modData = modem.modulate(data.flatten(order = 'F'))  # qpsk_modulate(data.flatten(), M)
        modData = modData.reshape(N, numSymbols, order = 'F')

        # OFDM
        ofdmSymbols = np.fft.ifft(modData, axis=0)
        cp_len = N // 4
        ofdmSymbols_with_cp = np.vstack([ofdmSymbols[-cp_len:], ofdmSymbols])  # Add CP
        ofdmSignal = ofdmSymbols_with_cp.flatten(order = 'F')

        # GFDM
        gfdmSymbols = gfdm_modulate(modData, N, M)
        gfdmSignal = gfdmSymbols.flatten(order = 'F')

        # FBMC
        fbmcSymbols = fbmc_modulate(modData, N, M)
        fbmcSignal = fbmcSymbols.flatten(order = 'F')

        # Pass through multipath channel
        ofdmRx = lfilter(channel, 1, ofdmSignal)
        gfdmRx = lfilter(channel, 1, gfdmSignal)
        fbmcRx = lfilter(channel, 1, fbmcSignal)

        # Add AWGN noise
        ofdmRx = awgn(ofdmRx, snr)
        gfdmRx = awgn(gfdmRx, snr)
        fbmcRx = awgn(fbmcRx, snr)

        # OFDM Receiver
        ofdmRx_reshaped = ofdmRx.reshape(N + cp_len, numSymbols, order = 'F')
        ofdmRx_no_cp = ofdmRx_reshaped[cp_len:]  # Remove CP
        ofdmRx_demod = np.fft.fft(ofdmRx_no_cp, axis=0)

        # GFDM Receiver
        gfdmRx_reshaped = gfdmRx.reshape(N, numSymbols, order = 'F')
        gfdmRx_demod = gfdm_demodulate(gfdmRx_reshaped, N, M)

        # FBMC Receiver
        fbmcRx_reshaped = fbmcRx.reshape(N, numSymbols, order = 'F')
        fbmcRx_demod = fbmc_demodulate(fbmcRx_reshaped, N, M)

        # QPSK Demodulation
        ofdmData = modem.demodulate(ofdmRx_demod.flatten(order = 'F'))  # qpsk_demodulate(ofdmRx_demod.flatten(), M)
        gfdmData = modem.demodulate(gfdmRx_demod.flatten(order = 'F'))  # qpsk_demodulate(gfdmRx_demod.flatten(), M)
        fbmcData = modem.demodulate(fbmcRx_demod.flatten(order = 'F'))  # qpsk_demodulate(fbmcRx_demod.flatten(), M)

        # Calculate BER
        ber_ofdm_iter = np.mean(data.flatten(order = 'F') != ofdmData)
        ber_gfdm_iter = np.mean(data.flatten(order = 'F') != gfdmData)
        ber_fbmc_iter = np.mean(data.flatten(order = 'F') != fbmcData)

        ber_ofdm += ber_ofdm_iter
        ber_gfdm += ber_gfdm_iter
        ber_fbmc += ber_fbmc_iter

        # Calculate Radar Ambiguity Functions
        amb_ofdm = calculate_ambiguity_function(ofdmSignal[:min(512, len(ofdmSignal))], delay_bins, doppler_bins)
        amb_gfdm = calculate_ambiguity_function(gfdmSignal[:min(512, len(gfdmSignal))], delay_bins, doppler_bins)
        amb_fbmc = calculate_ambiguity_function(fbmcSignal[:min(512, len(fbmcSignal))], delay_bins, doppler_bins)

        # Extract Radar Performance Metrics
        range_res_iter_ofdm, doppler_res_iter_ofdm, psl_iter_ofdm = extract_radar_metrics(amb_ofdm, delay_bins, doppler_bins)
        range_res_iter_gfdm, doppler_res_iter_gfdm, psl_iter_gfdm = extract_radar_metrics(amb_gfdm, delay_bins, doppler_bins)
        range_res_iter_fbmc, doppler_res_iter_fbmc, psl_iter_fbmc = extract_radar_metrics(amb_fbmc, delay_bins, doppler_bins)

        range_res_ofdm += range_res_iter_ofdm
        range_res_gfdm += range_res_iter_gfdm
        range_res_fbmc += range_res_iter_fbmc

        doppler_res_ofdm += doppler_res_iter_ofdm
        doppler_res_gfdm += doppler_res_iter_gfdm
        doppler_res_fbmc += doppler_res_iter_fbmc

        psl_ofdm += psl_iter_ofdm
        psl_gfdm += psl_iter_gfdm
        psl_fbmc += psl_iter_fbmc

    # Average BER
    BER_OFDM[snrIdx] = ber_ofdm / numIter
    BER_GFDM[snrIdx] = ber_gfdm / numIter
    BER_FBMC[snrIdx] = ber_fbmc / numIter

    # Average Radar Metrics
    range_resolution_OFDM[snrIdx] = range_res_ofdm / numIter
    range_resolution_GFDM[snrIdx] = range_res_gfdm / numIter
    range_resolution_FBMC[snrIdx] = range_res_fbmc / numIter

    doppler_resolution_OFDM[snrIdx] = doppler_res_ofdm / numIter
    doppler_resolution_GFDM[snrIdx] = doppler_res_gfdm / numIter
    doppler_resolution_FBMC[snrIdx] = doppler_res_fbmc / numIter

    PSL_OFDM[snrIdx] = psl_ofdm / numIter
    PSL_GFDM[snrIdx] = psl_gfdm / numIter
    PSL_FBMC[snrIdx] = psl_fbmc / numIter

print("Simulation completed!")

# Plot BER vs SNR
plt.figure(figsize=(10, 6))
plt.semilogy(SNR_dB, BER_OFDM, 'bo-', linewidth=2, label='OFDM')
plt.semilogy(SNR_dB, BER_GFDM, 'rs-', linewidth=2, label='GFDM')
plt.semilogy(SNR_dB, BER_FBMC, 'gd-', linewidth=2, label='FBMC')
plt.grid(True)
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.legend()
plt.title('BER Comparison of OFDM, GFDM, and FBMC under Multipath Fading')
plt.show()

# Plot Radar Performance Metrics
plt.figure(figsize=(12, 10))

# Range Resolution
plt.subplot(2, 2, 1)
plt.plot(SNR_dB, range_resolution_OFDM, 'bo-', linewidth=2, label='OFDM')
plt.plot(SNR_dB, range_resolution_GFDM, 'rs-', linewidth=2, label='GFDM')
plt.plot(SNR_dB, range_resolution_FBMC, 'gd-', linewidth=2, label='FBMC')
plt.grid(True)
plt.xlabel('SNR (dB)')
plt.ylabel('Range Resolution (samples)')
plt.legend()
plt.title('Range Resolution vs SNR')

# Doppler Resolution
plt.subplot(2, 2, 2)
plt.plot(SNR_dB, doppler_resolution_OFDM, 'bo-', linewidth=2, label='OFDM')
plt.plot(SNR_dB, doppler_resolution_GFDM, 'rs-', linewidth=2, label='GFDM')
plt.plot(SNR_dB, doppler_resolution_FBMC, 'gd-', linewidth=2, label='FBMC')
plt.grid(True)
plt.xlabel('SNR (dB)')
plt.ylabel('Doppler Resolution (normalized)')
plt.legend()
plt.title('Doppler Resolution vs SNR')

# Peak Sidelobe Level
plt.subplot(2, 2, 3)
plt.plot(SNR_dB, PSL_OFDM, 'bo-', linewidth=2, label='OFDM')
plt.plot(SNR_dB, PSL_GFDM, 'rs-', linewidth=2, label='GFDM')
plt.plot(SNR_dB, PSL_FBMC, 'gd-', linewidth=2, label='FBMC')
plt.grid(True)
plt.xlabel('SNR (dB)')
plt.ylabel('Peak Sidelobe Level (dB)')
plt.legend()
plt.title('Peak Sidelobe Level vs SNR')

plt.tight_layout()
plt.show()

# Plot Ambiguity Functions at high SNR
print("Plotting ambiguity functions...")
plot_ambiguity_functions(ofdmSignal, gfdmSignal, fbmcSignal, delay_bins, doppler_bins)

















