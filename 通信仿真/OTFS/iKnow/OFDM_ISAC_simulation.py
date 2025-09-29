#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 17:30:37 2025

@author: jack
Source code: https://github.com/YongzhiWu/OFDM_ISAC_simulator

OFDM_radar.py 是雷达感知部分代码，一帧，为了演示感知算法；
OFDM_ISAC_simulator.py 是通信+感知部分，一帧；
OFDM_ISAC_simulation.py 是通信+感知部分，蒙特卡洛仿真；

主要看 OFDM_ISAC_simulation.py

"""
import scipy
import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal
from scipy.constants import speed_of_light as c0
# from scipy.signal import fftconvolve
# import commpy
from Modulations import modulator

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

#%%
def Qfun(x):
    return 0.5 * scipy.special.erfc(x / np.sqrt(2))

def ser_awgn(EbN0dB, MOD_TYPE, M, COHERENCE = None):
    EbN0 = 10**(EbN0dB/10)
    EsN0 = np.log2(M) * EbN0
    SER = np.zeros(EbN0dB.size)
    if MOD_TYPE.lower() == "bpsk":
        SER = Qfun(np.sqrt(2 * EbN0))
    elif MOD_TYPE == "psk":
        if M == 2:
            SER = Qfun(np.sqrt(2 * EbN0))
        else:
            if M == 4:
                SER = 2 * Qfun(np.sqrt(2* EbN0)) - Qfun(np.sqrt(2 * EbN0))**2
            else:
                SER = 2 * Qfun(np.sin(np.pi/M) * np.sqrt(2 * EsN0))
    elif MOD_TYPE.lower() == "qam":
        SER = 1 - (1 - 2*(1 - 1/np.sqrt(M)) * Qfun(np.sqrt(3 * EsN0/(M - 1))))**2
    elif MOD_TYPE.lower() == "pam":
        SER = 2*(1-1/M) * Qfun(np.sqrt(6*EsN0/(M**2-1)))
    return SER

def ser_rayleigh(EbN0dB, MOD_TYPE, M):
    EbN0 = 10**(EbN0dB/10)
    EsN0 = np.log2(M) * EbN0
    SER = np.zeros(EbN0dB.size)
    if MOD_TYPE.lower() == "bpsk":
        SER = 1/2 * (1 - np.sqrt(EsN0/(1 + EsN0)))
    elif MOD_TYPE.lower() == "psk":
        SER = np.zeros(EsN0.size)
        for i in range(len(EsN0)):
            g = np.sin(np.pi/M)**2
            fun = lambda x: 1.0 / (1.0 + g * EsN0[i]/(np.sin(x)**2))
            SER[i] = 1/np.pi * scipy.integrate.quad(fun, 0, np.pi*(M-1)/M)[0]
    elif MOD_TYPE.lower() == "qam":
        SER = np.zeros(EsN0.size)
        for i in range(len(EsN0)):
            g = 1.5 / (M-1)
            fun = lambda x: 1.0 / (1.0 + g * EsN0[i]/np.sin(x)**2)
            SER[i] = 4/np.pi * (1 - 1/np.sqrt(M)) * scipy.integrate.quad(fun, 0, np.pi/2)[0] - 4/np.pi * (1 - 1/np.sqrt(M))**2 * scipy.integrate.quad(fun, 0, np.pi/4)[0]
    elif MOD_TYPE.lower() == "pam":
        SER = np.zeros(EsN0.size)
        for i in range(len(EsN0)):
            g = 3/(M**2 - 1)
            fun = lambda x: 1.0 / (1.0 + g * EsN0[i]/np.sin(x)**2)
            SER[i] = 2*(M-1)/(M*np.pi) * scipy.integrate.quad(fun, 0, np.pi/2)[0]
    return SER

def awgn(signal, snr_db, measured=True):
    """添加AWGN噪声"""
    if measured:
        signal_power = np.mean(np.abs(signal)**2)
    else:
        signal_power = 1.0  # 假设单位功率

    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

def range2time(range_val, c):
    """距离转换为时间"""
    return 2 * range_val / c

def speed2dop(speed, lambda_val):
    """速度转换为多普勒频率"""
    return 2 * speed / lambda_val

# %% ISAC Transmitter
# System parameters
c0_val = c0  # speed of light
fc = 30e9  # carrier frequency
lambda_val = c0_val / fc  # wavelength
N = 256  # number of subcarriers
M = 16  # number of symbols
delta_f = 15e3 * 2**6  # subcarrier spacing
T = 1 / delta_f  # symbol duration
Tcp = T / 4  # cyclic prefix duration
Ts = T + Tcp  # total symbol duration
L = 10
CPsize = int(N / 4)  # cyclic prefix length
# bitsPerSymbol = 2  # bits per symbol
# qam = 2 ** bitsPerSymbol  # 4-QAM modulation

# Transmit data
QAM_mod = 4
bps = int(np.log2(QAM_mod))
MOD_TYPE = "qam"
modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
map_table, demap_table = modem.getMappTable()

#%% Comm Part
EbN0dBs = np.arange(0, 30, 5)
EsN0dBs = 10*np.log10(bps*N/(N + CPsize)) + EbN0dBs # 10 * np.log10(bps) + EbN0dBs
nFrame = 500
SER_sim = np.zeros(len(EbN0dBs))

for idx, snr in enumerate(EsN0dBs):
    print(f"{idx+1}/{EsN0dBs.size}")
    errors = 0
    total_symbols = 0
    # sigma2 = 10 ** (-snr / 10)  # 噪声方差
    for t in range(nFrame):
        bits = np.random.randint(0, 2, size = M*N*bps).astype(np.int8)
        TxData = modem.modulate(bits)
        sym_int = np.array([demap_table[sym] for sym in TxData])
        TxData = TxData.reshape(N, M)

        # OFDM modulator
        TxSignal = np.fft.ifft(TxData, axis=0)  # IFFT

        # Add cyclic prefix
        TxSignal_cp = np.vstack([TxSignal[-CPsize:, :], TxSignal])
        TxSignal_cp = TxSignal_cp.T.flatten()  # time-domain transmit signal

        ### Communication channel
        # PowerdB = np.array([0, -8, -17, -21, -25])  # Channel tap power profile [dB]
        # Delay = np.array([0, 3, 5, 6, 8])           # Channel delay sample
        # Power = 10**(PowerdB/10)                    # Channel tap power profile
        # Ntap = len(PowerdB)                         # Channel tap number
        # Lch = Delay[-1] + 1                         # Channel length
        # # Rayleigh fading channel
        # channel = (np.random.randn(Ntap) + 1j * np.random.randn(Ntap)) * np.sqrt(Power/2)
        # h = np.zeros(Lch, dtype=complex)
        # h[Delay] = channel

        ## 下面这样的简单生成h时仿真性能可以与理论完美对上
        h = (np.random.randn(L) + 1j * np.random.randn(L))/np.sqrt(2)
        # h = h/np.linalg.norm(h)

        # Perfect channel estimation
        H_channel = np.fft.fft(h, n = N)
        H_channel = np.tile(H_channel[:, np.newaxis], (1, M))

        # Apply channel convolution
        RxSignal = scipy.signal.convolve(h, TxSignal_cp) # fftconvolve(TxSignal_cp, h, mode='full')
        RxSignal = RxSignal[:len(TxSignal_cp)]  # Trim to original length
        RxSignal = awgn(RxSignal, snr, measured=True)  # add AWGN

        ### Communication receiver
        RxSignal_reshaped = RxSignal.reshape(N + CPsize, M, order = 'F')
        RxSignal_remove_cp = RxSignal_reshaped[CPsize:, :]  # remove CP
        RxData = np.fft.fft(RxSignal_remove_cp, axis=0)  # FFT

        # C = np.conj(H_channel) / (np.conj(H_channel) * H_channel + 10**(-snr/10))
        # demodRxData = RxData * C
        ## Equalize in freq
        demodRxData = RxData / H_channel

        demod_bits = modem.demodulate(demodRxData.flatten(), 'hard')
        sym_int_hat = []
        for j in range(M*N):
            sym_int_hat.append( int(''.join([str(num) for num in demod_bits[j*bps:(j+1)*bps]]), base = 2) )
        sym_int_hat = np.array(sym_int_hat)

        errors += np.sum(sym_int_hat != sym_int)
        total_symbols += sym_int.size
    SER_sim[idx] = errors / total_symbols

SER_theory = ser_rayleigh(EbN0dBs, MOD_TYPE, QAM_mod)
colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)

axs.semilogy(EbN0dBs, SER_sim, 'bo-', label='OFDM (仿真)')
axs.semilogy(EbN0dBs, SER_theory, 'r*--', label='OFDM (理论)')

axs.set_xlabel('Eb/N0 (dB)')
axs.set_ylabel('SER')
axs.legend()

plt.title('OFDM-QPSK 符号错误率性能')
plt.show()
plt.close('all')

#%% Sensing Part
target_pos = 30  # target distance
target_delay = range2time(target_pos, c0_val)
target_speed = 20  # target velocity
target_dop = speed2dop(2 * target_speed, lambda_val)
RadarSNRdB =  20  # SNR of radar sensing channel
RadarSNR = 10**(RadarSNRdB/10)

SNRdBs = np.arange(-20, 10, 2)
nFrame = 1000
RmseList = np.zeros(SNRdBs.size)

bits = np.random.randint(0, 2, size = M*N*bps).astype(np.int8)
TxData = modem.modulate(bits)
sym_int = np.array([demap_table[sym] for sym in TxData])
TxData = TxData.reshape(N, M)

for idx, radarsnr in enumerate(SNRdBs):
    print(f"{idx+1}/{SNRdBs.size}")
    mse = 0
    sigma2 = 10**(-radarsnr/10)
    for t in range(nFrame):
        RxData_radar = np.zeros((N, M), dtype=complex)
        for kSubcarrier in range(N):
            for mSymbol in range(M):
                # Radar channel model with delay and Doppler
                phase_delay = np.exp(-1j * 2 * np.pi * fc * target_delay)
                phase_doppler = np.exp(1j * 2 * np.pi * mSymbol * Ts * target_dop)
                phase_subcarrier = np.exp(-1j * 2 * np.pi * kSubcarrier * target_delay * delta_f)
                signal_component =  TxData[kSubcarrier, mSymbol] * phase_delay * phase_doppler * phase_subcarrier
                noise_component = np.sqrt(sigma2/2.0) * (np.random.randn() + 1j * np.random.randn())
                RxData_radar[kSubcarrier, mSymbol] = signal_component + noise_component

        # Radar sensing algorithm (FFT)
        dividerArray = RxData_radar / TxData

        # Range estimation
        NPer = 16 * N
        normalizedPower = np.abs(np.fft.ifft(dividerArray, NPer, axis=0))
        mean_normalizedPower = np.mean(normalizedPower, axis=1)
        mean_normalizedPower = mean_normalizedPower / np.max(mean_normalizedPower)
        mean_normalizedPower_dB = 10 * np.log10(mean_normalizedPower + 1e-10)  # Avoid log(0)
        # range_axis = np.arange(NPer) * c0_val / (2 * delta_f * NPer)
        rangeEstimation = np.argmax(mean_normalizedPower_dB)
        distanceE = rangeEstimation * c0_val / (2 * delta_f * NPer)  # estimated target range
        mse += (target_pos - distanceE)**2
    RmseList[idx] = np.sqrt(mse/nFrame)

colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)
axs.plot(SNRdBs, RmseList, 'bo-', label='RMSE')
axs.set_xlabel('SNR (dB)')
axs.set_ylabel('RMSE')
axs.legend()
plt.title('RMSE')
plt.show()
plt.close('all')







































































