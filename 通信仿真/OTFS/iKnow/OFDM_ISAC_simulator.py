#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 01:23:01 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.constants import speed_of_light as c0
from scipy.signal import fftconvolve
import commpy
from Modulations import modulator

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
CPsize = int(N / 4)  # cyclic prefix length
# bitsPerSymbol = 2  # bits per symbol
# qam = 2 ** bitsPerSymbol  # 4-QAM modulation

# Transmit data
QAM_mod = 4
bps = int(np.log2(QAM_mod))
MOD_TYPE = "qam"
modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
map_table, demap_table = modem.getMappTable()
bits = np.random.randint(0, 2, size = M*N*bps).astype(np.int8)
TxData = modem.modulate(bits)
sym_int = np.array([demap_table[sym] for sym in TxData])
TxData = TxData.reshape(N, M)

# OFDM modulator
TxSignal = np.fft.ifft(TxData, axis=0)  # IFFT

# Add cyclic prefix
TxSignal_cp = np.vstack([TxSignal[-CPsize:, :], TxSignal])
TxSignal_cp = TxSignal_cp.T.flatten()  # time-domain transmit signal

#%% Communication channel
PowerdB = np.array([0, -8, -17, -21, -25])  # Channel tap power profile [dB]
Delay = np.array([0, 3, 5, 6, 8])           # Channel delay sample
Power = 10**(PowerdB/10)                    # Channel tap power profile
Ntap = len(PowerdB)                         # Channel tap number
Lch = Delay[-1] + 1                         # Channel length

# Rayleigh fading channel
channel = (np.random.randn(Ntap) + 1j * np.random.randn(Ntap)) * np.sqrt(Power/2)
h = np.zeros(Lch, dtype=complex)
h[Delay] = channel

ComSNRdB = 20  # SNR of communication channel

# Apply channel convolution
RxSignal = fftconvolve(TxSignal_cp, h, mode='full')
RxSignal = RxSignal[:len(TxSignal_cp)]  # Trim to original length
RxSignal = awgn(RxSignal, ComSNRdB, measured=True)  # add AWGN

### Communication receiver
RxSignal_reshaped = RxSignal.reshape(N + CPsize, M, order = 'F')
RxSignal_remove_cp = RxSignal_reshaped[CPsize:, :]  # remove CP
RxData = np.fft.fft(RxSignal_remove_cp, axis=0)  # FFT

# Perfect channel estimation
H_channel = np.fft.fft(np.concatenate([h, np.zeros(N - Lch)]))
H_channel = np.tile(H_channel[:, np.newaxis], (1, M))

# MMSE equalization
C = np.conj(H_channel) / (np.conj(H_channel) * H_channel + 10**(-ComSNRdB/10))
demodRxData = RxData * C

demod_bits = modem.demodulate(demodRxData.flatten(), 'hard')
sym_int_hat = []
for j in range(M*N):
    sym_int_hat.append( int(''.join([str(num) for num in demod_bits[j*bps:(j+1)*bps]]), base = 2) )
sym_int_hat = np.array(sym_int_hat)


# Error calculation
# data_bits = de2bi(data.flatten(), bitsPerSymbol)
# demod_bits = de2bi(demodRxData.flatten(), bitsPerSymbol)
errorCount = np.sum(bits != demod_bits)
comResult = f'Number of error bits: {errorCount}'
print(comResult)

#%% Radar channel
target_pos = 30  # target distance
target_delay = range2time(target_pos, c0_val)
target_speed = 20  # target velocity
target_dop = speed2dop(2 * target_speed, lambda_val)
RadarSNRdB =  20  # SNR of radar sensing channel
RadarSNR = 10**(RadarSNRdB/10)

### Radar receiver
### Received data in the frequency domain
RxData_radar = np.zeros((N, M), dtype=complex)

for kSubcarrier in range(N):
    for mSymbol in range(M):
        # Radar channel model with delay and Doppler
        phase_delay = np.exp(-1j * 2 * np.pi * fc * target_delay)
        phase_doppler = np.exp(1j * 2 * np.pi * mSymbol * Ts * target_dop)
        phase_subcarrier = np.exp(-1j * 2 * np.pi * kSubcarrier * target_delay * delta_f)
        signal_component = (np.sqrt(RadarSNR) * TxData[kSubcarrier, mSymbol] * phase_delay * phase_doppler * phase_subcarrier)
        noise_component = np.sqrt(0.5) * (np.random.randn() + 1j * np.random.randn())
        RxData_radar[kSubcarrier, mSymbol] = signal_component + noise_component

# Radar sensing algorithm (FFT)
dividerArray = RxData_radar / TxData

# Range estimation
NPer = 16 * N
normalizedPower = np.abs(np.fft.ifft(dividerArray, NPer, axis=0))
mean_normalizedPower = np.mean(normalizedPower, axis=1)
mean_normalizedPower = mean_normalizedPower / np.max(mean_normalizedPower)
mean_normalizedPower_dB = 10 * np.log10(mean_normalizedPower + 1e-10)  # Avoid log(0)

range_axis = np.arange(NPer) * c0_val / (2 * delta_f * NPer)
rangeEstimation = np.argmax(mean_normalizedPower_dB)
distanceE = rangeEstimation * c0_val / (2 * delta_f * NPer)  # estimated target range

# Velocity estimation
MPer = 128 * M
velocityProfile = np.abs(np.fft.fft(dividerArray, MPer, axis=1))
mean_velocityProfile = np.mean(velocityProfile, axis=0)
normalizedVelocityProfile = mean_velocityProfile / np.max(mean_velocityProfile)
normalizedVelocityProfile_dB = 10 * np.log10(normalizedVelocityProfile + 1e-10)
# Rearrange for symmetric velocity profile
velocityIndex = np.arange(-MPer//2, MPer//2) * c0_val / (2 * fc * Ts * 2*MPer)
velocityProfile_dB = np.concatenate([normalizedVelocityProfile_dB[MPer//2:], normalizedVelocityProfile_dB[:MPer//2]])
velocityEstimation = np.argmax(velocityProfile_dB)
velocityE = velocityIndex[velocityEstimation]  # estimated target velocity

# Display results
print(f'The estimated target range is {distanceE:.2f} m.')
print(f'The estimated target velocity is {velocityE:.2f} m/s.')

#%%
colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)

axs.plot(range_axis, mean_normalizedPower_dB, label='range')
axs.set_xlabel('Range [m]')
axs.set_ylabel('Normalized Range Profile [dB]')
axs.legend()
plt.title('Range Profile')
plt.show()
plt.close('all')


fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)
axs.plot(velocityIndex, velocityProfile_dB, label='periodogram')
axs.set_xlabel('Velocity (m/s)')
axs.set_ylabel('Normalized Velocity Profile [dB]')
axs.legend()
plt.title('Velocity Profile')
plt.show()
plt.close('all')




#%%





#%%



















































































